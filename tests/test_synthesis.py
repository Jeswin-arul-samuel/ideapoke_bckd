from unittest.mock import patch, MagicMock
from app.agents.synthesis import synthesis_node
from app.agents.state import AgentState


@patch("app.agents.synthesis.search_innovations_by_relevance")
@patch("app.agents.synthesis.create_agent")
def test_synthesis_node_produces_landscape(mock_create_agent, mock_search):
    # Mock the tool invocation
    mock_search.return_value = "- Solid state electrolyte (tech: ceramics; solves: safety)"

    # Mock the agent with structured response
    mock_structured = MagicMock()
    mock_structured.model_dump.return_value = {
        "patterns": ["solid-state trend"],
        "gaps": ["recycling"],
        "convergence": ["AI + battery"],
        "trends": ["solid-state accelerating"],
    }
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"structured_response": mock_structured, "messages": []}
    mock_create_agent.return_value = mock_agent

    state: AgentState = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "search_query": "Lithium Battery",
        "patents": [],
        "innovations": [],
        "synthesis": {},
        "generated_ideas": [],
        "status_updates": [],
    }

    result = synthesis_node(state)
    assert "synthesis" in result
    assert result["synthesis"] is not None
    assert "patterns" in result["synthesis"]
    assert any(u["agent"] == "synthesis" for u in result["status_updates"])
