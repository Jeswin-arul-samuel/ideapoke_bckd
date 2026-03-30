from unittest.mock import patch, MagicMock
from app.agents.ideation import ideation_node
from app.agents.state import AgentState


@patch("app.agents.ideation.SessionLocal")
@patch("app.agents.ideation.create_agent")
@patch("app.agents.ideation.update_analysis_results")
def test_ideation_node_generates_ideas(mock_update, mock_create_agent, mock_session):
    # Mock structured response with 2 ideas
    mock_idea_1 = MagicMock()
    mock_idea_1.model_dump.return_value = {
        "title": "Self-healing battery membranes",
        "explanation": "Bio-inspired polymer membranes.",
        "patent_trail": ["US20240001234"],
    }
    mock_idea_2 = MagicMock()
    mock_idea_2.model_dump.return_value = {
        "title": "AI-optimized electrolyte",
        "explanation": "ML system for electrolyte ratios.",
        "patent_trail": ["US20240009999"],
    }
    mock_structured = MagicMock()
    mock_structured.ideas = [mock_idea_1, mock_idea_2]

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"structured_response": mock_structured, "messages": []}
    mock_create_agent.return_value = mock_agent

    state: AgentState = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "search_query": "Lithium Battery",
        "patents": [],
        "innovations": [
            {"innovation_summary": "Solid state electrolyte", "technology_used": ["ceramics"], "problem_solved": "safety"}
        ],
        "synthesis": {"patterns": ["solid-state"], "gaps": ["recycling"]},
        "generated_ideas": [],
        "status_updates": [],
    }

    result = ideation_node(state)
    assert len(result["generated_ideas"]) == 2
    assert result["generated_ideas"][0]["title"] == "Self-healing battery membranes"
    assert any(u["agent"] == "ideation" for u in result["status_updates"])
