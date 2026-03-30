from unittest.mock import patch, MagicMock
from app.agents.patent_fetcher import patent_fetcher_node
from app.agents.state import AgentState


@patch("app.agents.patent_fetcher.SessionLocal")
@patch("app.agents.patent_fetcher.update_analysis_status")
@patch("app.agents.patent_fetcher.save_patents")
@patch("app.agents.patent_fetcher.search_patents")
def test_patent_fetcher_node_returns_patents(mock_search, mock_save, mock_update, mock_session):
    mock_search.return_value = [
        {"title": "Patent A", "publication_number": "US001", "search_query": "Lithium Battery"},
        {"title": "Patent B", "publication_number": "US002", "search_query": "Lithium Battery"},
    ]
    mock_save.return_value = [MagicMock(id=1), MagicMock(id=2)]
    state: AgentState = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "search_query": "Lithium Battery",
        "patents": [], "innovations": [], "synthesis": {},
        "generated_ideas": [], "status_updates": [],
    }
    result = patent_fetcher_node(state)
    assert len(result["patents"]) == 2
    assert any(u["agent"] == "patent_fetcher" for u in result["status_updates"])


@patch("app.agents.patent_fetcher.SessionLocal")
@patch("app.agents.patent_fetcher.update_analysis_status")
@patch("app.agents.patent_fetcher.save_patents")
@patch("app.agents.patent_fetcher.search_patents")
def test_patent_fetcher_node_handles_empty(mock_search, mock_save, mock_update, mock_session):
    mock_search.return_value = []
    mock_save.return_value = []
    state: AgentState = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "search_query": "NonexistentTech123",
        "patents": [], "innovations": [], "synthesis": {},
        "generated_ideas": [], "status_updates": [],
    }
    result = patent_fetcher_node(state)
    assert result["patents"] == []
