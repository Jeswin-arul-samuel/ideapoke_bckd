from unittest.mock import patch, MagicMock
from app.agents.innovation_extractor import innovation_extractor_node
from app.agents.state import AgentState

@patch("app.agents.innovation_extractor.SessionLocal")
@patch("app.agents.innovation_extractor.save_innovations")
@patch("app.agents.innovation_extractor.get_embeddings_batch")
@patch("app.agents.innovation_extractor.extract_innovations_from_text")
@patch("app.agents.innovation_extractor.chunk_text")
@patch("app.agents.innovation_extractor.download_pdf")
def test_extractor_processes_patents(mock_download, mock_chunk, mock_extract, mock_embed, mock_save, mock_session):
    mock_download.return_value = "Patent text content about batteries"
    mock_chunk.return_value = ["chunk 1 text", "chunk 2 text"]
    mock_extract.return_value = [
        {"innovation_summary": "Novel cathode", "technology_used": ["nanomaterials"], "problem_solved": "energy density"}
    ]
    mock_embed.return_value = [[0.1] * 1536]
    mock_save.return_value = [MagicMock()]
    state: AgentState = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "search_query": "Lithium Battery",
        "patents": [{"pdf_url": "https://example.com/p.pdf", "publication_number": "US001", "db_id": 1}],
        "innovations": [], "synthesis": {}, "generated_ideas": [], "status_updates": [],
    }
    result = innovation_extractor_node(state)
    assert len(result["innovations"]) > 0
    assert any(u["agent"] == "innovation_extractor" for u in result["status_updates"])

@patch("app.agents.innovation_extractor.SessionLocal")
@patch("app.agents.innovation_extractor.save_innovations")
@patch("app.agents.innovation_extractor.download_pdf")
def test_extractor_skips_failed_pdfs(mock_download, mock_save, mock_session):
    mock_download.return_value = None
    mock_save.return_value = []
    state: AgentState = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "search_query": "Lithium Battery",
        "patents": [{"pdf_url": "https://example.com/bad.pdf", "publication_number": "US001", "db_id": 1}],
        "innovations": [], "synthesis": {}, "generated_ideas": [], "status_updates": [],
    }
    result = innovation_extractor_node(state)
    assert "innovations" in result
