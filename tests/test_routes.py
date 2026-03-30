import uuid
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.api.routes.get_db")
@patch("app.api.routes.create_analysis")
def test_create_analysis_endpoint(mock_create, mock_db):
    mock_session = MagicMock()
    mock_db.return_value = iter([mock_session])
    test_id = uuid.uuid4()
    mock_create.return_value = MagicMock(id=test_id)
    response = client.post("/api/analysis", json={"query": "Lithium Battery"})
    assert response.status_code == 200
    assert response.json()["session_id"] == str(test_id)

@patch("app.api.routes.get_db")
@patch("app.api.routes.list_analyses")
def test_list_analyses_endpoint(mock_list, mock_db):
    mock_session = MagicMock()
    mock_db.return_value = iter([mock_session])
    mock_list.return_value = []
    response = client.get("/api/analyses")
    assert response.status_code == 200
    assert response.json() == []

@patch("app.api.routes.get_db")
@patch("app.api.routes.get_analysis")
def test_get_analysis_not_found(mock_get, mock_db):
    mock_session = MagicMock()
    mock_db.return_value = iter([mock_session])
    mock_get.return_value = None
    response = client.get(f"/api/analysis/{uuid.uuid4()}")
    assert response.status_code == 404

@patch("app.api.routes.resolve_llm_config")
@patch("app.api.routes.get_analysis")
@patch("app.api.routes.get_db")
def test_followup_endpoint_streams(mock_db, mock_get, mock_resolve):
    mock_session = MagicMock()
    mock_db.return_value = iter([mock_session])
    test_id = uuid.uuid4()
    mock_get.return_value = MagicMock(
        id=test_id, search_query="Lithium Battery",
        synthesis={"patterns": ["solid-state"]},
        generated_ideas=[{"title": "Idea 1", "explanation": "Test", "patent_trail": []}],
    )
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter([
        MagicMock(content="This"), MagicMock(content=" is"), MagicMock(content=" a response"),
    ])
    mock_config = MagicMock()
    mock_config.chat_model.return_value = mock_llm
    mock_resolve.return_value = mock_config

    response = client.post(
        "/api/followup",
        json={"session_id": str(test_id), "question": "Tell me more about idea 1"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
