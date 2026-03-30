import uuid
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@patch("app.api.websocket.ideation_node")
@patch("app.api.websocket.synthesis_node")
@patch("app.api.websocket.innovation_extractor_node")
@patch("app.api.websocket.patent_fetcher_node")
@patch("app.api.websocket.get_analysis")
def test_websocket_connects_and_receives_complete(
    mock_get, mock_fetcher, mock_extractor, mock_synthesis, mock_ideation
):
    test_id = uuid.uuid4()
    mock_get.return_value = MagicMock(
        id=test_id, status="pending", search_query="Lithium Battery",
        previous_session_id=None
    )
    # Each node returns minimal state updates
    mock_fetcher.return_value = {"patents": [], "status_updates": [
        {"type": "status", "agent": "patent_fetcher", "message": "Done", "step": 1}
    ]}
    mock_extractor.return_value = {"innovations": [], "status_updates": [
        {"type": "status", "agent": "patent_fetcher", "message": "Done", "step": 1},
        {"type": "status", "agent": "innovation_extractor", "message": "Done", "step": 2}
    ]}
    mock_synthesis.return_value = {"synthesis": {}, "status_updates": [
        {"type": "status", "agent": "patent_fetcher", "message": "Done", "step": 1},
        {"type": "status", "agent": "innovation_extractor", "message": "Done", "step": 2},
        {"type": "status", "agent": "synthesis", "message": "Done", "step": 3}
    ]}
    mock_ideation.return_value = {"generated_ideas": [], "status_updates": [
        {"type": "status", "agent": "patent_fetcher", "message": "Done", "step": 1},
        {"type": "status", "agent": "innovation_extractor", "message": "Done", "step": 2},
        {"type": "status", "agent": "synthesis", "message": "Done", "step": 3},
        {"type": "status", "agent": "ideation", "message": "Done", "step": 4}
    ]}

    with client.websocket_connect(f"/ws/analysis/{test_id}") as websocket:
        # Collect all messages until complete
        messages = []
        while True:
            data = websocket.receive_json()
            messages.append(data)
            if data["type"] == "complete":
                break
        assert messages[-1]["type"] == "complete"
        assert messages[-1]["session_id"] == str(test_id)
        # Should have received status updates before complete
        status_msgs = [m for m in messages if m["type"] == "status"]
        assert len(status_msgs) > 0


@patch("app.api.websocket.get_analysis")
def test_websocket_returns_error_for_missing_analysis(mock_get):
    mock_get.return_value = None
    test_id = uuid.uuid4()
    with client.websocket_connect(f"/ws/analysis/{test_id}") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "error"
