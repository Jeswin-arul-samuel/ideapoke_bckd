from unittest.mock import patch, MagicMock
from app.tools.embedding import get_embedding, get_embeddings_batch


@patch("app.tools.embedding._server_client")
def test_get_embedding_returns_vector(mock_client):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_response
    result = get_embedding("test text")
    assert len(result) == 1536
    assert result[0] == 0.1


@patch("app.tools.embedding._server_client")
def test_get_embeddings_batch(mock_client):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
    mock_client.embeddings.create.return_value = mock_response
    results = get_embeddings_batch(["text 1", "text 2"])
    assert len(results) == 2
    assert len(results[0]) == 1536
