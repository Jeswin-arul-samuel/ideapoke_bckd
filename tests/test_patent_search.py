from unittest.mock import patch, MagicMock
from app.tools.patent_search import search_patents

MOCK_SERP_RESPONSE = {
    "organic_results": [
        {
            "title": "Lithium Battery Improvement",
            "publication_number": "US20240001234",
            "patent_link": "https://patents.google.com/patent/US20240001234",
            "priority_date": "2023-01-15",
            "filing_date": "2023-06-20",
            "publication_date": "2024-01-01",
            "inventor": "John Doe",
            "assignee": "Battery Corp",
            "snippet": "A method for improving lithium battery...",
            "pdf": "https://patentimages.storage.googleapis.com/US20240001234.pdf",
        }
    ]
}

@patch("app.tools.patent_search.requests.get")
def test_search_patents_returns_structured_data(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SERP_RESPONSE
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    results = search_patents("Lithium Battery", max_pages=1)
    assert len(results) == 1
    assert results[0]["title"] == "Lithium Battery Improvement"
    assert results[0]["publication_number"] == "US20240001234"

@patch("app.tools.patent_search.requests.get")
def test_search_patents_handles_empty_results(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"organic_results": []}
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    results = search_patents("NonexistentTech", max_pages=1)
    assert results == []
