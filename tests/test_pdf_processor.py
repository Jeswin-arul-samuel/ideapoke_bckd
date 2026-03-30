from unittest.mock import patch, MagicMock
from app.tools.pdf_processor import download_pdf, chunk_text

def test_chunk_text_splits_correctly():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 600

def test_chunk_text_small_input():
    text = "This is a short text."
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text

@patch("app.tools.pdf_processor.requests.get")
@patch("app.tools.pdf_processor.fitz.open")
def test_download_pdf_returns_text(mock_fitz_open, mock_get):
    mock_response = MagicMock()
    mock_response.content = b"fake pdf bytes"
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Patent text content here"
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([mock_page])
    mock_doc.__enter__ = lambda self: self
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_fitz_open.return_value = mock_doc
    text = download_pdf("https://example.com/patent.pdf")
    assert text == "Patent text content here"

@patch("app.tools.pdf_processor.requests.get")
def test_download_pdf_returns_none_on_failure(mock_get):
    mock_get.side_effect = Exception("Connection error")
    text = download_pdf("https://example.com/bad.pdf")
    assert text is None
