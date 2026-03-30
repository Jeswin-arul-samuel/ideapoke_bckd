import io
import logging
import fitz  # PyMuPDF
import requests

logger = logging.getLogger(__name__)

def download_pdf(url: str) -> str | None:
    """Download a PDF from URL and extract text. Returns None on failure."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text if text.strip() else None
    except Exception as e:
        logger.warning(f"Failed to download/process PDF {url}: {e}")
        return None

def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 500) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
