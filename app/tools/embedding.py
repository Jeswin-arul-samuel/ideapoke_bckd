import logging

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = settings.EMBEDDING_MODEL

# Server's own client — used as fallback for Groq users
_server_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def _get_client(api_key: str | None = None) -> OpenAI:
    """Return an OpenAI client — user's key if provided, server's key otherwise."""
    if api_key and api_key != settings.OPENAI_API_KEY:
        return OpenAI(api_key=api_key)
    return _server_client


def get_embedding(text: str, api_key: str | None = None) -> list[float]:
    """Get embedding for a single text."""
    client = _get_client(api_key)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    client = _get_client(api_key)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]
