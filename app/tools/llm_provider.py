"""LLM provider management — switches between OpenAI and Groq based on user-provided API keys."""

import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from app.config import settings

# Default Groq model mappings (Groq uses different model names)
GROQ_MODELS = {
    "extraction": "llama-3.3-70b-versatile",  # 8b too small for structured output
    "synthesis": "llama-3.3-70b-versatile",
    "ideation": "llama-3.3-70b-versatile",
    "followup": "llama-3.3-70b-versatile",
}

OPENAI_MODELS = {
    "extraction": settings.EXTRACTION_MODEL,
    "synthesis": settings.SYNTHESIS_MODEL,
    "ideation": settings.IDEATION_MODEL,
    "followup": settings.FOLLOWUP_MODEL,
}


@dataclass
class LLMConfig:
    """Holds the resolved provider info for a single request."""
    provider: str  # "openai" or "groq"
    api_key: str
    models: dict  # {"extraction": "...", "synthesis": "...", ...}
    embedding_api_key: str = ""  # Always an OpenAI key (user's or server's)

    def agent_model(self, task: str) -> str:
        """Returns the model string for create_agent (e.g., 'openai:gpt-4.1-mini')."""
        return f"{self.provider}:{self.models[task]}"

    def chat_model(self, task: str) -> ChatOpenAI:
        """Returns a ChatOpenAI (or compatible) instance for streaming."""
        model_name = self.models[task]
        if self.provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model_name, api_key=self.api_key, temperature=0.5)
        else:
            return ChatOpenAI(model=model_name, api_key=self.api_key, temperature=0.5)


def resolve_llm_config(openai_key: str | None = None, groq_key: str | None = None) -> LLMConfig:
    """Determine which provider to use based on provided keys.

    Priority: user-provided keys > server .env keys.
    If user provides a Groq key, use Groq. If OpenAI key, use OpenAI.
    Falls back to server's .env OpenAI key if no user key provided.
    """
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        # Groq for LLM, server's OpenAI key for embeddings
        return LLMConfig(
            provider="groq", api_key=groq_key, models=GROQ_MODELS,
            embedding_api_key=settings.OPENAI_API_KEY,
        )

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        # User's OpenAI key for everything including embeddings
        return LLMConfig(
            provider="openai", api_key=openai_key, models=OPENAI_MODELS,
            embedding_api_key=openai_key,
        )

    # Fallback to server's own key for everything
    return LLMConfig(
        provider="openai", api_key=settings.OPENAI_API_KEY, models=OPENAI_MODELS,
        embedding_api_key=settings.OPENAI_API_KEY,
    )
