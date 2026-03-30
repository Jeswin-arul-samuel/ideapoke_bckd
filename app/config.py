import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ BEFORE anything else reads env vars
# This ensures all libraries (OpenAI SDK, LangChain, LangSmith) can find them
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    DATABASE_URL: str = ""
    OPENAI_API_KEY: str = ""
    SERP_API_KEY: str = ""
    LENS_API_KEY: str = ""

    # LLM Models — change these to switch models across the app
    EXTRACTION_MODEL: str = "gpt-4.1-nano"
    SYNTHESIS_MODEL: str = "gpt-4.1-mini"
    IDEATION_MODEL: str = "gpt-4.1-mini"
    FOLLOWUP_MODEL: str = "gpt-4.1-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Pipeline parameters
    MAX_TOTAL_PATENTS: int = 15
    MAX_PARALLEL_OPENAI: int = 5
    MAX_PARALLEL_GROQ: int = 2

    # LangSmith tracing
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "ideapoke"


settings = Settings()
