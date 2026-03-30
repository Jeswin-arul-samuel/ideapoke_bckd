from typing import TypedDict


class AgentState(TypedDict, total=False):
    session_id: str
    search_query: str
    patents: list[dict]
    innovations: list[dict]
    synthesis: dict
    generated_ideas: list[dict]
    status_updates: list[dict]
    previous_analysis: dict | None
    llm_provider: str        # "openai" or "groq"
    llm_api_key: str         # the user's LLM API key
    llm_models: dict         # {"extraction": "...", "synthesis": "...", ...}
    embedding_api_key: str   # OpenAI key for embeddings (user's or server's)
