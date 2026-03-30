import json
import logging
import os
import time
from uuid import UUID

from langchain.agents import create_agent
from pydantic import BaseModel, Field
from poml.integration.langchain import LangchainPomlTemplate
from sqlalchemy import text

from app.agents.state import AgentState
from app.config import settings
from app.database import SessionLocal
from app.tools.embedding import get_embedding

MAX_RETRIES = 3

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


# --- Structured output ---

class LandscapeSynthesis(BaseModel):
    """Synthesis of the patent innovation landscape."""
    patterns: list[str] = Field(description="Themes or directions multiple patents converge on")
    gaps: list[str] = Field(description="Areas underexplored or missing from the landscape")
    convergence: list[str] = Field(description="Where different technologies are meeting or combining")
    trends: list[str] = Field(description="What is accelerating and what is declining")


# --- Tools ---

def search_innovations_by_relevance(session_id: str, query: str, api_key: str | None = None, limit: int = 50) -> str:
    """Search innovations by semantic relevance using pgvector similarity search."""
    query_embedding = get_embedding(query, api_key=api_key)
    db = SessionLocal()
    try:
        result = db.execute(
            text("""
                SELECT innovation_summary, technology_used, problem_solved,
                       embedding <=> :query_embedding AS distance
                FROM innovations
                WHERE session_id = :session_id AND embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT :limit
            """),
            {"query_embedding": str(query_embedding), "session_id": session_id, "limit": limit},
        )
        rows = result.fetchall()
        innovations = []
        for row in rows:
            tech = row[1] or []
            innovations.append(
                f"- {row[0]} (tech: {', '.join(tech)}; solves: {row[2] or 'N/A'})"
            )
        return "\n".join(innovations) if innovations else "No innovations found."
    finally:
        db.close()


def synthesis_node(state: AgentState) -> dict:
    """LangGraph node: Synthesize the innovation landscape using pgvector-ranked innovations."""
    query = state["search_query"]
    session_id = state["session_id"]
    status_updates = list(state.get("status_updates", []))

    status_updates.append({
        "type": "status",
        "agent": "synthesis",
        "message": "Synthesizing innovation landscape...",
        "step": 3,
    })

    # Resolve model from state
    provider = state.get("llm_provider", "openai")
    models = state.get("llm_models", {"synthesis": settings.SYNTHESIS_MODEL})
    agent_model = f"{provider}:{models['synthesis']}"

    # Get ranked innovations via pgvector similarity search
    embedding_key = state.get("embedding_api_key")
    innovations_text = search_innovations_by_relevance(
        session_id=session_id, query=query, api_key=embedding_key
    )

    # Load POML template and create agent with structured output
    prompt_template = LangchainPomlTemplate.from_file(
        os.path.join(PROMPTS_DIR, "synthesis.poml")
    )

    messages = prompt_template.format(
        query=query, innovations_text=innovations_text
    ).to_messages()

    synthesis = {"patterns": [], "gaps": [], "convergence": [], "trends": []}
    for attempt in range(MAX_RETRIES):
        try:
            agent = create_agent(
                model=agent_model,
                tools=[],
                response_format=LandscapeSynthesis,
            )
            result = agent.invoke({"messages": messages})
            structured = result.get("structured_response")
            if structured:
                synthesis = structured.model_dump()
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                wait = (attempt + 1) * 8
                logger.warning(f"Synthesis rate limited, waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            elif "tool_use_failed" in error_str or "400" in error_str:
                logger.warning(f"Synthesis structured output failed, retrying (attempt {attempt + 1})")
                time.sleep(2)
            else:
                logger.error(f"Synthesis failed: {e}")
                break

    status_updates.append({
        "type": "status",
        "agent": "synthesis",
        "message": "Landscape synthesis complete",
        "step": 3,
    })

    return {"synthesis": synthesis, "status_updates": status_updates}
