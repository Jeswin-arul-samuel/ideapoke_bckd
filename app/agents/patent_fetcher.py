import logging
import os
import time
from uuid import UUID

from langchain.agents import create_agent
from pydantic import BaseModel, Field
from poml.integration.langchain import LangchainPomlTemplate

from app.agents.state import AgentState
from app.config import settings
from app.database import SessionLocal
from app.tools.db_tools import save_patents, update_analysis_status
from app.tools.patent_search import search_patents

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
MAX_TOTAL_PATENTS = settings.MAX_TOTAL_PATENTS
MAX_RETRIES = 3


# --- Structured output for query decomposition ---

class QueryDecomposition(BaseModel):
    """Decomposed search query into patent-searchable sub-topics."""
    sub_topics: list[str] = Field(description="1-3 concrete, searchable sub-topic strings")
    reasoning: str = Field(description="Why these sub-topics were chosen")


def decompose_query(query: str, provider: str, models: dict) -> list[str]:
    """Use LLM to break an abstract query into searchable patent sub-topics."""
    prompt_template = LangchainPomlTemplate.from_file(
        os.path.join(PROMPTS_DIR, "query_decomposer.poml")
    )
    messages = prompt_template.format(query=query).to_messages()

    model_name = models.get("extraction", settings.EXTRACTION_MODEL)
    agent_model = f"{provider}:{model_name}"

    for attempt in range(MAX_RETRIES):
        try:
            agent = create_agent(
                model=agent_model,
                tools=[],
                response_format=QueryDecomposition,
            )
            result = agent.invoke({"messages": messages})
            structured = result.get("structured_response")
            if structured and structured.sub_topics:
                logger.info(f"Query '{query}' decomposed into: {structured.sub_topics} ({structured.reasoning})")
                return structured.sub_topics
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                time.sleep((attempt + 1) * 8)
            elif "tool_use_failed" in error_str or "400" in error_str:
                time.sleep(2)
            else:
                logger.warning(f"Query decomposition failed: {e}")
                break

    # Fallback: use original query as-is
    return [query]


def patent_fetcher_node(state: AgentState) -> dict:
    """LangGraph node: Search patents directly first, decompose only if no results."""
    session_id = state["session_id"]
    query = state["search_query"]
    status_updates = list(state.get("status_updates", []))
    provider = state.get("llm_provider", "openai")
    models = state.get("llm_models", {"extraction": settings.EXTRACTION_MODEL})

    # Step 1: Try the original query directly
    status_updates.append({
        "type": "status", "agent": "patent_fetcher",
        "message": f"Searching patents for '{query}'...", "step": 1,
    })

    all_patents = search_patents(query, max_pages=2)

    # Step 2: If no results, decompose the query into sub-topics and try those
    if len(all_patents) == 0:
        status_updates.append({
            "type": "status", "agent": "patent_fetcher",
            "message": f"No direct results. Breaking down '{query}' into searchable sub-topics...",
            "step": 1,
        })

        sub_topics = decompose_query(query, provider, models)
        patents_per_topic = max(5, MAX_TOTAL_PATENTS // len(sub_topics))
        pages_per_topic = max(1, patents_per_topic // 10 + 1)

        status_updates.append({
            "type": "status", "agent": "patent_fetcher",
            "message": f"Searching {len(sub_topics)} sub-topic(s): {', '.join(sub_topics)}",
            "step": 1,
        })

        seen_pub_numbers = set()
        for topic in sub_topics:
            results = search_patents(topic, max_pages=pages_per_topic)
            for patent in results:
                pub_num = patent.get("publication_number", "")
                if pub_num and pub_num in seen_pub_numbers:
                    continue
                seen_pub_numbers.add(pub_num)
                all_patents.append(patent)
                if len(all_patents) >= MAX_TOTAL_PATENTS:
                    break
            if len(all_patents) >= MAX_TOTAL_PATENTS:
                break

    # Cap at MAX_TOTAL_PATENTS either way
    all_patents = all_patents[:MAX_TOTAL_PATENTS]

    status_updates.append({
        "type": "status", "agent": "patent_fetcher",
        "message": f"Found {len(all_patents)} patents", "step": 1,
    })

    # Save to database
    db = SessionLocal()
    try:
        update_analysis_status(db, UUID(session_id), "running")
        db_patents = save_patents(db, UUID(session_id), all_patents)
        for data, db_patent in zip(all_patents, db_patents):
            data["db_id"] = db_patent.id
    finally:
        db.close()

    return {"patents": all_patents, "status_updates": status_updates}
