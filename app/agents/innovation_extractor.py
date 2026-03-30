import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import UUID

from langchain.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from poml.integration.langchain import LangchainPomlTemplate

from app.agents.state import AgentState
from app.config import settings
from app.database import SessionLocal
from app.tools.db_tools import save_innovations
from app.tools.embedding import get_embeddings_batch
from app.tools.pdf_processor import chunk_text, download_pdf

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

MAX_PARALLEL_OPENAI = settings.MAX_PARALLEL_OPENAI
MAX_PARALLEL_GROQ = settings.MAX_PARALLEL_GROQ
MAX_RETRIES = 3


# --- Pydantic models for structured output ---

class Innovation(BaseModel):
    """A single innovation extracted from a patent."""
    innovation_summary: str = Field(description="Concise description of the innovation")
    technology_used: list[str] = Field(description="Technologies/methods used")
    problem_solved: str = Field(description="What problem this innovation addresses")


class InnovationList(BaseModel):
    """List of innovations extracted from a patent."""
    innovations: list[Innovation] = Field(description="All innovations found in the patent text")


# --- Tools for the agent ---

@tool
def fetch_patent_text(pdf_url: str, title: str, snippet: str) -> str:
    """Fetch patent text from a PDF URL. Falls back to title + snippet if PDF is unavailable.

    Args:
        pdf_url: URL to the patent PDF
        title: Patent title
        snippet: Patent abstract/snippet
    """
    if pdf_url:
        text = download_pdf(pdf_url)
        if text:
            chunks = chunk_text(text)
            return chunks[0]

    if title or snippet:
        return f"Patent: {title}\n\n{snippet}"

    return "No text available for this patent."


def extract_innovations_from_text(text: str, agent_model: str) -> list[dict]:
    """Use LLM with POML prompt and structured output to extract innovations.
    Includes retry with backoff for rate limits."""
    prompt_template = LangchainPomlTemplate.from_file(
        os.path.join(PROMPTS_DIR, "innovation_extractor.poml")
    )

    messages = prompt_template.format(patent_text=text).to_messages()

    for attempt in range(MAX_RETRIES):
        try:
            agent = create_agent(
                model=agent_model,
                tools=[],
                response_format=InnovationList,
            )
            result = agent.invoke({"messages": messages})

            structured = result.get("structured_response")
            if structured:
                return [inn.model_dump() for inn in structured.innovations]
            return []
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                wait = (attempt + 1) * 8  # 8s, 16s, 24s
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            elif "tool_use_failed" in error_str:
                logger.warning(f"Structured output failed, retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(2)
                continue
            else:
                raise
    logger.error(f"Failed after {MAX_RETRIES} retries for model {agent_model}")
    return []


def _process_single_patent(patent: dict, agent_model: str) -> dict:
    """Process a single patent: fetch text → extract innovations. Returns result dict.
    Designed to be called from a thread pool."""
    pdf_url = patent.get("pdf_url", "")
    pub_num = patent.get("publication_number", "unknown")
    title = patent.get("title", "")
    snippet = patent.get("snippet", "")
    patent_db_id = patent.get("db_id")

    # Fetch text
    text = fetch_patent_text.invoke({
        "pdf_url": pdf_url, "title": title, "snippet": snippet
    })

    if text == "No text available for this patent.":
        logger.warning(f"Skipping patent {pub_num}: no text available")
        return {"pub_num": pub_num, "innovations": []}

    # Extract innovations via LLM
    try:
        innovations = extract_innovations_from_text(text, agent_model)
        for inn in innovations:
            inn["chunk_text"] = text
            inn["patent_id"] = patent_db_id
            inn["patent_publication_number"] = pub_num
        return {"pub_num": pub_num, "innovations": innovations}
    except Exception as e:
        logger.error(f"Failed to extract from patent {pub_num}: {e}")
        return {"pub_num": pub_num, "innovations": []}


def innovation_extractor_node(state: AgentState) -> dict:
    """LangGraph node: Extract innovations from patents using parallel LLM calls."""
    session_id = state["session_id"]
    patents = state["patents"]
    status_updates = list(state.get("status_updates", []))
    all_innovations = []

    # Resolve model and parallelism from state
    provider = state.get("llm_provider", "openai")
    models = state.get("llm_models", {"extraction": settings.EXTRACTION_MODEL})
    agent_model = f"{provider}:{models['extraction']}"
    max_parallel = MAX_PARALLEL_GROQ if provider == "groq" else MAX_PARALLEL_OPENAI

    status_updates.append({
        "type": "status",
        "agent": "innovation_extractor",
        "message": f"Extracting innovations from {len(patents)} patents ({max_parallel} parallel, {provider})...",
        "step": 2,
    })

    # Process patents in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_process_single_patent, patent, agent_model): patent
            for patent in patents
        }

        for future in as_completed(futures):
            result = future.result()
            all_innovations.extend(result["innovations"])
            completed += 1

            # Progress update every 3 completed patents
            if completed % 3 == 0 or completed == len(patents):
                status_updates.append({
                    "type": "status",
                    "agent": "innovation_extractor",
                    "message": f"Processed {completed}/{len(patents)} patents...",
                    "step": 2,
                })

    # Generate embeddings in batch — uses OpenAI key (user's if OpenAI, server's if Groq)
    embedding_key = state.get("embedding_api_key")
    if all_innovations:
        summaries = [inn["innovation_summary"] for inn in all_innovations]
        all_embeddings = []
        for j in range(0, len(summaries), 100):
            batch = summaries[j : j + 100]
            all_embeddings.extend(get_embeddings_batch(batch, api_key=embedding_key))
        for inn, emb in zip(all_innovations, all_embeddings):
            inn["embedding"] = emb

    # Save to database
    db = SessionLocal()
    try:
        save_innovations(db, UUID(session_id), all_innovations)
    finally:
        db.close()

    status_updates.append({
        "type": "status",
        "agent": "innovation_extractor",
        "message": f"Extracted {len(all_innovations)} innovations",
        "step": 2,
    })

    return {"innovations": all_innovations, "status_updates": status_updates}
