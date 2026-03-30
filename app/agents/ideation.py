import json
import logging
import os
import time
from uuid import UUID

from langchain.agents import create_agent
from pydantic import BaseModel, Field
from poml.integration.langchain import LangchainPomlTemplate

MAX_RETRIES = 3

from app.agents.state import AgentState
from app.config import settings
from app.database import SessionLocal
from app.tools.db_tools import update_analysis_results

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

PREVIOUS_CONTEXT_TEMPLATE = """Previously analyzed "{prev_query}" and found these ideas:
{prev_ideas}

Consider cross-domain connections between the previous analysis and the current one."""


# --- Structured output ---

class GeneratedIdea(BaseModel):
    """A single novel innovation idea."""
    title: str = Field(description="Concise, compelling name for the innovation")
    explanation: str = Field(description="What it is, why it's a logical next step, and what makes it novel (2-3 sentences)")
    patent_trail: list[str] = Field(description="Publication numbers from existing patents that inspired this idea")


class IdeationOutput(BaseModel):
    """Collection of generated innovation ideas."""
    ideas: list[GeneratedIdea] = Field(description="Novel future innovation ideas")


def ideation_node(state: AgentState) -> dict:
    """LangGraph node: Generate novel future innovation ideas using structured output."""
    query = state["search_query"]
    synthesis = state["synthesis"]
    innovations = state["innovations"]
    previous = state.get("previous_analysis")
    status_updates = list(state.get("status_updates", []))

    status_updates.append({
        "type": "status",
        "agent": "ideation",
        "message": "Generating future innovation ideas...",
        "step": 4,
    })

    # Prepare context
    synthesis_text = json.dumps(synthesis, indent=2)
    innovations_text = "\n".join(
        f"- [{inn.get('patent_publication_number', 'N/A')}] {inn['innovation_summary']}"
        for inn in innovations[:30]
    )

    previous_context = ""
    if previous:
        prev_ideas_text = "\n".join(
            f"- {idea['title']}: {idea['explanation']}"
            for idea in previous.get("generated_ideas", [])
        )
        previous_context = PREVIOUS_CONTEXT_TEMPLATE.format(
            prev_query=previous.get("search_query", ""),
            prev_ideas=prev_ideas_text,
        )

    # Resolve model from state
    provider = state.get("llm_provider", "openai")
    models = state.get("llm_models", {"ideation": settings.IDEATION_MODEL})
    agent_model = f"{provider}:{models['ideation']}"

    # Load POML template
    prompt_template = LangchainPomlTemplate.from_file(
        os.path.join(PROMPTS_DIR, "ideation.poml")
    )

    messages = prompt_template.format(
        query=query,
        synthesis_text=synthesis_text,
        innovations_text=innovations_text,
        previous_context=previous_context,
    ).to_messages()

    ideas = []
    for attempt in range(MAX_RETRIES):
        try:
            agent = create_agent(
                model=agent_model,
                tools=[],
                response_format=IdeationOutput,
            )
            result = agent.invoke({"messages": messages})
            structured = result.get("structured_response")
            if structured:
                ideas = [idea.model_dump() for idea in structured.ideas]
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                wait = (attempt + 1) * 8
                logger.warning(f"Ideation rate limited, waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            elif "tool_use_failed" in error_str or "400" in error_str:
                logger.warning(f"Ideation structured output failed, retrying (attempt {attempt + 1})")
                time.sleep(2)
            else:
                logger.error(f"Ideation failed: {e}")
                break

    # Save results to database
    db = SessionLocal()
    try:
        update_analysis_results(db, UUID(state["session_id"]), synthesis, ideas)
    finally:
        db.close()

    status_updates.append({
        "type": "status",
        "agent": "ideation",
        "message": f"Generated {len(ideas)} future innovation ideas",
        "step": 4,
    })

    return {"generated_ideas": ideas, "status_updates": status_updates}
