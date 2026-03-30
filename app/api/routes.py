import json
import os
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from poml.integration.langchain import LangchainPomlTemplate

from app.config import settings
from app.tools.llm_provider import resolve_llm_config
from app.database import get_db
from app.models.schemas import (
    AnalysisCreate, AnalysisResponse, AnalysisSummary, SessionResponse, FollowupRequest,
)
from app.tools.db_tools import create_analysis, get_analysis, list_analyses

router = APIRouter()

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/analysis", response_model=SessionResponse)
def start_analysis(request: AnalysisCreate, db: Session = Depends(get_db)):
    analysis = create_analysis(db, request.query, previous_session_id=request.previous_session_id)
    return SessionResponse(session_id=analysis.id)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
def get_analysis_by_id(analysis_id: UUID, db: Session = Depends(get_db)):
    analysis = get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@router.get("/analyses", response_model=list[AnalysisSummary])
def list_all_analyses(db: Session = Depends(get_db)):
    return list_analyses(db)


@router.post("/followup")
def followup(request: FollowupRequest, db: Session = Depends(get_db), openai_key: str = Header(None, alias="X-OpenAI-Key"), groq_key: str = Header(None, alias="X-Groq-Key")):
    analysis = get_analysis(db, request.session_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    ideas_text = json.dumps(analysis.generated_ideas, indent=2) if analysis.generated_ideas else "No ideas generated yet"
    synthesis_text = json.dumps(analysis.synthesis, indent=2) if analysis.synthesis else "No synthesis available"

    # Resolve LLM provider from user-provided keys
    llm_config = resolve_llm_config(openai_key=openai_key, groq_key=groq_key)

    # Load POML template for follow-up
    prompt_template = LangchainPomlTemplate.from_file(
        os.path.join(PROMPTS_DIR, "followup.poml"),
        speaker_mode=True,
    )

    messages = prompt_template.format(
        query=analysis.search_query,
        ideas_text=ideas_text,
        synthesis_text=synthesis_text,
        question=request.question,
    ).to_messages()

    llm = llm_config.chat_model("followup")

    def generate():
        from app.tools.llm_logger import log_llm_stream_usage
        full_output = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                full_output += chunk.content
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        log_llm_stream_usage("gpt-4o", "followup", str(messages), full_output)
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
