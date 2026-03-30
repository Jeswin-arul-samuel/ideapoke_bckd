from datetime import datetime
from uuid import UUID
from pydantic import BaseModel

class AnalysisCreate(BaseModel):
    query: str
    previous_session_id: UUID | None = None

class GeneratedIdea(BaseModel):
    title: str
    explanation: str
    patent_trail: list[str]

class AnalysisResponse(BaseModel):
    id: UUID
    search_query: str
    status: str
    synthesis: dict | None = None
    generated_ideas: list[GeneratedIdea] | None = None
    created_at: datetime
    model_config = {"from_attributes": True}

class AnalysisSummary(BaseModel):
    id: UUID
    search_query: str
    status: str
    created_at: datetime
    model_config = {"from_attributes": True}

class FollowupRequest(BaseModel):
    session_id: UUID
    question: str

class SessionResponse(BaseModel):
    session_id: UUID
