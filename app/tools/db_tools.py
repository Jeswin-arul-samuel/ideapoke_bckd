import uuid
from uuid import UUID
from sqlalchemy.orm import Session
from app.models.tables import Analysis, Innovation, Patent

def create_analysis(db: Session, search_query: str, previous_session_id: UUID | None = None) -> Analysis:
    analysis = Analysis(id=uuid.uuid4(), search_query=search_query, status="pending", previous_session_id=previous_session_id)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis

def get_analysis(db: Session, analysis_id: UUID) -> Analysis | None:
    return db.query(Analysis).filter(Analysis.id == analysis_id).first()

def update_analysis_status(db: Session, analysis_id: UUID, status: str) -> None:
    db.query(Analysis).filter(Analysis.id == analysis_id).update({"status": status})
    db.commit()

def update_analysis_results(db: Session, analysis_id: UUID, synthesis: dict, generated_ideas: list[dict]) -> None:
    db.query(Analysis).filter(Analysis.id == analysis_id).update(
        {"synthesis": synthesis, "generated_ideas": generated_ideas, "status": "completed"}
    )
    db.commit()

def save_patents(db: Session, session_id: UUID, patents_data: list[dict]) -> list[Patent]:
    patents = []
    for data in patents_data:
        patent = Patent(session_id=session_id, **data)
        db.add(patent)
        patents.append(patent)
    db.commit()
    for p in patents:
        db.refresh(p)
    return patents

def save_innovations(db: Session, session_id: UUID, innovations_data: list[dict]) -> list[Innovation]:
    allowed_fields = {"patent_id", "chunk_text", "innovation_summary", "technology_used", "problem_solved", "embedding"}
    innovations = []
    for data in innovations_data:
        filtered = {k: v for k, v in data.items() if k in allowed_fields}
        innovation = Innovation(session_id=session_id, **filtered)
        db.add(innovation)
        innovations.append(innovation)
    db.commit()
    for i in innovations:
        db.refresh(i)
    return innovations

def get_innovations_by_session(db: Session, session_id: UUID) -> list[Innovation]:
    return db.query(Innovation).filter(Innovation.session_id == session_id).all()

def list_analyses(db: Session) -> list[Analysis]:
    return db.query(Analysis).order_by(Analysis.created_at.desc()).all()
