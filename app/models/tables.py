import uuid
from datetime import datetime, timezone
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Date, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from app.database import Base

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    search_query = Column(Text, nullable=False)
    synthesis = Column(JSONB, nullable=True)
    generated_ideas = Column(JSONB, nullable=True)
    previous_session_id = Column(UUID(as_uuid=True), nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    patents = relationship("Patent", back_populates="analysis")
    innovations = relationship("Innovation", back_populates="analysis")

class Patent(Base):
    __tablename__ = "patents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False)
    search_query = Column(Text)
    title = Column(Text)
    publication_number = Column(String(50))
    patent_link = Column(Text)
    priority_date = Column(Date, nullable=True)
    filing_date = Column(Date, nullable=True)
    publication_date = Column(Date, nullable=True)
    inventor = Column(Text, nullable=True)
    assignee = Column(Text, nullable=True)
    snippet = Column(Text, nullable=True)
    pdf_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    analysis = relationship("Analysis", back_populates="patents")
    innovations = relationship("Innovation", back_populates="patent")

class Innovation(Base):
    __tablename__ = "innovations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False)
    patent_id = Column(Integer, ForeignKey("patents.id"), nullable=True)
    chunk_text = Column(Text, nullable=True)
    innovation_summary = Column(Text, nullable=True)
    technology_used = Column(ARRAY(Text), nullable=True)
    problem_solved = Column(Text, nullable=True)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    analysis = relationship("Analysis", back_populates="innovations")
    patent = relationship("Patent", back_populates="innovations")
