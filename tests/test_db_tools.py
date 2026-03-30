import uuid
import pytest
from app.tools.db_tools import (
    create_analysis, get_analysis, update_analysis_status,
    save_patents, save_innovations, get_innovations_by_session,
    list_analyses, update_analysis_results,
)

def test_create_analysis(db_session):
    analysis = create_analysis(db_session, "Lithium Battery")
    assert analysis.search_query == "Lithium Battery"
    assert analysis.status == "pending"
    assert analysis.id is not None

def test_get_analysis(db_session):
    analysis = create_analysis(db_session, "Lithium Battery")
    fetched = get_analysis(db_session, analysis.id)
    assert fetched is not None
    assert fetched.search_query == "Lithium Battery"

def test_update_analysis_status(db_session):
    analysis = create_analysis(db_session, "Solar Cells")
    update_analysis_status(db_session, analysis.id, "running")
    fetched = get_analysis(db_session, analysis.id)
    assert fetched.status == "running"

def test_save_patents(db_session):
    analysis = create_analysis(db_session, "Lithium Battery")
    patents_data = [
        {"title": "Patent A", "publication_number": "US001", "search_query": "Lithium Battery"},
        {"title": "Patent B", "publication_number": "US002", "search_query": "Lithium Battery"},
    ]
    saved = save_patents(db_session, analysis.id, patents_data)
    assert len(saved) == 2
    assert saved[0].session_id == analysis.id

def test_save_innovations(db_session):
    analysis = create_analysis(db_session, "Lithium Battery")
    patents_data = [{"title": "P1", "publication_number": "US001", "search_query": "Lithium Battery"}]
    patents = save_patents(db_session, analysis.id, patents_data)
    innovations_data = [
        {"patent_id": patents[0].id, "innovation_summary": "Novel cathode design",
         "technology_used": ["nanomaterials"], "problem_solved": "energy density"}
    ]
    saved = save_innovations(db_session, analysis.id, innovations_data)
    assert len(saved) == 1
    assert saved[0].innovation_summary == "Novel cathode design"

def test_get_innovations_by_session(db_session):
    analysis = create_analysis(db_session, "AI Chips")
    patents_data = [{"title": "P1", "publication_number": "US001", "search_query": "AI Chips"}]
    patents = save_patents(db_session, analysis.id, patents_data)
    innovations_data = [
        {"patent_id": patents[0].id, "innovation_summary": "Neural engine",
         "technology_used": ["ASIC"], "problem_solved": "inference speed"}
    ]
    save_innovations(db_session, analysis.id, innovations_data)
    results = get_innovations_by_session(db_session, analysis.id)
    assert len(results) == 1

def test_list_analyses(db_session):
    create_analysis(db_session, "Topic A")
    create_analysis(db_session, "Topic B")
    results = list_analyses(db_session)
    assert len(results) >= 2
