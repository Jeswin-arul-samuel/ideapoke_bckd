import uuid
from app.models.tables import Analysis, Patent, Innovation

def test_analysis_model_has_required_fields():
    analysis = Analysis(id=uuid.uuid4(), search_query="Lithium Battery", status="pending")
    assert analysis.search_query == "Lithium Battery"
    assert analysis.status == "pending"
    assert analysis.synthesis is None
    assert analysis.generated_ideas is None

def test_patent_model_has_required_fields():
    session_id = uuid.uuid4()
    patent = Patent(session_id=session_id, search_query="Lithium Battery", title="Test Patent", publication_number="US20240001")
    assert patent.session_id == session_id
    assert patent.title == "Test Patent"

def test_innovation_model_has_required_fields():
    innovation = Innovation(session_id=uuid.uuid4(), patent_id=1, innovation_summary="A novel approach", technology_used=["AI", "ML"], problem_solved="Efficiency")
    assert innovation.technology_used == ["AI", "ML"]
