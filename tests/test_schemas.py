from app.models.schemas import AnalysisCreate, AnalysisResponse, FollowupRequest, GeneratedIdea

def test_analysis_create_minimal():
    req = AnalysisCreate(query="Lithium Battery")
    assert req.query == "Lithium Battery"
    assert req.previous_session_id is None

def test_analysis_create_with_previous():
    req = AnalysisCreate(query="Solar Cells", previous_session_id="550e8400-e29b-41d4-a716-446655440000")
    assert req.previous_session_id is not None

def test_generated_idea_shape():
    idea = GeneratedIdea(title="Self-healing membranes", explanation="Bio-inspired approach...", patent_trail=["US20240123456", "US20230789012"])
    assert len(idea.patent_trail) == 2

def test_followup_request():
    req = FollowupRequest(session_id="550e8400-e29b-41d4-a716-446655440000", question="Tell me more about idea 1")
    assert req.question == "Tell me more about idea 1"
