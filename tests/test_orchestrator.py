from app.agents.orchestrator import build_pipeline

def test_build_pipeline_returns_compiled_graph():
    graph = build_pipeline()
    assert graph is not None
    assert hasattr(graph, "invoke")

def test_pipeline_has_correct_nodes():
    graph = build_pipeline()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {"patent_fetcher", "innovation_extractor", "synthesizer", "ideation"}
    assert expected.issubset(node_names)
