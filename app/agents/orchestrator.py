from langgraph.graph import StateGraph, START, END
from app.agents.state import AgentState
from app.agents.patent_fetcher import patent_fetcher_node
from app.agents.innovation_extractor import innovation_extractor_node
from app.agents.synthesis import synthesis_node
from app.agents.ideation import ideation_node

def build_pipeline():
    """Build and compile the LangGraph patent analysis pipeline."""
    builder = StateGraph(AgentState)
    builder.add_node("patent_fetcher", patent_fetcher_node)
    builder.add_node("innovation_extractor", innovation_extractor_node)
    builder.add_node("synthesizer", synthesis_node)
    builder.add_node("ideation", ideation_node)
    builder.add_edge(START, "patent_fetcher")
    builder.add_edge("patent_fetcher", "innovation_extractor")
    builder.add_edge("innovation_extractor", "synthesizer")
    builder.add_edge("synthesizer", "ideation")
    builder.add_edge("ideation", END)
    return builder.compile()

pipeline = build_pipeline()
