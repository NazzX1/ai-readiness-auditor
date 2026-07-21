from langchain_ollama import ChatOllama
from src.helpers.config import get_settings
from src.graph.state import AgentState



settings = get_settings()


def build_profiler_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.ML_PLANNER_TEMP
    )

def ml_evaluation_planner(state : AgentState):
    pass