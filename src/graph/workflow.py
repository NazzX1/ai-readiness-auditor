from pathlib import Path
from langgraph.graph import START, StateGraph, END
import sqlite3
from src.helpers.config import get_settings
from langgraph.checkpoint .sqlite import SqliteSaver
from src.agents.data_profiler import data_profiler_node
from src.agents.ml_evaluation_planner import ml_evaluation_planner_node
from src.agents.human_approval import human_approval_node
from src.agents.executor import executor_node
from src.agents.evaluator import evaluator_node
from src.graph.state import AgentState


settings = get_settings()


def route_after_approval(state : dict) -> str:
    if state.get("approved", False):
        return "ml_evaluation_planner"
    return "data_profiler"


def build_graph(db_path : str, interrupt_before : list | None = None):

    Path("data").mkdir(exist_ok=True)

    builder = StateGraph(AgentState)

    builder.add_node("data_profiler", data_profiler_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("ml_evaluation_planner", ml_evaluation_planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("evaluator", evaluator_node)



    builder.add_edge(START, "data_profiler")

    builder.add_edge("data_profiler", "human_approval")
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "data_profiler" : "data_profiler",
            "ml_evaluation_planner" : "ml_evaluation_planner"
        }
    )
    builder.add_edge("ml_evaluation_planner", "executor")
    builder.add_edge("executor", "evaluator")
    builder.add_edge("evaluator", END)


    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or []
    )



graph = build_graph(db_path=settings.CHECKPOINT_DB, interrupt_before=None)