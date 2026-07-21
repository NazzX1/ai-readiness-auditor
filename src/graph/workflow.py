from pathlib import Path
from langgraph.graph import START, StateGraph
import sqlite3
from src.helpers.config import get_settings
from langgraph.checkpoint .sqlite import SqliteSaver
from src.agents.data_profiler import data_profiler_node
from state import AgentState


settings = get_settings()


def build_graph(db_path : str, interrupt_before : list | None = None):

    Path("data").mkdir(exist_ok=True)

    builder = StateGraph(AgentState)

    builder.add_node("data_profiler", data_profiler_node)



    builder.add_edge(START, "data_profiler")



    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or []
    )



graph = build_graph(db_path=settings.CHECKPOINT_DB, interrupt_before=None)