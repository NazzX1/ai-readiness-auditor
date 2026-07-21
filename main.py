import argparse
import uuid
from src.graph.state import initial_state
from src.graph.workflow import graph 


def run_session(session_id : str):


    is_resume = session_id is not None
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    
    print("="*10)
    print("Learning Accelerator")
    print(f"session id : {session_id}")

    if is_resume:
        print("Resuming existing session")
    else:
        print("")
    print("="*10)


    


    state = initial_state(session_id, "", "") if not is_resume else None

    config = {"configurable" : {"thread_id" : uuid.uuid4()}}

    try:
        result = graph.invoke(state, config=config)
    except Exception as e:
        return



    









if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "AI Readiness auditor : ...."
        ),
        epilog=(
            ""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )