import argparse
import uuid
from src.graph.state import initial_state
from src.graph.workflow import graph 
from src.data_connectors.DataConnectorEnums import DataConnectorEnums


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

    x = input("the data source is a database or a file? (press enter to continue)")
    if x in ("yes", "y"):
        db_type = input("database type: ")
        if db_type == "mongo":
            data_source = DataConnectorEnums.MONGO.value
            user = input("user: ")
            password = input("pwd: ")
            db_name = input("database name: ")
            uri = f"mongodb://{user}:{password}@localhost:27017"
           
        elif db_type == "postgres":
            data_source = DataConnectorEnums.POSTGRES.value
            user = input("user: ")
            password = input("pwd: ")
            db_name = input("database name: ")
            uri = f"postgresql://{user}:{password}@localhost:5432/{db_name}" 
        else:
            print("Unsupported type")
            return

        data_source_params = {"uri" : uri, "db_name" : db_name}
    elif x in ("no", "n"):
        data_source = DataConnectorEnums.DEFAULT.value
        file_path = input("Enter the path to the file to analyze (press enter to skip): ")
        data_source_params = {"file_path" : file_path}


    state = initial_state(session_id, data_source,**data_source_params) if not is_resume else None

    config = {"configurable" : {"thread_id" : uuid.uuid4()}}

    try:
        result = graph.invoke(state, config=config)
    except Exception as e:
        print(f"\n[ERROR] Could not resume session '{session_id}': {e}")
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

    parser.add_argument(
        "--resume", metavar="SESSION_ID",
        help="Resume an existing session by its 8-char ID",
    )

    args = parser.parse_args()

    
    run_session(session_id=args.resume)