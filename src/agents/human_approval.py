from langgraph.types import interrupt

from src.graph.state import DataMetadata, AgentState
from src.stores.RedisStore import RedisStore
from src.helpers.config import get_settings
from pprint import pprint



settings = get_settings()

def create_store() -> RedisStore:
    return RedisStore(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD
    )

def human_approval_node(state : dict) -> dict:

    print(f"\n[Human approval] inferred task type: {state}")

    store = create_store()
    print(f"\nSamples fro redis store:\n {state.get('session_id')}")
    pprint(store.get_samples(session_id=state.get("session_id")))


    profile : DataMetadata | None = state.get("data_metadata", None)
    context = ""
    print(f"\n[Human approval] Data Profile for review:\n {profile}")
    if profile is None:
        print(f"[Human approval] No Data Profile found, skipping approval")
        return {
            "approved" : True
        }
    print(f"\n[Human approval] pausing for Data metadata review...")
    decision = interrupt({
        "type" : "data profile approval",
        "profile" : profile,
        "prompt" : (
            "Does this data profile look good?\n"
            "  Type 'yes' to proceed\n"
            "  Type 'no' to generate a different profile"
        )
    })
    approved = str(decision).lower().strip() in ("yes", "y", "ok")
    if approved:
        print(f"[Human approval] Data Profileapproved, starting Evaluation Planning...")
    else:
        print(f"[Human approval] Data Profile rejected, regenerating...")
        context = input("Add context or notes for the next iteration and press Enter to continue...")


    payload = state.get("contextual_payload") or {}

    if isinstance(payload, dict):
        existing_info = payload.get("additional_info") or ""
    else:
        existing_info = getattr(payload, "additional_info", None) or ""

    approval_note = f"\n[Human approval context]: {context}"
    new_info = f"{existing_info}{approval_note}".strip()

    if isinstance(payload, dict):
        updated_payload = {**payload, "additional_info": new_info}
    else:
        payload.additional_info = new_info
        updated_payload = payload
    
    print(f"\n[DEBUG] : {updated_payload}")


    print(f" \n[DEBUG] : {state.get("contextual_payload")}")
    return {
        "approved": approved,
        "session_id": state.get("session_id", ""),
        "data_source": state.get("data_source", ""),
        "data_source_params": state.get("data_source_params", ""),
        "data_metadata": profile,
        "domain": state.get("domain", ""),
        "task_type": state.get("task_type", ""),
        "contextual_payload": updated_payload,
        "error": None
    }