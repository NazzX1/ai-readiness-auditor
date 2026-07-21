from langchain_core.messages import HumanMessage, SystemMessage
from models.schema_models import SensitivityLevel
from src.graph.state import AgentState, DataMetadata
from langchain_ollama import ChatOllama
from src.helpers.config import get_settings
import json
from datetime import datetime, timezone
from data_connectors.DataConnectorFactory import DataConnectorFactory



settings = get_settings()



DATA_PROFILER_SYSTEM_PROMPT = """
You are a Data Structural Profiler. Your goal is to analyze a dataset's schema and sample data to provide structural metadata.

Your output MUST be a JSON object with the following fields:
{
  "name": "The best descriptive name for this dataset",
  "description": "A technical summary of what this data contains (no governance scores)",
  "file_type": "The detected format (e.g., csv, json, postgres_table)",
  "modality": "Classify as: structured, semi_structured, unstructured, or media",
  "structure": "A technical representation (e.g., column names and types, or JSON hierarchy)",
  "tags": ["list", "of", "relevant", "keywords"],
  "data_lineage": ["origin details or source name"]
}

Do not provide governance scores, sensitivity ratings, or security alerts. Focus strictly on structural and descriptive data.
"""



def build_profiler_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.PROFILER_TEMP
    )



def parse_metadata_json(raw_json: str) -> DataMetadata:
    try:
        data = json.loads(raw_json)
        return DataMetadata(
            name=data.get("name", ""),
            description=data.get("description", ""),
            file_type=data.get("file_type", ""),
            modality=data.get("modality", ""),
            structure=data.get("structure", {}),
            data_lineage=data.get("data_lineage", []),
            sensitivity_lvl=SensitivityLevel.INTERNAL,  
            tags=data.get("tags", []),
            size_bytes=0,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    

def data_profiler_node(state: AgentState) -> dict:
    llm = build_profiler_llm().bind(format="json")
    
    connector = DataConnectorFactory.getConnector(
        state.get("data_source"), 
        **state.get("data_source_params", {})
    )
    
    raw_schema = connector.get_full_metadata()
    table_name = list(raw_schema.keys())[0] if raw_schema else "dataset"
    sample = connector.get_sample(table_name=table_name, limit=5)
    

    messages = [
            SystemMessage(content=DATA_PROFILER_SYSTEM_PROMPT),
            HumanMessage(content=f"Analyze this structure: {raw_schema} and this sample: {sample}")
        ]
    try:
        response = llm.invoke(messages)
    except Exception as e:
        return{
            "error" : f"Failed to invoke LLM: {str(e)}",
            "messages" : messages
        }
    
    try:
        metadata = parse_metadata_json(response.content)
    except Exception as e:
        return{
            "error" : f"Failed to parse metadata JSON: {str(e)}",
            "messages" : messages + [response]
        }
    
    metadata.size_bytes = sum(table.get("size_bytes",0) for table in raw_schema.values())

    return {
        "metadata" : metadata,
        "messages" : messages + [response],
        "error" : None
    }


