from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage
from src.models.schema_models import SensitivityLevel
from src.graph.state import AgentState, DataMetadata
from langchain_ollama import ChatOllama
from src.helpers.config import get_settings
import json
from datetime import datetime, timezone
from src.data_connectors.DataConnectorFactory import DataConnectorFactory
from src.stores.RedisStore import RedisStore
from pprint import pprint



settings = get_settings()



DATA_PROFILER_SYSTEM_PROMPT = """
You are an expert Data Structural Profiler and AI Readiness Auditor. Your goal is to analyze the dataset's schema, sample rows, 
and structural properties to extract technical metadata while actively screening for data governance gaps, compliance risks, 
and structural blockers that could cause AI pipelines or models to fail.

### Instructions:
1. Examine the provided data sample, headers, and schema carefully.
2. Evaluate structural readiness and potential data governance flags (such as raw PII exposure risks, or unstructured data hazards).
3. Output **ONLY** a valid JSON object matching the schema below. Do not include markdown code blocks, conversational text, or outside explanations.


### Required JSON Output Schema:
{
  "name": "A concise, descriptive name for the dataset based on its contents",
  "description": "A technical summary of the dataset's domain, records, and structural layout",
  "file_type": "The detected format (e.g., csv, json, parquet, xlsx, postgres_table)",
  "modality": "Must be strictly one of: structured, semi_structured, unstructured, media",
  "domain": "A high-level domain classification (e.g., finance, healthcare, retail, social_media, IoT)(!important)",
  "structure": "A detailed representation of columns and data types or JSON hierarchy",
  "tags": ["3 to 5 relevant technical keywords or domain tags"],
  "data_lineage": ["Origin details, source filename, or connection identifier"],
  "governance_assessment": {
    "contains_potential_pii": true or false,
    "pii_risk_fields": ["List any column names that look like sensitive identifiers, emails, names, or phones, or empty list"],
  }
}
"""



def build_profiler_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.PROFILER_TEMP
    )


def create_store() -> RedisStore:
    return RedisStore(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD
    )



def parse_metadata_json(raw_json: str, sample: list) -> DataMetadata:
    try:
        data = json.loads(raw_json)
        pprint(data)
        return DataMetadata(
            name=data.get("name", ""),
            description=data.get("description", ""),
            file_type=data.get("file_type", ""),
            modality=data.get("modality", ""),
            structure=data.get("structure", {}),
            sample="",
            data_lineage=data.get("data_lineage", []),
            sensitivity_lvl=SensitivityLevel.INTERNAL.value, 
            governance_assessment=data.get("governance_assessment", {}), 
            tags=data.get("tags", []),
            size_bytes=0,
            created_at=datetime.now(timezone.utc).isoformat()

        ), str(data.get("domain", "")) if data.get("domain", "") else ""
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    

def data_profiler_node(state: AgentState) -> dict:


    store = create_store()

    
    llm = build_profiler_llm().bind(format="json")
    connector = DataConnectorFactory.getConnector(
        state.get("data_source"), 
        **state.get("data_source_params", {})
    )
    raw_schema = connector.get_full_metadata()
    sample = connector.get_sample()

    store.save_samples(session_id=state.get("session_id"), samples=[sample], ttl_seconds=3600)

    if isinstance(sample, dict):
        sample_preview = {table: rows[:5] if isinstance(rows, (list, tuple)) else rows for table, rows in list(sample.items())[:1]}
    elif isinstance(sample, (list, tuple)):
        sample_preview = sample[:1]
    else:
        sample_preview = sample
    messages = [
            SystemMessage(content=DATA_PROFILER_SYSTEM_PROMPT),
            HumanMessage(content=f"Analyze this structure: {raw_schema} and this sample: {sample_preview}")
        ]
    try:
        response = llm.invoke(messages)
    except Exception as e:
        return{
            "error" : f"Failed to invoke LLM: {str(e)}",
            "messages" : messages
        }
    try:
        metadata, domain = parse_metadata_json(response.content, sample)
    except Exception as e:
        return{
            "error" : f"Failed to parse metadata JSON: {str(e)}",
            "messages" : messages + [response]
        }
    total_size = 0
    if isinstance(raw_schema, dict):
        items = raw_schema.values()
    elif isinstance(raw_schema, list):
        items = raw_schema
    else:
        items = []
    for item in items:
        if isinstance(item, dict):
            total_size += item.get("size_bytes", 0)
        else:
            total_size += getattr(item, "size_bytes", 0)
    metadata.size_bytes = total_size
    return {
        "session_id" : state.get("session_id", ""),
        "data_metadata" : asdict(metadata),
        "task_type" : state.get("contextual_payload", {}).get("task"),  
        "domain" : domain,
        "messages" : messages + [response],
        "error" : None
    }
    


