from langchain_core.messages import HumanMessage, SystemMessage
from src.models.schema_models import SensitivityLevel
from src.graph.state import AgentState, DataMetadata
from langchain_ollama import ChatOllama
from src.helpers.config import get_settings
import json
from datetime import datetime, timezone
from src.data_connectors.DataConnectorFactory import DataConnectorFactory
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
            sensitivity_lvl=SensitivityLevel.INTERNAL.value, 
            governance_assessment=data.get("governance_assessment", {}), 
            tags=data.get("tags", []),
            size_bytes=0,
            created_at=datetime.now(timezone.utc).isoformat()

        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    

def data_profiler_node(state: AgentState) -> dict:
    print(f"\n[INFO] Profiling data source: {state.get('data_source')}")
    llm = build_profiler_llm().bind(format="json")
    
    connector = DataConnectorFactory.getConnector(
        state.get("data_source"), 
        **state.get("data_source_params", {})
    )
    
    raw_schema = connector.get_full_metadata()
    sample = connector.get_sample()
    

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
    
    metadata.size_bytes = sum(getattr(table, "size_bytes", 0) for table in raw_schema.values())

    print(f"\n[INFO] Data profiling completed for: {state.get('data_source')}")
    pprint(f"{metadata}")

    return {
        "metadata" : metadata,
        "messages" : messages + [response],
        "error" : None
    }


