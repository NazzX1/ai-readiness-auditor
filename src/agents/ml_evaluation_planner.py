from __future__ import annotations
import json
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from src.models.schema_models import Metric

from src.graph.state import AgentState, ContextualPayload
from src.helpers.config import get_settings
from src.stores.vectordb.GraphDB import GraphDB
from pprint import pprint


settings = get_settings()

DEFAULT_METRICS: list[Metric] = [
    Metric(
        name="Total Missing Value Rate",
        description="Share of missing cells across the whole dataset.",
        formula=None,
        pass_threshold=None,
        fail_threshold=None,
        warn_threshold=None
    ),
    Metric(
        name="Class Imbalance Ratio",
        description="Ratio of majority to minority class counts in the target column.",
        formula=None,
        pass_threshold=None,
        fail_threshold=None,
        warn_threshold=None,
    ),
    Metric(
        name="Potential Target Leakage Candidate Count",
        description="Count of feature columns suspiciously correlated with the target.",
        formula=None,
        pass_threshold=None,
        fail_threshold=None,
        warn_threshold=None,
    ),
    Metric(
        name="Raw PII Field Count",
        description="Count of columns matching PII patterns.",
        formula=None,
        pass_threshold=None,
        fail_threshold=None,
        warn_threshold=None,
    ),
]

_TASKS_REQUIRING_TARGET = ("classif", "regress", "forecast")

ML_PLANNER_SYSTEM_PROMPT = """
You are an expert Data Quality, Feature Engineering, and AI Governance Agent.

Your task is to analyze a provided dataset schema, sample data, and ML task context to produce a structured ML Dataset Evaluation Plan.

### INPUT EXPECTATIONS
You will evaluate the dataset based on:
1. Dataset Schema (Column names, Data Types, Nullability, Sample Values/Stats)
2. ML Task Definition (Problem Type, Target Column, Domain)
3. Retrieved/Domain Task Metrics (Specific metrics provided in the prompt context)

### METRIC CLASSIFICATION RULES
- classic_quality_metrics: Baseline structural checks applicable across standard tabular schemas (e.g., Completeness Rate, Uniqueness Rate, Format Validity Rate, Numerical Outlier Rate).
- retrieved_task_metrics: Task or domain metrics explicitly requested in the retrieved context. Every metric provided in the task context MUST appear here.
- suggested_task_metrics: Important task- or domain-specific metrics that were NOT present in the retrieved context, but that you evaluate as highly beneficial for model success, reliability, or fairness based on the task type and schema signals.
- Target Columns Rule: Provide "target_columns" as an array of strings matching exact column names. 
  - For multi-feature metrics, list all relevant column names in the array (e.g., ["age", "income"]).
  - If a metric needs to be computed across every column in the dataset, set "target_columns" to ["all"].
  - If a metric applies strictly to the overall dataset as a single aggregate rather than columns, set "target_columns" to null.

### GOVERNANCE & COMPLIANCE EVALUATION
- Evaluate dataset readiness qualitatively and flag regulatory risks (e.g., GDPR, PCI-DSS, HIPAA, CCPA).
- ONLY flag a regulation if explicit column names or sample values contain direct signals (e.g., SSN, Email, IP Address -> GDPR/CCPA; Credit Card Number, CVV -> PCI-DSS; Medical Diagnoses, ICD Codes -> HIPAA).
- Do NOT flag regulations based on vague assumptions. If no explicit signals exist, set "governance_flags" to [].
- Strictly avoid providing remediation, cleanup, or mitigation advice.

### OUTPUT FORMAT
Respond STRICTLY with valid JSON matching the schema below. 
Do NOT include markdown formatting (no ```json code blocks), preambles, or postscript explanations.

{
  "llm_analysis": "<2-5 sentence qualitative readiness narrative evaluating dataset suitability for the target task>",
  "confidence": "<\"low\" | \"medium\" | \"high\">",
  "classic_quality_metrics": [
    {
      "metric_name": "<classic data quality metric name>",
      "target_columns": ["<exact column name>", ...] or ["all"] or null,
      "reasoning": "<1 sentence explaining why this metric is applied>"
    }
  ],
  "retrieved_task_metrics": [
    {
      "metric_name": "<exact metric name from context>",
      "target_columns": ["<exact column name>", ...] or ["all"] or null,
      "reasoning": "<1 sentence explaining why this metric operates on these target columns>"
    }
  ],
  "suggested_task_metrics": [
    {
      "metric_name": "<name of beneficial metric omitted from retrieved context>",
      "target_columns": ["<exact column name>", ...] or ["all"] or null,
      "description": "<1-2 sentences defining what the metric measures and why it is critical for this specific task/schema>",
      "reasoning": "<1 sentence explaining why this specific metric is missing and useful here>"
    }
  ],
  "limitations": [
    "<short string explaining why analysis may be incomplete, e.g., missing sample values, unstated target variable>"
  ],
  "governance_flags": [
    {
      "name": "<regulation name, e.g., 'GDPR'>",
      "assessment": "<1-2 sentences detailing the specific column(s) or signal that triggered this flag>"
    }
  ]
}
""" 
TARGET_INFERENCE_SYSTEM_PROMPT = """
You identify the ML target/label column for a dataset given its schema and user-provided context (domain, task, dataset description).

Use the user-provided context as the primary signal for what the model is meant to predict. If the user does not explicitly state the target column or provides minimal context, form a domain-informed hypothesis based on the schema (e.g., standard business outcomes, fraud flags, churn indicators, or transaction statuses) to select the most plausible target column.

Respond with exactly one JSON object and nothing else — no markdown fences, no commentary before or after. It must match this shape exactly:

{
  "target_column": "<exact column name copied from the schema's data_types keys, or null if none is plausible>",
  "reasoning": "<1-2 sentences explaining the choice. If inferred via context, cite the context. If hypothesized due to missing explicit instructions, clearly state the hypothesis and why this column serves as the primary objective indicator.>"
}

If no column in the schema is a plausible target even after domain evaluation, "target_column" must be null (not the string "unknown", not an empty string).
"""

FIELD_TRANSLATION_SYSTEM_PROMPT = """
You are a Schema Normalization Agent. Map incoming context to the target keys: `task_type`, `domain`, and `modality`.

### EXECUTION PIPELINE (Run for each key):

1. **EXTRACT & CLEAN**:
   - Retrieve candidate values from `inferred_fields` or fallback text in `datasource_context`.
   - Strip code prefixes (e.g., `TaskCategory.CLASSIFICATION` -> `classification`) and trim whitespace.

2. **SYNONYM RESOLUTION**:
   - Map aliases to canonical database equivalents before matching:
     - `retail` / `e-commerce` -> `e_commerce_and_retail`
     - `structured` -> `tabular`
     - `nlp` -> `text_nlp`
     - `cv` -> `computer_vision`

3. **MATCH TO ALLOWED VALUES**:
   - Compare candidate against `database_allowed_values[key]`.
   - **Ignore whitespace when matching** (e.g., treat `"regression "` in the database array as matching `"regression"`).
   - If a match is found, output the corresponding string from `database_allowed_values[key]`.

4. **STRICT FALLBACK**:
   - If no candidate or synonym exists in `database_allowed_values[key]`, set the field to `""`. NEVER hallucinate values.

### OUTPUT FORMAT:
Return ONLY a raw JSON object with NO markdown formatting (no ```json fences) and NO commentary.

{
  "task_type": "<matched_allowed_value_or_empty_string>",
  "domain": "<matched_allowed_value_or_empty_string>",
  "modality": "<matched_allowed_value_or_empty_string>"
}
"""


def _get_graph_store() -> GraphDB | None:
    return GraphDB(
                uri=settings.NEO4J_URI,
                user=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
            )


def search_knowledge_graph(domain: str, modality: str, task: str) -> dict[str, Any]:
    store = _get_graph_store()
    results = store.search_metrics(domain, modality, task)

    metrics = [
         Metric(
             name=row.get("name"),
             description=row.get("description", ""),
             formula=row.get("formula"),
             pass_threshold=row.get("threshold_pass"),
             fail_threshold=row.get("threshold_fail"),  
             warn_threshold=row.get("threshold_warn"),
         )
        for row in results
    ]
    return {"metrics": metrics}


def build_planner_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.ML_PLANNER_TEMP,
    )


def _invoke_llm_infer_target(context: ContextualPayload, metadata_profile: dict[str, Any]):

    fallback = {"target_column": None, "reasoning": "LLM target inference failed or returned unparsable output."}
    try:
        llm = build_planner_llm().bind(format="json")
        sample = metadata_profile.get("sample", [])
        sample_to_use = sample[:5] if isinstance(sample, (list, tuple)) else []
        prompt = {
            "user_context": context,
            "data_types": metadata_profile,
            "sample_rows_example": sample_to_use,
        }
        messages = [
            SystemMessage(content=TARGET_INFERENCE_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(prompt, default=str)),
        ]
        response = llm.invoke(messages)
        content = response.content
        parsed = json.loads(content) if isinstance(content, str) else (content if isinstance(content, dict) else {})
        return {
            "target_column": parsed.get("target_column") or None,
            "reasoning": parsed.get("reasoning") or "No reasoning returned by LLM.",
        }
    except Exception:
        return fallback



def resolve_target_column(context: ContextualPayload, metadata_profile: dict[str, Any]):
    task_lower = context.get("task", "")
    needs_target = any(k in task_lower.value for k in _TASKS_REQUIRING_TARGET)

    data_types = (
        metadata_profile.get("data_types", {})
        if isinstance(metadata_profile, dict)
        else {}
    )
    schema_known = isinstance(data_types, dict) and bool(data_types)

    explicit_target = context.get("target_column")
    if explicit_target:
        is_in_schema = not schema_known or explicit_target in data_types
        return {
            "target_column": explicit_target,
            "status": "known" if is_in_schema else "specified_but_not_in_schema",
            "needs_target": needs_target,
            "reasoning": (
                "Target column was provided explicitly by the caller."
                if is_in_schema
                else f"Caller specified target column '{explicit_target}', but it was not found among profiled columns."
            ),
        }

    if not needs_target:
        return {
            "target_column": None,
            "status": "not_required",
            "needs_target": False,
            "reasoning": f"Task '{context.get("task", "")}' does not require a target column.",
        }

    inference = _invoke_llm_infer_target(context, metadata_profile)
    inferred_column = inference.get("target_column")
    reasoning = inference.get(
        "reasoning", "No reasoning provided by LLM target inference."
    )

    if not inferred_column:
        status = "required_but_unresolved"
    elif schema_known and inferred_column not in data_types:
        status = "inferred_but_not_in_schema"
    else:
        status = "inferred"

    return {
        "target_column": inferred_column,
        "status": status,
        "needs_target": True,
        "reasoning": reasoning,
    }

def build_metrics_to_compute(knowledge_rules: dict[str, Any], target_info: dict[str, Any]):
    if knowledge_rules.get("metrics"):
        base_metrics = knowledge_rules["metrics"]
    else:
        base_metrics = DEFAULT_METRICS

    target_known = target_info["status"] in ("known", "inferred")
    metrics = []
    for metric in base_metrics:
        # computable = not metric.requires_target or target_known
        metrics.append({
            "name": metric.name,
            "description": metric.description,
            "formula": metric.formula,
            # "threshold": metric.threshold,
            # "computable": computable,
            # "skip_reason": None if computable else "Requires a target column, which is not resolved yet (status: %s)." % target_info["status"],
        })
    return metrics


def invoke_llm_analysis(
    metadata_profile: dict[str, Any],
    sample_rows: list[Any],
    context: ContextualPayload,
    knowledge_rules: dict[str, Any],
    target_info: dict[str, Any],
) -> dict[str, Any]:


    llm = build_planner_llm().bind(format="json")
    sample_to_use = sample_rows[:5] if isinstance(sample_rows, (list, tuple)) else []
    prompt = {
        "task_context": context,
        "metadata_profile": metadata_profile,
        "sample_rows_example": sample_to_use,
        "knowledge_base_rules": knowledge_rules,
        "target_column": target_info["target_column"],
        "target_column_status": target_info["status"],
    }
    messages = [
        SystemMessage(content=ML_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(prompt, default=str)),
    ]
    response = llm.invoke(messages)
    content = response.content
    parsed = json.loads(content) if isinstance(content, str) else (content if isinstance(content, dict) else {})
    return {
        "llm_analysis": parsed.get("llm_analysis"),
        "metrics" : (parsed.get("classic_quality_metrics"), parsed.get("retrieved_task_metrics"), parsed.get("suggested_quality_metrics")),
        "confidence": parsed.get("confidence", "unspecified"),
        "limitations": parsed.get("limitations", []),
        "governance_flags": parsed.get("governance_flags", []) if isinstance(parsed.get("governance_flags"), list) else [],
    }
    

def normalize_fields(domain, task_type, modality, data_source):
    try:

        llm = build_planner_llm()

        messages =[
            SystemMessage(content=FIELD_TRANSLATION_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps({
                "inferred_fields": {
                    "domain": domain,
                    "task_type": task_type,
                    "modality": modality
                },
                "database_allowed_values": {
                    "domain":  _get_graph_store().get_domains(),
                    "task_type":_get_graph_store().get_tasks(),
                    "modality": _get_graph_store().get_modalities()
                },
                "datasource_context": data_source,
            }, default=str))
        ]  

        response = llm.invoke(messages)
        content = response.content
        parsed = json.loads(content) if isinstance(content, str) else (content if isinstance(content, dict) else {})
        return parsed.get("domain", ""), parsed.get("task_type", ""), parsed.get("modality", "")

           
    except Exception as e:
        print(f"[ERROR] Field normalization failed: {e}")
        return domain, task_type, modality



def ml_evaluation_planner_node(state: AgentState) -> dict[str, Any]:

    print(f"[ML_PLANNER]:\n {state}")

    data_metadata_obj = state.get("data_metadata")
    if data_metadata_obj is not None:
        if hasattr(data_metadata_obj, 'to_dict'):
            data_metadata = data_metadata_obj.to_dict()
        elif isinstance(data_metadata_obj, dict):
            data_metadata = data_metadata_obj
        else:
            data_metadata = {}
    else:
        data_metadata = {}
    target_info = resolve_target_column(state.get("contextual_payload"),
                                        data_metadata)
    modality_value = data_metadata.get("modality", "")
    domain, task_type, modality = normalize_fields(domain=state.get("domain"),
                     task_type=state.get("task_type").value,
                     modality=modality_value,
                     data_source=state.get("data_source", ""))
    print(f"\n[DEBUG] new modality: {modality}")
    print(f"\n[DEBUG] new task_type: {task_type}")
    print(f"\n[DEBUG] new domain: {domain}")
    retrieved_metrics = search_knowledge_graph(domain,
                                               modality,
                                               task_type
                                               )

    data_metadata["domain"] = domain
    data_metadata["task_type"] = task_type  
    data_metadata["modality"] = modality
    metrics_to_compute = build_metrics_to_compute(retrieved_metrics, target_info)
    llm_analysis_result = invoke_llm_analysis(data_metadata, 
                                              data_metadata.get("sample", []),
                                              state.get("contextual_payload"), 
                                              retrieved_metrics, 
                                              target_info)
    evaluation_plan = {
        "dataset_context": {
            "dataset_name": data_metadata.get("name", ""),
            "domain": data_metadata.get("domain", ""),
            "task": data_metadata.get("task_type", ""),
        },
        "target": {
            "column": target_info["target_column"],
            "status": target_info["status"],
            "reasoning": target_info["reasoning"],
        },
        "metrics": llm_analysis_result["metrics"],
        "governance_checks": llm_analysis_result["governance_flags"],
    }
    pprint(f"\n[ML_EVALUATION_PLANNER]:\n {evaluation_plan}")
    return {
        "status": "success",
        "reasoning_trace": [
            {"step": "TARGET COLUMN CHECK", "detail": target_info["reasoning"]},
            {"step": "KNOWLEDGE BASE LOOKUP", "detail": f"domain='{state.get('domain', '')}' task='{state.get('task_type')}' rules_found={bool(retrieved_metrics.get('metrics'))}"},
            {"step": "READINESS ANALYSIS", "detail": f"confidence={llm_analysis_result['confidence']}, governance_flags_found={len(llm_analysis_result['governance_flags'])}"},
        ],
        "evaluation_plan": evaluation_plan,
        "llm_analysis": llm_analysis_result["llm_analysis"],
    }
    