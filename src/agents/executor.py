from __future__ import annotations

from collections import defaultdict
import json
import inspect
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
import numpy as np
import pandas as pd

from src.graph.state import AgentState, DataMetadata
from src.helpers.config import get_settings

from src.mcp_servers.metrics_server import (
    compute_completeness_tool,
    compute_uniqueness_tool,
    check_format_validity_tool,
    compute_aspect_ratio_variance_tool,
    compute_class_imbalance_tool,
    check_schema_violations_tool,
    compute_r_squared_tool,
    compute_vif_tool,
    count_pii_fields_tool,
    detect_change_points_tool,
    compute_blur_index_tool,
    compute_outliers_tool,
    compute_iou_tool,
    detect_target_leakage_tool,
    detect_population_shift_tool,
    measure_demographic_parity_tool,
    measure_pii_contamination_tool,
    measure_phi_violations_tool
)
from src.stores.RedisStore import RedisStore

settings = get_settings()


def create_store() -> RedisStore:
    return RedisStore(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
    )


store = create_store()

EXECUTOR_TOOLS: List[BaseTool] = [
    compute_completeness_tool,
    compute_uniqueness_tool,
    check_format_validity_tool,
    compute_aspect_ratio_variance_tool,
    compute_class_imbalance_tool,
    check_schema_violations_tool,
    compute_r_squared_tool,
    compute_vif_tool,
    count_pii_fields_tool,
    detect_change_points_tool,
    compute_blur_index_tool,
    compute_outliers_tool,
    compute_iou_tool,
    detect_target_leakage_tool,
    detect_population_shift_tool,
    measure_demographic_parity_tool,
    measure_pii_contamination_tool,
    measure_phi_violations_tool
]


def build_executor_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.ML_PLANNER_TEMP,
    ).bind_tools(EXECUTOR_TOOLS)


TOOL_MAP = {getattr(tool, "name", getattr(tool, "__name__", str(tool))): tool
            for tool in EXECUTOR_TOOLS}
print(F"[EXECUTOR]:\n {TOOL_MAP}")


EXECUTOR_SYSTEM_PROMPT = """You are an expert Data Evaluation Executor Agent.
Your primary responsibility is to autonomously execute quantitative metric evaluation tools against specified dataset columns and return precise results in structured JSON format.

### OPERATIONAL DIRECTIVES:
1. **Semantic Tool Matching (CRITICAL):**
   - Requested metrics in the plan may use functional, high-level, or shorthand names (e.g., "null_rate", "missing_values", "unique_count", "duplicates").
   - Match these requested metrics to the corresponding tool that performs the same underlying calculation, even if the exact names differ (e.g., map "null_rate" or "missing_values" -> `compute_completeness_tool`, map "duplicates" -> `compute_uniqueness_tool`).
   - Do NOT reject a metric or claim a tool is missing if an equivalent tool exists in your available toolset.

2. **Autonomous Tool Invocation:**
   - You MUST invoke matching tools immediately when target columns and requested metrics are provided.
   - Do NOT ask for clarification, sample sizes, preferences, or additional user input.
   - Issue tool calls for ALL target column and metric pairs.

3. **Parameter Binding:**
   - Always extract and pass `dataset_id` (using the provided Session ID).
   - Infer `table_name` by inspecting which table contains the target `column` in the provided schema.
   - Pass the target `column` name as a string.

### OUTPUT FORMAT:
- During tool execution: Emit ONLY tool calls. Do NOT write conversational text, preamble, or commentary.
- Post-tool execution: Provide your final summary response strictly as a valid JSON object wrapped in a ```json code block with NO extra prose before or after.

Use the following JSON structure for your final response:
```json
{
  "session_id": "<session_id>",
  "summary": {
    "total_metrics_evaluated": <int>,
    "successful_evaluations": <int>,
    "failed_evaluations": <int>
  },
  "results": [
    {
      "table": "<table_name>",
      "column": "<column_name>",
      "requested_metric": "<original_metric_name>",
      "executed_tool": "<tool_name_used>",
      "value": <computed_value_or_null>,
      "status": "success|error",
      "error_message": null
    }
  ],
  "observations": [
    "<data quality observation based on computed values>"
  ]
}
"""


def extract_metrics_by_column(data: Any) -> Dict[str, List[str]]:
    col_metrics = defaultdict(list)
    metric_items = []

    if isinstance(data, (tuple, list)):
        for group in data:
            if isinstance(group, list):
                metric_items.extend(group)
            elif isinstance(group, dict):
                metric_items.append(group)
    elif isinstance(data, dict):
        metric_items.append(data)

    for item in metric_items:
        metric_name = item.get("metric_name")
        target_cols = item.get("target_columns", [])

        if not metric_name or not target_cols:
            continue

        for col in target_cols:
            if metric_name not in col_metrics[col]:
                col_metrics[col].append(metric_name)

    return dict(col_metrics)


def executor_node(state: AgentState) -> dict:
    print("[EXECUTOR]: Starting executor node...")

    session_id = state.get("session_id")
    eval_plan = state.get("evaluation_plan", {})
    metrics_raw = (
        eval_plan.get("metrics", [])
        if isinstance(eval_plan, dict)
        else getattr(eval_plan, "metrics", [])
    )

    met_to_cols = extract_metrics_by_column(metrics_raw)

    dataframes = store.get_samples_as_dataframes(session_id=session_id)
    if not dataframes:
        print("[EXECUTOR]: No dataframes found in Redis Store.")
        return {"execution_results": {}, "messages": []}

    table_schema = {
        table_name: list(df.columns) for table_name, df in dataframes.items()
    }

    llm = build_executor_llm()

    user_prompt = f"""Dataset Session ID: '{session_id}'

        Available Tables & Columns:
        {json.dumps(table_schema, indent=2)}

        Target Columns and Requested Metrics:
        {json.dumps(met_to_cols, indent=2)}

        Compute the metrics for the given target columns using available tools.
    """

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    print("[EXECUTOR]: Invoking LLM...")
    response = llm.invoke(messages)
    messages.append(response)

    execution_results = {}

    while response.tool_calls:
        print(f"[EXECUTOR]: Processing {len(response.tool_calls)} tool calls...")

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if "dataset_id" not in tool_args or not tool_args["dataset_id"]:
                tool_args["dataset_id"] = session_id

            col = tool_args.get("column", "unknown")
            table = tool_args.get("table_name")

            if not table:
                for tbl_name, df in dataframes.items():
                    if col in df.columns:
                        table = tbl_name
                        tool_args["table_name"] = table
                        break

            target_tool = TOOL_MAP.get(tool_name)
            if target_tool:
                try:
                    computed_value = target_tool.invoke(tool_args)
                    result_payload = {
                        "status": "success",
                        "metric_name": tool_name,
                        "column": col,
                        "table": table,
                        "value": computed_value,
                    }
                except Exception as e:
                    computed_value = None
                    result_payload = {
                        "status": "error",
                        "metric_name": tool_name,
                        "column": col,
                        "table": table,
                        "error": str(e),
                    }
            else:
                computed_value = None
                result_payload = {
                    "status": "error",
                    "metric_name": tool_name,
                    "column": col,
                    "error": f"Tool '{tool_name}' not registered.",
                }

            execution_results[f"{tool_name}__{col}"] = result_payload

            messages.append(
                ToolMessage(
                    content=json.dumps(result_payload),
                    tool_call_id=tool_call["id"],
                )
            )

        response = llm.invoke(messages)
        messages.append(response)

    print("[EXECUTOR]: Final LLM Response:")
    print(response.content)

    return {
        "execution_results": execution_results,
        "messages": messages,
    }