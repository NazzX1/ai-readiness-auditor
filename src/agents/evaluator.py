from __future__ import annotations
from langchain_ollama import ChatOllama
from src.helpers.config import get_settings
from src.graph.state import AgentState
from langchain_community.tools import DuckDuckGoSearchRun
import json
import inspect
from typing import Any, Dict, List, Optional
from src.stores.vectordb.QdrantDB import QdrantDB

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage




settings = get_settings()

web_search_tool = DuckDuckGoSearchRun()


def build_evaluator_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.EVALUATOR_TEMP,
        format="json"
    )


def create_store():
    return QdrantDB(
        settings = settings
    )


EVALUATOR_SYSTEM_PROMPT = """You are an expert Data Evaluation Quality Auditor and Data Remediation Engineer.
Your job is to audit computed metric execution results, cross-reference them against research paper excerpts (Vector Knowledge Base) and web benchmarks, assign a plan score (0-100), and provide concrete data remediation recommendations.

### AUDIT DIRECTIVES:
1. **Semantic Validity Check:** Evaluate if each computed metric value makes logical and statistical sense given the column's domain context.
   - Example: Primary Keys should have 1.0 (100%) uniqueness and completeness.
   - Example: Missingness/completeness must align with standard domain expectations.
2. **Context Grounding:** Use retrieved paper chunks and web benchmarks to justify your semantic decisions.
3. **Remediation Plan:** Identify columns with quality issues and outline actionable steps to fix them (e.g., median/mode imputation, deduplication with UPSERT, regex validation, outlier capping).
4. **Plan Scoring:** Output a plan score from 0 to 10 based on result coverage, accuracy, and metric soundness.

### OUTPUT FORMAT:
You MUST respond strictly with a valid JSON object wrapped inside a ```json code block. Do NOT include conversational text before or after the JSON.

Required JSON Schema:
```json
{
  "plan_score": <int_0_to_10>,
  "plan_status": "APPROVED|REJECTED|NEEDS_REVISION",
  "semantic_evaluations": [
    {
      "table": "<table_name>",
      "column": "<column_name>",
      "metric": "<metric_name>",
      "computed_value": <number_or_null>,
      "is_semantically_valid": true|false,
      "reasoning": "<explanation incorporating paper context or web context>"
    }
  ],
  "data_remediation_recommendations": [
    {
      "table": "<table_name>",
      "column": "<column_name>",
      "issue": "<detected quality issue, e.g., high null rate, low uniqueness>",
      "recommended_action": "<actionable fix, e.g., median imputation, deduplication with UPSERT, regex validation>"
    }
  ],
  "overall_observations": [
    "<key takeaway 1>",
    "<key takeaway 2>"
  ]
}
"""



def perform_web_research(queries: List[str]) -> str:
    print(f"[EVALUATOR]: Performing web search...")
    if not queries:
        print("[EVALUATOR]: No queries found for the web researcher")
        return "No specific metrics identified for web research."

    combined_context = []
    for query in queries[:2]:  
        try:
            print(f"[EVALUATOR]: Running web research for: '{query}'...")
            results = web_search_tool.invoke(query)
            combined_context.append(f"Query [{query}]: {str(results)[:600]}")
        except Exception as e:
            print(f"[EVALUATOR]: Web research warning for '{query}': {e}")

    return "\n\n".join(combined_context) if combined_context else "Web research context unavailable."


def evaluator_node(state: AgentState) -> dict:
    print("[EVALUATOR]: Starting Evaluation Node...")

    session_id = state.get("session_id", "")
    execution_results = state.get("execution_results", {})
    eval_plan = state.get("evaluation_plan", {})

    if not execution_results:
        print("[EVALUATOR]: Execution results are empty.")
        fallback_report = {
            "plan_score": 0,
            "plan_status": "REJECTED",
            "semantic_evaluations": [],
            "data_remediation_recommendations": [],
            "overall_observations": ["No metrics were executed or provided."],
        }
        return {"evaluation_report": fallback_report, "messages": []}

    print(f"[EVALUATOR] execution_results:\n {execution_results}")

    if isinstance(execution_results, dict):
        results_list = execution_results.get("results", [])
    elif isinstance(execution_results, list):
        results_list = execution_results
    else:
        results_list = []


    vector_queries = []
    web_queries = []

    print(f"[EVALUATOR]: metrics:\n {results_list}")

    for item in results_list:
        if not isinstance(item, dict):
            continue

        table = item.get("table", "")
        col = item.get("column") or "all columns"
        desc = item.get("description", "")
        metric = item.get("requested_metric", "").strip()

        if metric:

            vector_queries.append(f"{desc} for table '{table}' column '{col}'")

            web_queries.append(f"industry benchmark for {metric} in table {table} column {col}")

    vector_queries = list(dict.fromkeys(vector_queries))
    web_queries = list(dict.fromkeys(web_queries))

    if not vector_queries:
        vector_queries = ["relational database data quality evaluation metrics completeness uniqueness validity"]

    paper_chunks = []
    vectordb = create_store()
    try:
        for query in vector_queries[:3]:
            print(f"[EVALUATOR]: Searching Vector KB for concept: '{query}'...")
            hits = vectordb.search_by_text(
                collection_name="ml_related_papers",
                query_text=query,
                top_k=2,
            )
            for hit in hits:
                payload = hit.get("payload", {})
                text = payload.get("text", payload.get("content", ""))
                source = payload.get("source", "Research Paper")
                if text:
                    paper_chunks.append(f"[Source: {source}]: {text.strip()}")
    except Exception as e:
        print(f"[EVALUATOR]: Qdrant vector search error: {e}")
    finally:
        vectordb.disconnect()

    web_context = perform_web_research(web_queries)

    llm = build_evaluator_llm()

    user_prompt = f"""Dataset Session ID: '{session_id}'
    Computed Execution Metric Results:{json.dumps(execution_results, indent=2)}

    Original Evaluation Plan:{json.dumps(eval_plan, indent=2)}

    Retrieved Research Paper Chunks (Vector KB): {json.dumps(paper_chunks, indent=2)}

    Web Research Context:
    {web_context}

    Audit all computed metrics against the paper excerpts and web context. Provide semantic evaluations, a plan score (0 to 10), and actionable dataset remediation recommendations. Return strictly valid JSON inside ```json blocks.
    """

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    print("[EVALUATOR]: Invoking Ollama LLM for evaluation...")
    response = None
    try:
        response = llm.invoke(messages)
        report_data = json.loads(response.content)
    except Exception as err:
        print(f"[EVALUATOR]: Failed to parse JSON or invoke LLM ({err}). Returning fallback dictionary.")
        report_data = {
            "plan_score": 5,
            "plan_status": "NEEDS_REVISION",
            "semantic_evaluations": [],
            "data_remediation_recommendations": [],
            "overall_observations": [
                "Audit output could not be parsed as valid JSON.",
            ],
        }

    print(f"[EVALUATOR] report:\n {report_data}")
    score = report_data.get("plan_score", 0)
    status = report_data.get("plan_status", "NEEDS_REVISION")
    print(f"[EVALUATOR]: Audit complete. Score: {score}/10 | Status: {status}")

    return {
        "evaluation_report": report_data,
        "messages": [response] if response else [],
    }