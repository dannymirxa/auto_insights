import sys
import os
from typing import List, Dict, Any, Optional
from typing_extensions import Optional
import pandas as pd
from dataclasses import dataclass
import logfire
from pydantic_ai import Agent, RunContext, ModelSettings
from pymilvus import MilvusClient
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(".env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_agent.model import OPENAI_MODEL
from ai_agent.utils.rag.retriever import retriever
from insights.data_process import data
from insights.drivers import correlations, findings, hotspots
from ai_agent.schemas import SummarySuccess, SummaryInvalidRequest, SummaryResponse

logfire.configure(token=os.getenv("LOGFIRE_API_KEY"))  
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

@dataclass
class Dependencies:
    data_file: str
    df: pd.DataFrame
    df_qcode: pd.DataFrame
    metric_columns: List[str]
    milvus_client: MilvusClient
    collection_name: str
    driver_name: Optional[List[str]] = None
    demographic_cols: Optional[List[str]] = None

summarization_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=Dependencies,
    output_type=SummaryResponse,
    retries=3,
    model_settings=ModelSettings(temperature=0.1)
)

def preprocess_data(data_path: str, map_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(data_path, skiprows=1)
    df_map = pd.read_excel(map_path)
    df_qcode = pd.read_excel(data_path, header=None)
    df_qcode = data.preprocess_question_qcode(df_qcode)
    df_driver_qcode = pd.DataFrame.merge(df_map, df_qcode, on="qcode", how="inner")

    return df, df_driver_qcode

def get_columns_by_driver(drivers: List[str], config: List[dict]) -> List[str]:
    # Combine metric columns for all specified drivers.
    # Unknown drivers are ignored. Duplicates are removed while preserving order.
    index = {item["driver_name"]: item["metric_columns"] for item in config}

    combined: List[str] = []
    for driver in drivers:
        cols = index.get(driver)
        if not cols:
            continue
        for col in cols:
            if col not in combined:
                combined.append(col)
    return combined

@summarization_agent.tool(require_parameter_descriptions=True)
def get_questions_to_qcode_map(ctx: RunContext[Dependencies]) -> Dict[str, Any]:
    """Return mapping of drivers, questions, and qcodes (Step 2 in the run plan).

    Args:
        ctx: RunContext containing Dependencies with attribute `df_qcode`.
    """
    try:
        return ctx.deps.df_qcode.to_dict(orient='records')
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"get_questions_to_qcode_map failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def data_availability(ctx: RunContext[Dependencies]) -> Dict[str, Any]:
    """Compute and return data availability metrics for the configured metric columns.

    Args:
        ctx: RunContext containing Dependencies with attributes `df` and `metric_columns`.
    """
    try:
        result = data.data_availability(ctx.deps.df, ctx.deps.metric_columns)
        return result
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"data_availability failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def mean_score_by_metric_columns(ctx: RunContext[Dependencies]) -> Dict[str, Any]:
    """Compute mean score for each metric column and attach driver and question labels.

    Args:
        ctx: RunContext containing Dependencies with attributes `df` and `metric_columns`.
    """
    try:
        df_metrics = ctx.deps.df[ctx.deps.metric_columns].mean().reset_index()
        df_metrics_renamed = df_metrics.rename(columns={"index": "qcode", 0: "mean_score"})
        df_driver_metrics_renamed = pd.DataFrame.merge(ctx.deps.df_qcode, df_metrics_renamed, on="qcode", how="inner").to_dict(orient="records")
        return df_driver_metrics_renamed
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"mean_score_by_metric_columns failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def hotspots_and_bright_spots_by_function_and_joblevel(ctx: RunContext[Dependencies], num_spots: int = 3) -> Dict[str, Any]:
    """Find hotspots and bright spots grouped by function and job level.

    Args:
        ctx: RunContext containing Dependencies with attributes `df` and `metric_columns`.
        num_spots: Number of top hotspots / bright spots to return per group.
    """
    try:
        result = hotspots.hotspots_and_bright_spots_by_function_and_joblevel(ctx.deps.df, ctx.deps.metric_columns, num_spots=num_spots)
        # results are already lists/dicts with native types; ensure numeric primitives are native
        return result
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"hotspots_and_bright_spots_by_function_and_joblevel failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def hotspots_and_bright_spots_by_function(ctx: RunContext[Dependencies], num_spots: int = 3) -> Dict[str, Any]:
    """Find hotspots and bright spots grouped by function.

    Args:
        ctx: RunContext containing Dependencies with attributes `df` and `metric_columns`.
        num_spots: Number of top hotspots / bright spots to return per function.
    """
    try:
        result = hotspots.hotspots_and_bright_spots_by_function(ctx.deps.df, ctx.deps.metric_columns, num_spots=num_spots)
        return result
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"hotspots_and_bright_spots_by_function failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def hotspots_and_bright_spots_by_joblevel(ctx: RunContext[Dependencies], num_spots: int = 3) -> Dict[str, Any]:
    """Find hotspots and bright spots grouped by job level.

    Args:
        ctx: RunContext containing Dependencies with attributes `df` and `metric_columns`.
        num_spots: Number of top hotspots / bright spots to return per job level.
    """
    try:
        result = hotspots.hotspots_and_bright_spots_by_joblevel(ctx.deps.df, ctx.deps.metric_columns, num_spots=num_spots)
        return result
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"hotspots_and_bright_spots_by_joblevel failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def hotspots_and_bright_spots_by_function_and_driver(ctx: RunContext[Dependencies], num_spots: int = 3) -> Dict[str, Any]:
    """Find hotspots and bright spots grouped by driver.

    Args:
        ctx: RunContext containing Dependencies with attributes `df` and `metric_columns`.
        num_spots: Number of top hotspots / bright spots to return per driver.
    """
    try:
        result = hotspots.hotspots_and_bright_spots_by_function_and_driver(ctx.deps.df, ctx.deps.metric_columns, num_spots=num_spots)
        return result
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"hotspots_and_bright_spots_by_function_and_driver failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def correlations_matrix(ctx: RunContext[Dependencies]) -> Dict[str, Any]:
    """Compute correlations matrix for the configured metric columns.

    Args:
        ctx: RunContext containing Dependencies with attributes `df`, `metric_columns`, and `df_qcode` (with ['driver','qcode','question']).
    """
    try:
        # Build mapping from qcode -> "[Driver] Question"
        mapping = {row["qcode"]: f"[{row['driver']}] {row['question']}" for _, row in ctx.deps.df_qcode.iterrows()}
        corr = correlations.correlations_matrix(ctx.deps.df, ctx.deps.metric_columns)
        corr_renamed = corr.rename(columns=mapping, index=mapping)
        return corr_renamed.to_dict(orient='records')
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"correlations_matrix failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def correlations_between_qcode_and_single_driver(ctx: RunContext[Dependencies], driver_name: str) -> Dict[str, Any]:
    """Compute correlations between the selected driver (mean of its questions) and questions not belonging to that driver.

    Args:
        ctx: RunContext containing Dependencies with attributes `df` and `df_qcode` (with ['driver','qcode','question']).
        driver_name: The driver to compute the mean over before correlating with all other questions.
    """
    try:
        # Identify qcodes belonging to the selected driver
        driver_qcodes = ctx.deps.df_qcode.loc[ctx.deps.df_qcode["driver"] == driver_name, "qcode"].tolist()

        if not driver_qcodes:
            return SummaryInvalidRequest(error_message=f"No qcodes found for driver '{driver_name}'")

        # Compute correlations using only the selected driver's qcodes as the "metric_columns" (target)
        df_result = correlations.correlations_between_qcode_and_single_driver(
            ctx.deps.df,
            ctx.deps.df_qcode,
            driver_qcodes
        )
        # Attach the driver of the correlated (other) questions and annotate target driver
        df_driver_result = pd.merge(df_result, ctx.deps.df_qcode[["qcode", "driver"]], on="qcode", how="left")
        df_driver_result["target_driver"] = driver_name
        # Reorder/select expected columns
        df_driver_result = df_driver_result[["target_driver", "driver", "question", "qcode", "score"]]
        return df_driver_result.to_dict(orient='records')
    except Exception as e:
        return SummaryInvalidRequest(error_message=f"correlations_between_qcode_and_single_driver failed: {str(e)}")

@summarization_agent.tool(require_parameter_descriptions=True)
def retriever_tool(ctx: RunContext[Dependencies], query: str) -> str:
    """Query the RAG retriever for contextual documents.

    Args:
        ctx: RunContext containing Dependencies with attributes `milvus_client`, `collection_name`, and `driver_name`.
        query: The natural-language query string to send to the retriever.

    """
    try:
        # Prepare a prefix of driver names (if provided) to increase relevance
        prefix = ctx.deps.driver_name
        if isinstance(prefix, list):
            prefix = ", ".join(prefix)
        prefix = prefix or ""
        full_query = f"{prefix}\n{query}" if prefix else query
        return retriever(ctx.deps.milvus_client, ctx.deps.collection_name, question=full_query, timestamp=datetime.now())
    except Exception as e:
        # Return an InvalidRequest model so the Agent can surface the error downstream
        return SummaryInvalidRequest(error_message=f"retriever_tool failed: {str(e)}")

@summarization_agent.system_prompt
def system_prompt():
    return """
    You are a Summarization Agent for executive-friendly reporting. Synthesize insights from organizational datasets in clear, concise language. Prefer human-readable question text over qcodes in narratives.

    Data model: df_qcode has ['driver', 'qcode', 'question']. Drivers group questions (qcodes). All insights must respect this grouping.
    
    qcode handling:
    - When tools include both 'question' and 'qcode', present the 'question' and optionally include the qcode in parentheses if needed. Do not fabricate question text. If only qcode is available, use it plainly.

    Failures:
    - If information is insufficient (e.g., missing sample size/demographics, tool failure), return SummaryInvalidRequest describing what's missing.

    Language:
    - Use clear, non-technical language suitable for executives and regular readers. Avoid jargon unless defined in retriever context or tool outputs.

    Threshold guidance (typical 1-7 scale; state assumptions if different):
    - >= 5.5: Good / Strong
    - 4.0-5.49: Mixed / Needs attention
    - < 4.0: Concerning / Underperformance

    Opinion, tone, and justification:
    - Give a labeled opinion for each key finding (Good / Mixed / Concerning).
    - Justify with a single key metric, comparison, or threshold.
    - Provide confidence (High / Medium / Low) based on robustness (sample size, variance, demographic consistency).
    - Recommendations must be actionable and prioritized.

    Retriever usage policy (strict):
    - Default stance: do NOT call retriever_tool unless a specific, documented gap is identified.
    - Allowed purposes only:
      1) Clarify the user's question intent/domain context when it cannot be inferred from df_qcode question text or analytical outputs.
      2) Define driver meanings when Dependencies.driver_name is provided and semantics are unclear from question text.
      3) Provide business-context guidance for recommendations when the dataset alone cannot justify actions.
    - Prohibitions:
      - Do NOT call retriever_tool for analytics (means, hotspots/bright spots, correlations) or per section/qcode.
      - Do NOT repeat retrieval for the same driver or purpose; reuse previously retrieved context.
    - Budget:
      - Max total retriever_tool calls per run: 3.
      - Typical budget: 1 for question intent, and up to 2 for drivers OR 1 for business-context recommendations as needed. Reuse retrieved content across all sections.
      - If Dependencies.driver_name > 3, limit per-driver retrieval to the top 2 drivers (by metric_columns coverage or hotspot prominence).
    - Pre-check:
      - Before any retrieval, ask: “Is there a concrete interpretation or recommendation gap that retrieval will close?” If not, skip. Prefer deriving meanings from df_qcode 'question' text.
    - Required logging in summary:
      - Add one header line at the very top: "Retrieval Calls Used: N; Purposes: [list]" to document decisions and discourage unnecessary calls.

    Execution plan (ordered, retrieval is optional and gated):
    1) If the pre-check passes, call retriever_tool with the user's full question (one call). If multiple drivers truly need definition, call retriever_tool for up to 2 drivers with "what is {driver_name}". Otherwise, skip retrieval and proceed.
    2) Call get_questions_to_qcode_map and use it to translate qcodes to questions and attach each question's driver.
    3) Derive 2-4 one-line meanings for key drivers/questions:
       - Prefer df_qcode 'question' text paraphrases; only use retrieved content when available and necessary.
    4) Use analytical tools (means, hotspots/bright spots, correlations). For each tool invoked, produce at least one concrete insight, explain why it matters, provide confidence, and an actionable recommendation when appropriate. Explicitly include driver-level extremes via hotspots_and_bright_spots_by_function_and_driver and report with translated question/driver labels.

    Length and simplicity:
    - Target ~1000 words; do not exceed 1200.
    - Executive Summary: 4-6 short items; at most one number per item.
    - Use short sentences and everyday words; avoid jargon.
    - Show only the most important numbers; round to 1 decimal.
    - In each section report only the top 2-3 highlights/issues; group the rest.

    Output:
    - Return SummarySuccess with summary, or SummaryInvalidRequest with error_message. Never return malformed objects.

    Structure (for SummarySuccess):
    0. Question Meanings (Context)
    - 2-4 bullets summarizing the practical meaning/intention of key drivers/questions (prefer df_qcode text; use retrieved context only if necessary).
    1. Executive Summary (Top {number} Impactful Findings)
    - 4-6 items: finding (use meaning first), evaluative judgment (Good / Concerning / Mixed), short rationale, confidence (High / Medium / Low), and one actionable recommendation.
    2. Bright Spots (Groups Exceeding Expectations)
    - 2.1 By Function and Job Level: top 2-3 strongest groups with one key score (rounded). Explain why strong.
    - 2.2 By Function: top 2-3 performers with brief interpretation and one tactical recommendation each.
    - 2.3 By Job Level: top 2-3 performers with brief interpretation and one tactical recommendation each.
    - 2.4 By Driver (Function-level extremes per question): use hotspots_and_bright_spots_by_function_and_driver; translate each qcode to 'question' and 'driver'; present 2-3 standout items as "[Driver] Question — best Function X at Y.Y; rationale and a tactical recommendation."
    Note: Bright-spot tools aggregate mean scores by demographic fields. Use human-readable question text in narratives.
    3. Hotspots (gGroups Needing Attention)
    - 3.1 By Function and Job Level (aggregate by both Function and Job Level): top 2-3 critical areas; impact and prioritized remediation with confidence.
    - 3.2 By Function (aggregate by Function only): brief interpretation, concise root-cause hypotheses, next steps.
    - 3.3 By Job Level (aggregate by Job Level only): brief interpretation, concise root-cause hypotheses, next steps.
    - 3.4 By Driver (Function-level extremes per question): weakest Function groups via hotspots_and_bright_spots_by_function_and_driver; translate qcodes; report 2-3 priority issues with concise remediation steps.
    Note: Hotspot tools aggregate by demographic fields; 3.1 uses Function and Job Level together; 3.2 uses Function only; 3.3 uses Job Level only. Use question text in narratives.
    4. Metric Correlations (Inter-Question Relationships)
    - Report the top 2-3 strongest relationships (by absolute value) with simple implications. Use labels in the form "[Driver] Question". Treat causal claims as hypotheses with suggested follow-ups.
    5. Correlations with Other Drivers (Selected Drivers)
    - Use correlations_between_qcode_and_single_driver to explain how a selected driver's mean relates to questions outside that driver. For each selected driver, report 2-3 key links with interpretation (use retrieved context only if necessary), confidence, and suggested experiments.
    6. Recommendations (Action Plan)
    - Provide 3-5 actionable, prioritized recommendations (High / Medium / Low) with brief rationale, owner, metric to monitor, and timeframe. Use business context from retrieval only if the dataset cannot justify actions.

    """

@summarization_agent.output_validator
def summarization_agent_output_validator(ctx: RunContext[Dependencies], output: SummaryResponse) -> SummaryResponse:
    """
    Validates the parsed output object from the Summary Agent.
    """
    if isinstance(output, SummaryInvalidRequest):
        print(f"Summary Agent Result Validator: Received InvalidRequest: {output.error_message}")
        return output
    
    if isinstance(output, SummarySuccess):
        if not output.summary:
            print("Summary Agent Result Validator: SummarySuccess has no summary.")
            return SummaryInvalidRequest(error_message="SummarySuccess has no summary.")
        
        print("Summary Agent Result Validator: SummarySuccess object passed custom validation.")
        return output

from vector_db.insert_data import insert_chat_into_vector_db

ANALYSIS_CONFIG = [
    {
        'driver_name': "Vision and Direction",
        'metric_columns': ["adb|vision_lv1", "adb|vision_agree_lv1", "adb|conf_lv1_ldr", "enb|awareness", "adb|understand_purpose"]
    },
    {
        'driver_name': "Communication",
        'metric_columns': ["adb|info_managers", "adb|info_written", "ada|info_intranet"]
    },
    {
        'driver_name': "Business Leadership",
        'metric_columns': ["enb|ldr_support_system", "enb|ldr_time_resources", "rsb|quick_remedial", "sfb|current_change_mgmt", "enb|conf_lv2_ldr"]
    }
]

drivers = ["Vision and Direction", "Communication"]
columns = get_columns_by_driver(drivers=drivers, config=ANALYSIS_CONFIG)

def main():
    data_file = "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle.xlsx"
    map_file = "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle driver qcode.xlsx"
    df, df_map_qcode = preprocess_data(data_file, map_file)
    milvus_client = MilvusClient(uri="/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db")
    deps = Dependencies(
            data_file = data_file,
            df = df,
            df_qcode = df_map_qcode,
            driver_name = drivers,
            metric_columns = columns,
            demographic_cols = ['Job Level', 'Length of Service', 'Function', 'Sub-Function'],
            milvus_client = milvus_client,
            collection_name = "TGPS_transformation_model_action_recommendation"
    )
    question = "Give insights for this file"
    # question = "Shut up"
    reply = summarization_agent.run_sync(question, deps=deps)
    
    # print(insert_chat_into_vector_db(
    #     client=milvus_client,
    #     collection_name="driver_insights_summary",
    #     file_name=deps.data_file,
    #     organization="Accenture", 
    #     survey="Transformation GPS Learning Activity Cycle", 
    #     cycle="TGPS Learning Activity Cycle", 
    #     question=question,
    #     driver_name=deps.driver_name,
    #     summary=reply.output.summary
    # ))

    print(reply.output.summary)

if __name__=="__main__":
    main()