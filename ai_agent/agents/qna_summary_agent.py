import sys
import os
from typing import List, Dict, Any, Optional
from typing_extensions import Optional
import pandas as pd
from dataclasses import dataclass
import logfire
from pydantic_ai import Agent, RunContext
from pymilvus import MilvusClient
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(".env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_agent.model import OPENAI_MODEL
from ai_agent.utils.rag.embeddings_model import embed_text
from ai_agent.utils.rag.retriever import retriever
from src.insights import correlations, data, findings, hotspots
from ai_agent.schemas import QnASuccess, QnAInvalidRequest, QnAResponse

logfire.configure(token=os.getenv("LOGFIRE_API_KEY"))  
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

@dataclass
class Dependencies:
    data_file: str
    collection_name: str
    organization: str
    survey: str
    cycle: str
    # question: str
    db_path: Optional[str] = "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db"
    milvus_client: Optional[MilvusClient] = MilvusClient("/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db")
    driver_name: Optional[List[str]] = None
    demographic_cols: Optional[List[str]] = None

qna_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=Dependencies,
    output_type=QnAResponse,
    retries=3,
)

def clean_text(text: str) -> str:
    # Strip each line, drop empties, then rejoin with a single newline
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

@qna_agent.tool
def retriever_summary(ctx: RunContext[Dependencies], query: str) -> List[Dict[str, Any]]:
    """
    Retrieve recent summaries from Milvus filtered by organization/survey/cycle and,
    when provided, one search per driver_name. Results across drivers are combined,
    de-duplicated, and sorted by created_at descending.
    """
    # Normalize driver_name to a list
    driver_field = ctx.deps.driver_name
    drivers: List[str] = driver_field if isinstance(driver_field, list) else ([driver_field] if driver_field else [])
 
    all_hits: List[Dict[str, Any]] = []
 
    if drivers:
        # Run a separate search per driver to ensure strict filtering and relevant embeddings
        for dn in drivers:
            dn = dn or ""
            search_res = ctx.deps.milvus_client.search(
                collection_name=ctx.deps.collection_name,
                data=[embed_text(f"driver: {dn} \n" + query)],
                limit=5,
                filter=f"""
                            organization like "%{ctx.deps.organization}%" and
                            survey like "%{ctx.deps.survey}%" and
                            cycle like "%{ctx.deps.cycle}%" and
                            driver_name == "{dn}"
                        """,
                output_fields=["summary", "created_at"]
            )
            for hit in search_res[0]:
                all_hits.append(
                    {
                        "created_at": str(datetime.fromtimestamp(hit["entity"]["created_at"])),
                        "summary": clean_text(hit["entity"]["summary"]),
                        "driver_name": dn
                    }
                )
    else:
        # Fallback: no driver provided, search by general context only
        search_res = ctx.deps.milvus_client.search(
            collection_name=ctx.deps.collection_name,
            data=[embed_text(query)],
            limit=5,
            filter=f"""
                        organization like "%{ctx.deps.organization}%" and
                        survey like "%{ctx.deps.survey}%" and
                        cycle like "%{ctx.deps.cycle}%"
                    """,
            output_fields=["summary", "created_at"]
        )
        for hit in search_res[0]:
            all_hits.append(
                {
                    "created_at": str(datetime.fromtimestamp(hit["entity"]["created_at"])),
                    "summary": clean_text(hit["entity"]["summary"]),
                    "driver_name": ""
                }
            )
 
    # De-duplicate by (summary, created_at, driver_name)
    seen = set()
    unique_hits: List[Dict[str, Any]] = []
    for h in all_hits:
        key = (h["summary"], h["created_at"], h.get("driver_name", ""))
        if key not in seen:
            seen.add(key)
            unique_hits.append(h)
 
    # Sort by timestamp descending (parse string into datetime for robust ordering)
    def parse_dt(dt_str: str):
        try:
            return datetime.fromisoformat(dt_str)
        except Exception:
            # As a fallback, return minimal ordering; malformed dates sink to bottom
            return datetime.min
 
    sorted_hits = sorted(unique_hits, key=lambda h: parse_dt(h["created_at"]), reverse=True)
    return sorted_hits


@qna_agent.system_prompt
def system_prompt(ctx: RunContext[Dependencies] ):
    # Normalize driver label for display (handles list or single string)
    driver_label = ", ".join(ctx.deps.driver_name) if isinstance(ctx.deps.driver_name, list) else (ctx.deps.driver_name or "")
    return f"""
You are a QnA assistant that answers user questions about summaries retrieved from a Milvus vector store. Follow these rules exactly.

A. Retrieval (required)
 - For every user question, first call the tool retriever_summary with the user's exact question as its query argument.
 - If multiple drivers are provided, the tool will search once per driver and combine the results. Treat all returned snippets as your ONLY retrieval context.
 - Do NOT call any other external sources or rely on memory.

B. Forming your response (required)
 - Your final response must conform to the QnAResponse schema. Provide two distinct parts:
   1) survey_detail: A block that lists survey metadata lines exactly in this format (one per line):
       Organization: {ctx.deps.organization}
       Survey: {ctx.deps.survey}
       Cycle: {ctx.deps.cycle}
       Driver: {driver_label}
       Timestamp: ###
     - If multiple distinct surveys or drivers are present in the retrieved context, list each survey's metadata blocks one after another (separated by a blank line).
     - Use only information present in the retrieved context to fill these fields. If a field is not present in the retrieved context, omit that field line for that survey.
   2) answer: A concise, factual answer to the user's question (2-6 sentences) that uses ONLY the retrieved context.
 - If the retrieved context contains multiple snippets that inform the answer, synthesize them and explicitly cite the timestamps used in parentheses, e.g. (based on entries from 2025-01-01 12:00:00, 2025-02-02 13:30:00).
 - If the retrieved context does NOT contain enough information to answer, set survey_detail as extracted (if any) and set:
       answer: I don't know based on the provided summary.
   Do not invent facts.

C. Style and safety
 - Be concise and professional.
 - For recommendations, label them "Recommendations:" and ensure they are strictly grounded in the retrieved summaries.
 - If asked for sources, list only the timestamps from the retrieved context; do not create URLs or external citations.

D. Clarifying questions
 - If the user's question is ambiguous or too broad, ask one short clarifying question first. Do not call retriever_summary until the user answers.

E. Execution order
 1) (Optional) Ask a clarifying question if needed and wait for user's reply.
 2) Call retriever_summary with the user's exact question.
 3) Extract survey metadata and compose the QnAResponse with survey_detail and answer as described.

Always follow this retrieval-first, schema-driven process.
"""

@qna_agent.output_validator
def qna_agent_output_validator(ctx: RunContext[Dependencies], output: QnAResponse) -> QnAResponse:
    """
    Validates the parsed output object from the QnAAgent.
    """
    if isinstance(output, QnAInvalidRequest):
        # If pydantic-ai already determined it's an InvalidRequest, just return it.
        print(f"QnAAgent Result Validator: Received InvalidRequest: {output.error_message}")
        return output
    
    if isinstance(output, QnASuccess):
        # Perform additional validation on the QnASuccess object if needed.
        # For example, ensure critical fields are not empty or have expected formats.
        if not output.answer:
            print("QnAAgent Result Validator: No answer provided")
            return QnAInvalidRequest(error_message="No answer provided")
        
        print("QnAAgent Result Validator: QnASuccess object passed custom validation.")
        return output
    

# def main():
#     milvus_client = MilvusClient(uri="/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db")
#     deps = Dependencies(
#             data_file = "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle.xlsx",
#             collection_name = "driver_insights_summary",
#             organization = "Accenture",
#             survey = "Transformation GPS Learning Activity Cycle", 
#             cycle = "TGPS Learning Activity Cycle", 
#             db_path = "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db",
#             milvus_client = milvus_client,
#             driver_name = "Communication"
#     )
#     reply = qna_agent.run_sync("What are some highlights from the last survey regarding driver Communication?", deps=deps)
    
#     print(reply.output)

# if __name__=="__main__":
#     main()