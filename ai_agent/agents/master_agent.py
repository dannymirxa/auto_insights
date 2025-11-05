import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from typing_extensions import Annotated, TypeAlias, Union, Optional
from annotated_types import MinLen
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pymilvus import MilvusClient

load_dotenv('.env')

from ai_agent.model import OPENAI_MODEL

# define which agent
SUMMARIZATION_AGENT = "summarization_agent"
QnA_AGENT = "qna_agent"
BOTH_AGENT = "both_agent"
NONE = 'none'

class MasterAgentResponse(BaseModel):
    agent: str

# @dataclass
# class MasterDependencies:
#     df: pd.DataFrame
#     df_qcode: pd.DataFrame
#     milvus_client: MilvusClient
#     collection_name: str

master_agent = Agent(
    model=OPENAI_MODEL,
    output_type=MasterAgentResponse,
    # deps_type=MasterDependencies,
    retries=3,
)

@master_agent.system_prompt
def master_system_prompt() -> str:
    return f"""\
    
    You are a master routing agent. Your task is to decide which specialized agent should handle the user's request.

    You must choose **exactly one** of the following:
    - `{SUMMARIZATION_AGENT}` → when the user primarily wants a summary, overview, condensed explanation, insights, or analysis of a dataset or text (examples: "summarize", "summary", "overview", "insights", "key takeaways", "drivers of").
    - `{QnA_AGENT}` → when the user is asking a targeted question, seeking factual information, clarification, or asking to retrieve/answer with respect to a previously supplied dataset or an ongoing "survey"/Q&A context.
    - `{BOTH_AGENT}` → when the user's intent clearly requires both summarization and answering specific questions (e.g., "Summarize this dataset and also tell me the key drivers of X").
    - `{NONE}` → when the request does not require summarization or Q&A (e.g., small talk, greetings, unsupported tasks).

    Important disambiguation rules (apply in order):
    1. If the user explicitly asks for a "summary", "summarize", "overview", "insights", "analysis", "key drivers", or similar terms about a dataset, file, or driver (for example: "I need a summary of this file with driver {{driver_name}}), choose `{SUMMARIZATION_AGENT}`.
    2. If the user explicitly references a previous "survey", "previous analysis", "prior conversation", or asks follow-up questions about an earlier Q&A session, prefer `{QnA_AGENT}`.
    3. If the user combines both explicit summary terms and specific questions in the same request, choose `{BOTH_AGENT}`.
    4. If intent is unclear and none of the above rules match, choose `{NONE}` rather than guessing.

    Additional guidelines:
    - Look for dataset-specific cues (e.g., file names, driver identifiers, dataset, survey) and analytical verbs (summary, analyze, insights) to map to `{SUMMARIZATION_AGENT}`.
    - Look for explicit references to past interactions, surveys, or follow-up questions to map to `{QnA_AGENT}`.
    - Always respond with only one of: `{SUMMARIZATION_AGENT}`, `{QnA_AGENT}`, `{BOTH_AGENT}`, `{NONE}`.
    """

@master_agent.output_validator
def validate_result(ctx: RunContext[None], response: MasterAgentResponse) -> MasterAgentResponse:
    if response.agent not in [SUMMARIZATION_AGENT, QnA_AGENT, BOTH_AGENT, NONE]:
        raise ModelRetry(
            f"Invalid action. Please choose from `{SUMMARIZATION_AGENT}`, `{QnA_AGENT}`, `{BOTH_AGENT}` or `{NONE}`"
        )

    return response