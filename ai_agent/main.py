import os, sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import List, TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage
from pymilvus import MilvusClient
import pandas as pd

from ai_agent.graphs.summarizer_graph import summarizer_graph
from agents.summarizer_agent import get_columns_by_driver
from agents.qna_summary_agent import Dependencies as qna_dependencies, qna_agent
from agents.master_agent import (master_agent, SUMMARIZATION_AGENT, QnA_AGENT, BOTH_AGENT, NONE)
from ai_agent.schemas import QnASuccess, QnAInvalidRequest

class AllState(TypedDict):
    request: Annotated[list[AnyMessage], add_messages]
    agent: str

    data_file: str
    map_file: Optional[str] = "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle driver qcode.xlsx"
    # df: pd.DataFrame
    # df_qcode: pd.DataFrame
    organization: str
    survey: str
    cycle: str

    db_path: Optional[str] = "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db"
    milvus_client: Optional[str] = "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db",
    collection_name: Optional[str] = "TGPS_transformation_model_action_recommendation"
    demographic_cols: Optional[List[str]] = ['Job Level', 'Length of Service', 'Function', 'Sub-Function'],
    
    driver_name: Optional[List[str]
]
    summary: Optional[str]
    survey_detail: Optional[str]
    answer: Optional[str]

    summarization_error: Optional[str]
    insert_data_error: Optional[str]

    qna_error = Optional[str]

def master_agent_node(state: AllState):
    master_agent_response = master_agent.run_sync(
        user_prompt=state["request"][-1].content,
        )
    return {
        "agent": master_agent_response.output.agent
    }

def summarizer_graph_node(state: AllState):
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

    driver_name = state["driver_name"]
    metric_columns = get_columns_by_driver(driver_name, ANALYSIS_CONFIG)

    initial_state = {
                        "request": state["request"][-1].content,
                        "data_file": state["data_file"],
                        "map_file": state["map_file"],
                        "db_path": state["db_path"],
                        "metric_columns": metric_columns,
                        # "demographic_cols": state["demographic_cols"],
                        "demographic_cols": ['Job Level', 'Length of Service', 'Function', 'Sub-Function'],

                        "organization": state["organization"],
                        "survey": state["survey"],
                        "cycle": state["cycle"],
                        "driver_name": driver_name,
                        
                        # If state already contains a MilvusClient instance, use it directly.
                        # Otherwise create a new MilvusClient from the provided connection (string/path).
                        # "milvus_client": MilvusClient(state["milvus_client"]),
                        "milvus_client": state["milvus_client"] if isinstance(state["milvus_client"], MilvusClient) else MilvusClient(state["milvus_client"]),
                        "collection_name": state["collection_name"]
                    }

    graph = summarizer_graph()
    output = graph.invoke(initial_state)
    return output

def qna_node(state: AllState):
    deps = qna_dependencies(
            data_file = state["data_file"],
            collection_name = "driver_insights_summary",
            organization = state["organization"],
            survey = state["survey"], 
            cycle = state["cycle"], 
            db_path = state["db_path"],
            # milvus_client = MilvusClient(state["milvus_client"]),
                # if isinstance(state.get("milvus_client"), MilvusClient)
                # else MilvusClient(state["milvus_client"]),
            milvus_client = state["milvus_client"] if isinstance(state["milvus_client"], MilvusClient) else MilvusClient(state["milvus_client"]),
            driver_name = state["driver_name"]
    )
    qna_agent_response = qna_agent.run_sync(
        user_prompt = state["request"][-1].content,
        deps=deps
    )
    # if isinstance(qna_agent_response.output, QnASuccess):
    #     print(qna_agent_response.output)
    #     return {
    #         "survey_detail": qna_agent_response.output.survey_detail,
    #         "answer": qna_agent_response.output.answer
    #     }
    # elif isinstance(qna_agent_response.output, QnAInvalidRequest):
    #     print("something wrong")
    #     return {
    #         "qna_error": qna_agent_response.output.error_message
    #     }
    # else:
    #     print("something wrong")
    #     return {
    #         "qna_error": "invalid schema returned"
    #     }
    # Diagnostic: print type information to understand why isinstance(...) may be False
    if isinstance(qna_agent_response.output, QnASuccess):
        # answer_val = qna_agent_response.output.answer if hasattr(qna_agent_response.output, "answer") else None
        return {
            "survey_detail": qna_agent_response.output.survey_detail,
            "answer": qna_agent_response.output.answer
        }
    elif isinstance(qna_agent_response.output, QnAInvalidRequest):
        return {
            "qna_error": qna_agent_response.output.error_message
        }
    else:
        return {
            "qna_error": "different schema returned"
        }

def decide_summarizer_or_qna(state: AllState):
    if state["agent"] == BOTH_AGENT:
        print("both agent")
        return ["summarizer_graph", "qna_agent"]
    elif state["agent"] == SUMMARIZATION_AGENT:
        print("summarization agent")
        return ["summarizer_graph"]
    elif state["agent"] == QnA_AGENT:
        print("qna agent")
        return ["qna_agent"]
    elif state["agent"] == NONE:
        print("no agent")
        return ["end"]
    else:
        return ["end"]
    
def create_graph():
    graph = StateGraph(AllState)

    graph.add_node("master", master_agent_node)
    graph.add_node("summarizer", summarizer_graph_node)
    graph.add_node("qna", qna_node)

    graph.add_conditional_edges(
        "master",
        decide_summarizer_or_qna,
        {
            "summarizer_graph": "summarizer",
            "qna_agent": "qna",
            "end": END
        }
    )

    graph.add_edge("summarizer", END)
    graph.add_edge("qna", END)

    graph.set_entry_point("master")

    return graph.compile()

graph = create_graph()

def main():
    from langchain_core.runnables.graph import MermaidDrawMethod

    # Ensure the output directory exists before attempting to write files into it.
    # The FileNotFoundError occurred because the "images" directory didn't exist.
    # os.makedirs("./images", exist_ok=True)

    # graph_png = graph.get_graph().draw_mermaid_png(output_file_path="./images/main.png",
    #     draw_method=MermaidDrawMethod.PYPPETEER,
    # )

    driver_name = ["Business Leadership"]

    initial_state = {
                    "request":
                        [HumanMessage(content="" \
                            f"Give insights of this dataset regarding driver {driver_name}?" \
                        "")],
                        "data_file": "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle.xlsx",
                        "map_file": "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle driver qcode.xlsx",
                        "driver_name": driver_name,
                        "demographic_cols": ['Job Level', 'Length of Service', 'Function', 'Sub-Function'],
                        "organization": "Accenture", 
                        "survey": "Transformation GPS Learning Activity Cycle", 
                        "cycle": "TGPS Learning Activity Cycle", 
                        "milvus_client": "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db",
                        "collection_name": "TGPS_transformation_model_action_recommendation",
                        "db_path": "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db"
                    }

    # with open("./images/main.png", "wb") as f:
    #     f.write(graph_png)

    for event in graph.stream(initial_state):
        for key in event:
            print("\n-----------------------------------")
            print("Done with " + key)
            print("\n***********************************\n")
    res = graph.invoke(initial_state)
    print(res)


if  __name__ == "__main__":
    main()
