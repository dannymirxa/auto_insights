import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import List, TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage
from pymilvus import MilvusClient
import re, io
import pandas as pd

load_dotenv()

from agents.summarizer_agent import Dependencies as summarizer_dependencies, summarization_agent, preprocess_data, get_columns_by_driver
from ai_agent.schemas import SummarySuccess, SummaryInvalidRequest
from vector_db.insert_data import insert_chat_into_vector_db

class AllState(TypedDict):
    request: Annotated[list[AnyMessage], add_messages]

    data_file: str
    map_file: Optional[str] = "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle driver qcode.xlsx"
    # df: pd.DataFrame
    # df_qcode: pd.DataFrame
    db_path: Optional[str] = "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db"
    milvus_client: Optional[MilvusClient]
    collection_name: Optional[str] = "TGPS_transformation_model_action_recommendation"
    metric_columns: List[str]
    demographic_cols: Optional[List[str]]

    organization: str
    survey: str
    cycle: str
    driver_name: Optional[List[str]]

    # summary should be a typed field, not an assignment. Use Optional[str] so
    # state updates that add "summary" (from the node return) are accepted and
    # typed correctly.
    summary: Optional[str]
    summarization_error: Optional[str]
    insert_data_error: Optional[str]

def summerizer_node(state: AllState):
    df, df_qcode = preprocess_data(state["data_file"], state["map_file"])
    deps = summarizer_dependencies(
        data_file=state["data_file"],
        df=df,
        df_qcode=df_qcode,
        driver_name=state["driver_name"],
        metric_columns=state["metric_columns"],
        demographic_cols=state["demographic_cols"],
        milvus_client=state["milvus_client"],
        collection_name=state["collection_name"]
    )

    # The Agent.run_sync expects the question as the first positional argument
    # (see usage in ai_agent/agents/summarizer_agent.py -> main). Passing a
    # keyword like `user_prompt` can result in the Agent returning no output.
    summarizer_agent_response = summarization_agent.run_sync(
        state["request"][-1].content,
        deps=deps
    )

    if isinstance(summarizer_agent_response.output, SummarySuccess):
        return {
            "summary": summarizer_agent_response.output.summary,
        }
    else:
        return {
            "summarization_error": summarizer_agent_response.output.error_message
        }

    # if hasattr(summarizer_agent_response, "output") and isinstance(summarizer_agent_response.output, SummarySuccess):
    #     summary_val = summarizer_agent_response.output.summary if hasattr(summarizer_agent_response.output, "summary") else None
    #     return {
    #         "summary": str(summary_val) if summary_val is not None else None,
    #         "summarization_error": None
    #     }
    # elif isinstance(summarizer_agent_response.output, SummaryInvalidRequest):
    #     return {
    #         "summarization_error": summarizer_agent_response.output.error_message
    #     }

    # else:
    #     return {
    #         "summarization_error": "different schema returned"
    #     }

def insert_data_into_db_node(state: AllState):
    """
    Insert the summary into the vector DB. If multiple driver names are provided,
    loop and insert one record per driver.
    """
    driver_field = state.get("driver_name")
    driver_list = driver_field if isinstance(driver_field, list) else [driver_field]

    errors = []
    for dn in driver_list:
        dn_str = dn if dn is not None else ""
        result = insert_chat_into_vector_db(
            client=state["milvus_client"],
            collection_name="driver_insights_summary",
            file_name=state["data_file"],
            organization=state["organization"],
            survey=state["survey"],
            cycle=state["cycle"],
            question=state["request"][-1].content,
            driver_name=dn_str,
            summary=state["summary"]
        )
        if not (isinstance(result, str) and result.strip().lower() == "insert data successfully"):
            errors.append(str(result))

    if not errors:
        return {}
    else:
        return {
            "insert_data_error": "; ".join(errors)
        }

def check_if_errors_occurs(state: AllState):
    # If there's no summarization_error (i.e. summary succeeded), we should
    # proceed to insert the data. If there is an error, end the flow.
    if state.get("summarization_error") is None or state.get("summarization_error") == "":
        # print("go_to_insert_data_into_db")
        return "go_to_insert_data_into_db"
    else:
        # print("end")
        return "end"

def summarizer_graph():
    graph = StateGraph(AllState)

    graph.add_node("summerizer_agent", summerizer_node)
    graph.add_node("insert_data_into_db", insert_data_into_db_node)

    graph.add_conditional_edges("summerizer_agent",
                                    check_if_errors_occurs,
                                        {
                                            "go_to_insert_data_into_db": "insert_data_into_db",
                                            "end": END
                                        }
                                )
    
    graph.add_edge("insert_data_into_db", END)
    graph.set_entry_point("summerizer_agent")

    return graph.compile()

graph = summarizer_graph()

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

# print(get_columns_by_driver(["Vision and Direction", "Communication"], ANALYSIS_CONFIG))

# def main():

#     from langchain_core.runnables.graph import MermaidDrawMethod

#     # graph_png = graph.get_graph().draw_mermaid_png(output_file_path="ai_agent/graphs/summarizer_graph.png",
#     #     draw_method=MermaidDrawMethod.PYPPETEER,
#     # )

#     driver_name = ["Communication"]
#     metric_columns = get_columns_by_driver(driver_name, ANALYSIS_CONFIG)

#     initial_state = {
#                         "request":
#                             [HumanMessage(content="" \
#                                 "Give insights of this dataset" \
#                             "")],
#                         "data_file": "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle.xlsx",
#                         "map_file": "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle driver qcode.xlsx",
#                         "driver_name": driver_name,
#                         "metric_columns": metric_columns,
#                         "demographic_cols": ['Job Level', 'Length of Service', 'Function', 'Sub-Function'],
#                         "organization": "Accenture", 
#                         "survey": "Transformation GPS Learning Activity Cycle", 
#                         "cycle": "TGPS Learning Activity Cycle", 
#                         "milvus_client": MilvusClient( "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db"),
#                         "collection_name": "TGPS_transformation_model_action_recommendation"
#                     }

#     # with open("ai_agent/graphs/summarizer_graph.png", "wb") as f:
#     #     f.write(graph_png)

#     # for event in graph.stream(initial_state):
#     #     for key in event:
#     #         print("\n-----------------------------------")
#     #         print("Done with " + key)
#     #         print("\n***********************************\n")
#     res = graph.invoke(initial_state)
#     print(res)


# if  __name__ == "__main__":
#     main()
