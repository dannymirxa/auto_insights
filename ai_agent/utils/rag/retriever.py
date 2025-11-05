from datetime import datetime
from pymilvus import MilvusClient

try:
    # Prefer package-relative import when retriever.py is imported via ai_agent
    from .embeddings_model import embed_text
except ImportError:
    # Fallback to absolute import if executed in a different context
    from ai_agent.utils.rag.embeddings_model import embed_text

def retriever(milvus_client: MilvusClient, collection_name: str, question: str, timestamp: datetime = datetime.now()) -> str:
    # Convert the question into an embedding vector and perform a search
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            embed_text(question)
        ],  # Use the `embed_text` function to convert the question to an embedding vector
        limit=3,  # Return top 2 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        filter=f'created_at < {int(timestamp.timestamp())}',
        output_fields=["source_id", "text", "created_at"],  # Return the source and text fields
    )

    # Process the search results to extract relevant information
    retrieved_lines_with_distances = [
        (res["entity"]["source_id"], res["entity"]["created_at"], res["entity"]["text"]) for res in search_res[0]
    ]

    # Format the retrieved information into a readable context
    context = "\n".join(
        [line_with_distance[0] + ", " + str(datetime.fromtimestamp(line_with_distance[1])) + ": " + line_with_distance[2] for line_with_distance in retrieved_lines_with_distances]
    )
    return context

# question = "What is Vision and Direction?"
# print(retriever(milvus_client=MilvusClient(uri="/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db"), collection_name="TGPS_transformation_model_action_recommendation", question=question, timestamp=datetime.now()))

from dataclasses import dataclass
from typing_extensions import List, Optional
@dataclass
class Deps:
    organization: str
    survey: str
    cycle: str
    milvus_client: MilvusClient
    collection_name: str
    driver_name: str

def retriever_summary(
        organization: str,
        survey: str,
        cycle: str,
        driver_name: str,
        milvus_client: MilvusClient, 
        collection_name: str, 
        question: str, 
        timestamp: datetime = datetime.now()):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[embed_text(question)],
        limit=5,
        filter= f"""
                    organization like "%{organization}%" and
                    survey like "%{survey}%" and
                    cycle like "%{cycle}%" and
                    driver_name == "{driver_name}"
                """,
        output_fields=["summary", "created_at"]
    )
    res = [
        {
            "created_at": str(datetime.fromtimestamp(hit["entity"]["created_at"])),
            "summary": hit["entity"]["summary"]
        }
        for hit in search_res[0]
    ]

    sorted_hits = sorted(res, key=lambda h: h["created_at"], reverse=True)

    return sorted_hits

def retriever_tool(deps: Deps, query) -> str:
    return retriever_summary(
        organization=deps.organization,
        survey=deps.survey,
        cycle=deps.cycle,
        driver_name=deps.driver_name,
        milvus_client=deps.milvus_client, 
        collection_name=deps.collection_name, 
        question=f"{deps.driver_name}\n" + query, 
        timestamp=datetime.now())

milvus_client = MilvusClient(uri="/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db")

deps = Deps(
            organization = "Accenture",
            survey = "Transformation GPS Learning Activity Cycle", 
            cycle = "TGPS Learning Activity Cycle", 
            # db_path = "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db",
            milvus_client = milvus_client,
            driver_name = "Vision and Direction",
            collection_name = "driver_insights_summary"
        )   

# print(retriever_tool(deps, "What is communication?"))
