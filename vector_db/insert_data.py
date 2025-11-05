import os, sys
# ensure project root (drivers_insights) is on sys.path so imports like `ai_agent...` resolve
# dirname(__file__) -> .../vector_db; dirname(dirname(__file__)) -> .../drivers_insights (project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import pandas as pd
from pymilvus import MilvusClient, Collection
from ai_agent.utils.rag.embeddings_model import embed_text

def insert_chat_into_vector_db(client: MilvusClient, collection_name: str, file_name: str, \
                                organization: str, survey: str, cycle: str,\
                                question: str, driver_name: str, summary: str) -> str:

    try:
        summarization =  f"""
                Question: {question}

                Summary: {summary}
                """

        data = [{
                    "file_name": file_name,
                    "organization": organization,
                    "survey": survey,
                    "cycle": cycle,
                    "driver_name": driver_name,
                    "summary": summarization,
                    "vector": embed_text(summarization),
                    "created_at": int(datetime.now().timestamp())
                }]

        client.insert(collection_name=collection_name, data=data)
        return "insert data successfully"
    except Exception as e:
        return str(e)


client = MilvusClient("/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db")

# kwargs = {
#     "client": client,
#     "collection_name": "driver_insights_summary",
#     "file_name": "/mnt/c/Projects/drivers_insights/data/TGPS Learning Activity Cycle.xlsx",
#     "organization": "Accenture", 
#     "survey": "Transformation GPS Learning Activity Cycle", 
#     "cycle": "TGPS Learning Activity Cycle", 
#     "question": "Summarize this file for driver Vision and Direction?", 
#     "driver_name": "Vision and Direction",
#     "summary": """## 1. Executive Summary
#         The analysis of the Vision and Direction driver reveals a mix of strengths and weaknesses across different functions and job levels. The overall mean scores for key metrics under this driver are relatively balanced, with some notable variations.

#         ### Key Observations
#         - The mean scores for key metrics are as follows: understanding the vision (5.27), agreeing with the vision (5.34), confidence in leadership (5.01), awareness of the need for change (4.90), and understanding the purpose (4.54).
#         - The Executive Leadership Team and Marketing function emerged as bright spots with high average scores, while Local Global Organization and Supply Chain/Procurement functions are identified as areas of concern.

#         ## 2. Bright Spots
#         - **Functions:**
#         - Executive Leadership Team: Highest scores in understanding the vision (6.57), agreeing with the vision (6.14), confidence in leadership (6.29), awareness of the need for change (5.86), and understanding the purpose (6.29).
#         - Marketing: Second highest scores across several metrics with scores like understanding the vision (5.84) and agreeing with the vision (5.98).
#         - **Job Levels:**
#         - Executive: Highest average score (6.23).
#         - VP: Second highest average score (5.45).

#         ## 3. Hotspots
#         - **Functions:**
#         - Local Global Organization: Lowest scores across several metrics including understanding the vision (5.05) and agreeing with the vision (4.55).
#         - Supply Chain/Procurement: Scores like understanding the vision (5.11) and awareness of the need for change (4.71).
#         - **Job Levels:**
#         - Salaried Professional: Lower scores with an average of 4.93.
#         - Manager: An average score of 5.02.

#         ## 4. Metric Correlations
#         - There is a strong correlation between understanding the vision and agreeing with the vision (0.66), suggesting alignment between these two aspects.
#         - Moderate correlations are observed between confidence in leadership and understanding the vision (0.49) and agreeing with the vision (0.72).
#         - Fair correlation exists between awareness of the need for change and understanding the vision (0.34).

#         ## 5. Correlations with Other Drivers
#         - Positive correlation: Strongest with the perception of how well the change process is being led (0.66), indicating that effective change management enhances the perception of vision and direction.
#         - Negative correlation: Issues such as inadequate systems and tools (-0.14), conflicting priorities (-0.18), and lack of management support (-0.40) adversely affect the perception of vision and direction.

#         ## 6. Recommendations
#         - **Enhance communication and support**: Leverage the Executive Leadership Team's exemplary scores by involving them in mentoring and sharing best practices across other functions.
#         - **Address low-scoring areas**: Conduct targeted interventions in the Local Global Organization and Supply Chain/Procurement functions to address specific challenges and improve employee sentiment.
#         - **Foster alignment across levels**: Implement programs to boost confidence and trust in leadership at the Salaried Professional and Manager levels, aligning them more closely with higher levels.
#         - **Focus on change management**: Strengthen change management practices to enhance overall perceptions of vision and direction, ensuring adequate systems, tools, and management support are in place.

#         By addressing these areas, the organization can leverage its strengths and address its weaknesses, driving a more cohesive and aligned approach to its vision and direction.""",
        
# }

# print(insert_chat_into_vector_db(**kwargs))

# def clean_text(text: str) -> str:
#     # Strip each line, drop empties, then rejoin with a single newline
#     lines = [line.strip() for line in text.splitlines() if line.strip()]
#     return "\n".join(lines)


# search_res = client.search(
#     collection_name= "driver_insights_summary",
#     # embed_text(...) returns a single vector (list[float]). Milvus expects a list of vectors, so wrap it.
#     data=[embed_text("Summarize this file for driver Vision and Direction?")],
#     limit=3,
#     filter= """
#         organization like "Accenture%" and
#         survey like "%Transformation GPS Learning Activity Cycle%" and
#         cycle like "%TGPS Learning Activity Cycle%" and
#         driver_name == "Vision and Direction"
#     """,
#     output_fields=["summary", "created_at"]
# )

# import json

# res = [
#         {
#             "created_at": str(datetime.fromtimestamp(hit["entity"]["created_at"])),
#             "summary": clean_text(hit["entity"]["summary"])
#         }
#         for hit in search_res[0]
# ]

# sorted_hits = sorted(res, key=lambda h: h["created_at"], reverse=True)


# print(json.dumps(sorted_hits, indent=2))

# dict(sorted(search_res.items(), key=lambda x: x["created_at"], reverse=True)[:1])

# response = pd.DataFrame.from_dict(res)

# context = "\n".join(
#         ["time: " + line_with_distance[0] + "\n" + line_with_distance[1] for line_with_distance in response]
#     )


# print(json.dumps(sorted_hits, indent=2))
# print(sorted(search_res, key=lambda x: x["created_at"], reverse=True)[:1])