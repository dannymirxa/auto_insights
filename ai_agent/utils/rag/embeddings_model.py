import os
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = "text-embedding-3-large"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

def embed_text(text: str):
    response = client.embeddings.create(
        input=text,
        model=deployment
    )
    return response.data[0].embedding

