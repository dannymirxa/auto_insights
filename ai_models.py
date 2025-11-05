
import os
from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from dotenv import load_dotenv

load_dotenv('.env')

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

client = AsyncAzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint="https://llmcoechangemateopenai2.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_KEY")
        )

OPENAI_MODEL = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(openai_client=client),
)
