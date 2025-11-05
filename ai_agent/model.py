import os
from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

load_dotenv('.env')

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

client = AsyncAzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=AZURE_OPENAI_KEY
)

OPENAI_MODEL = OpenAIChatModel(
    'gpt-5',
    provider=OpenAIProvider(openai_client=client),
)