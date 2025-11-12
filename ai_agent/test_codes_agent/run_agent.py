import sys
import os
# Ensure the project root is on sys.path so local packages like `ai_agent` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_agent.model import OPENAI_MODEL

from pydantic_ai import Agent

from dotenv import load_dotenv
load_dotenv(".env")

agent = Agent(
    model=OPENAI_MODEL,
    retries=3,
)

reply = agent.run_sync("What is love?")

print(reply)