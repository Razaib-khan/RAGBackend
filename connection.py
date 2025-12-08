import os
from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

load_dotenv()

hf_api_key = os.getenv("HF_API_KEY") # Renamed from gemini_api_key

# Check if the API key is present; if not, raise an error
if not hf_api_key:
    raise ValueError("HF_API_KEY is not set. Please ensure it is defined in your .env file.") # Updated check

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=hf_api_key, # Use the new hf_api_key
    base_url="https://router.huggingface.co/v1",
)

model = OpenAIChatCompletionsModel(
    model="moonshotai/Kimi-K2-Instruct-0905",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)