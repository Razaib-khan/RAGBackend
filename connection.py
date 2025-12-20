import os
from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Check if the API key is present; if not, raise an error
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set. Please ensure it is defined in your .env file.")

# Reference: https://openrouter.ai/docs
# OpenRouter provides an OpenAI-compatible API
external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# You can choose from various models available on OpenRouter
# Examples: "meta-llama/llama-3.1-70b-instruct", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash"
# See available models at: https://openrouter.ai/models
model = OpenAIChatCompletionsModel(
    model="meta-llama/llama-3.1-70b-instruct",  # You can change this to any OpenRouter model
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=False
)