from agents import Agent, Runner
from agents import set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
from agents import enable_verbose_stdout_logging
from connection import config
import cohere
from qdrant_client import QdrantClient

enable_verbose_stdout_logging()

load_dotenv()
set_tracing_disabled(disabled=True)




# Initialize Cohere client
cohere_client = cohere.Client(os.getenv("COHERE_MODEL_API"))
# Connect to Qdrant
qdrant = QdrantClient(
    url=os.getenv("QDRANT_VECTOR_DATABASE_URL_ENDPOINT"),
    api_key=os.getenv("QDRANT_VECTOR_DATABASE_API_KEY") 
)



def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


@function_tool
def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="Physical-ai-book-cluster",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]



agent = Agent(
    name="Assistant",
    instructions="""
You are an AI tutor for the Physical AI & Humanoid Robotics textbook.
To answer the user question, first call the tool `retrieve` with the user query.
Use ONLY the returned content from `retrieve` to answer.
If the answer is not in the retrieved content, say "I don't know".
""",
    tools=[retrieve]
)


result = Runner.run_sync(
    agent,
    input="what is physical ai?",
    run_config= config)


print(result.final_output)