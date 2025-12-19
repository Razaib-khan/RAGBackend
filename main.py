from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from agent import agent # Import the agent instance
from agents import Runner # Import Runner from the agents framework
from connection import config # Import the run configuration

# Validate required environment variables (warning only, doesn't block startup)
REQUIRED_ENV_VARS = [
    "GEMINI_API_KEY",
    "COHERE_MODEL_API",
    "QDRANT_VECTOR_DATABASE_URL_ENDPOINT",
    "QDRANT_VECTOR_DATABASE_API_KEY",
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
    print("⚠️  The application may not function correctly without these variables.")

app = FastAPI()

# Configure CORS
# Using wildcard for development. In production, replace with specific frontend URLs.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Backend!"}

@app.post("/query")
async def process_query(query: dict):
    """
    Processes a user query using the RAG agent.
    """
    user_query = query.get("text")

    # Validate query text exists and is not empty/whitespace
    if not user_query or not user_query.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "No query text provided"}
        )

    # Validate query length (defense-in-depth, frontend also validates)
    if len(user_query) > 1000:
        return JSONResponse(
            status_code=400,
            content={"error": "Query too long (max 1000 characters)"}
        )

    try:
        # Run the RAG agent with the user's query asynchronously
        result = await Runner.run(
            agent,
            input=user_query,
            run_config=config
        )
        # Extract the final output from the agent
        response_text = result.final_output
        return {"response": response_text}
    except Exception as e:
        # Handle any exceptions that occur during the agent's execution
        print(f"Error processing query with RAG agent: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred while processing your query."}
        )

# You can import and use functions from ingestion.py here if needed for API endpoints
# For example, to trigger ingestion via an API endpoint (though typically not recommended for public exposure)
# from .ingestion import ingest_book
# @app.post("/ingest")
# async def trigger_ingestion():
#     ingest_book()
#     return {"message": "Ingestion process started."}
