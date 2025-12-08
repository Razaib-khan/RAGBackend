from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from agent import agent # Import the agent instance
from agents import Runner # Import Runner from the agents framework
from connection import config # Import the run configuration

app = FastAPI()

# Configure CORS
# In a production environment, you should replace "*" with the actual
# origin(s) of your frontend application (e.g., "https://your-frontend-app.vercel.app")
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000", # Common for React development server
    "http://localhost:5173", # Common for Vite development server
    "*" # WARNING: This allows all origins. Restrict in production.
]

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
    if not user_query:
        return {"error": "No query text provided"}, 400

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
        return {"error": "An error occurred while processing your query."}, 500

# You can import and use functions from ingestion.py here if needed for API endpoints
# For example, to trigger ingestion via an API endpoint (though typically not recommended for public exposure)
# from .ingestion import ingest_book
# @app.post("/ingest")
# async def trigger_ingestion():
#     ingest_book()
#     return {"message": "Ingestion process started."}
