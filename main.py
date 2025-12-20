from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import time
import hashlib
from collections import OrderedDict
from typing import Dict, Tuple
from agent import agent # Import the agent instance
from agents import Runner # Import Runner from the agents framework
from connection import config # Import the run configuration

# ============================================================================
# CACHING AND RATE LIMITING CONFIGURATION
# ============================================================================

# LRU Cache with TTL (Time To Live)
class TTLCache:
    """
    Thread-safe LRU cache with TTL support.
    Automatically evicts least recently used items when size limit is reached.
    """
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache: OrderedDict[str, Tuple[dict, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        """Get value from cache if exists and not expired."""
        
        if key not in self.cache:
            self.misses += 1
            return None

        value, timestamp = self.cache[key]

        # Check if expired
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            self.misses += 1
            return None

        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: dict):
        """Add value to cache, evicting LRU item if necessary."""
        # Remove if already exists
        if key in self.cache:
            del self.cache[key]

        # Add new item
        self.cache[key] = (value, time.time())

        # Evict LRU if over size limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest (first) item

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "ttl": self.ttl
        }

# Rate limiter per IP
class RateLimiter:
    """
    Per-IP rate limiter using token bucket algorithm.
    """
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.user_requests: Dict[str, list] = {}

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for this IP."""
        current_time = time.time()

        # Initialize or get existing request history
        if client_ip not in self.user_requests:
            self.user_requests[client_ip] = []

        # Remove requests older than 1 minute
        self.user_requests[client_ip] = [
            req_time for req_time in self.user_requests[client_ip]
            if current_time - req_time < 60
        ]

        # Check if under limit
        if len(self.user_requests[client_ip]) >= self.requests_per_minute:
            return False

        # Add current request
        self.user_requests[client_ip].append(current_time)
        return True

    def get_wait_time(self, client_ip: str) -> float:
        """Get time to wait before next request is allowed."""
        if client_ip not in self.user_requests or not self.user_requests[client_ip]:
            return 0.0

        oldest_request = min(self.user_requests[client_ip])
        wait_time = 60 - (time.time() - oldest_request)
        return max(0.0, wait_time)

# Initialize cache and rate limiter
response_cache = TTLCache(max_size=100, ttl=3600)  # 100 items, 1 hour TTL
rate_limiter = RateLimiter(requests_per_minute=10)  # 10 requests per minute per IP

# Validate required environment variables (warning only, doesn't block startup)
REQUIRED_ENV_VARS = [
    "OPENROUTER_API_KEY",
    "COHERE_MODEL_API",
    "QDRANT_VECTOR_DATABASE_URL_ENDPOINT",
    "QDRANT_VECTOR_DATABASE_API_KEY",
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
    print("⚠️  The application may not function correctly without these variables.")

app = FastAPI()

response_cache = {}

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

@app.get("/stats")
async def get_stats():
    """
    Get cache and rate limiting statistics.
    """
    cache_stats = response_cache.stats()
    return {
        "cache": cache_stats,
        "rate_limiter": {
            "requests_per_minute": rate_limiter.requests_per_minute,
            "active_ips": len(rate_limiter.user_requests)
        },
        "message": "Cache stats - Higher hit rate = fewer API calls = lower costs!"
    }

def normalize_query(query: str) -> str:
    """
    Normalize query for better cache hits.
    - Convert to lowercase
    - Remove extra whitespace
    - Strip punctuation at end
    """
    normalized = query.lower().strip()
    normalized = ' '.join(normalized.split())  # Remove extra whitespace
    # Remove trailing punctuation
    while normalized and normalized[-1] in '!?.,:;':
        normalized = normalized[:-1]
    return normalized

def get_cache_key(query: str) -> str:
    """Generate cache key from normalized query."""
    normalized = normalize_query(query)
    # Use hash for very long queries to save memory
    if len(normalized) > 200:
        return hashlib.md5(normalized.encode()).hexdigest()
    return normalized

@app.post("/query")
async def process_query(query: dict, request: Request):
    """
    Processes a user query using the RAG agent with caching and rate limiting.
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

    # Get client IP for rate limiting
    client_ip = request.client.host

    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        wait_time = rate_limiter.get_wait_time(client_ip)
        return JSONResponse(
            status_code=429,
            content={
                "error": f"Rate limit exceeded. Please wait {int(wait_time)} seconds.",
                "retry_after": int(wait_time)
            }
        )

    # Generate cache key
    cache_key = get_cache_key(user_query)

    # Check cache first
    cached_response = response_cache.get(cache_key)
    if cached_response:
        print(f"✅ Cache HIT for query: {user_query[:50]}...")
        # Add cache hit indicator to response
        cached_response["cached"] = True
        return cached_response

    print(f"❌ Cache MISS for query: {user_query[:50]}...")

    try:
        # Run the RAG agent with the user's query asynchronously
        print(f"🤖 Running agent for query: {user_query[:50]}...")
        result = await Runner.run(
            agent,
            input=user_query,
            run_config=config
        )
        # Extract the final output from the agent
        response_text = result.final_output
        print(f"✅ Agent response: {response_text[:100]}...")

        response = {
            "response": response_text,
            "cached": False
        }

        # Store in cache
        response_cache.set(cache_key, response)
        print(f"💾 Cached response for: {user_query[:50]}...")

        return response
    except Exception as e:
        # Handle any exceptions that occur during the agent's execution
        import traceback
        print(f"Error processing query with RAG agent: {e}")
        traceback.print_exc()

        # Check for specific error types and return appropriate messages
        error_message = str(e)
        status_code = 500

        # Check if it's a rate limit error
        if "429" in error_message or "quota" in error_message.lower() or "rate limit" in error_message.lower():
            error_message = "⚠️ API quota exceeded. The free tier limit has been reached. Please try again later or contact support to upgrade."
            status_code = 429
        # Check if it's an authentication error
        elif "401" in error_message or "unauthorized" in error_message.lower() or "authentication" in error_message.lower():
            error_message = "Authentication error. Please check API credentials."
            status_code = 500
        # Check if it's a connection error
        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
            error_message = "Failed to connect to the AI service. Please try again."
            status_code = 503
        # Generic error - show partial error for debugging but keep it user-friendly
        else:
            error_message = f"An error occurred while processing your query. Details: {error_message[:200]}"

        return JSONResponse(
            status_code=status_code,
            content={"error": error_message}
        )

# You can import and use functions from ingestion.py here if needed for API endpoints
# For example, to trigger ingestion via an API endpoint (though typically not recommended for public exposure)
# from .ingestion import ingest_book
# @app.post("/ingest")
# async def trigger_ingestion():
#     ingest_book()
#     return {"message": "Ingestion process started."}
