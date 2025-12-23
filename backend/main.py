import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError
from pinecone import Pinecone, PineconeException

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize external clients
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    logger.critical(f"Failed to initialize external services: {e}")
    # Note: Depending on deployment, you might want to exit here or handle lazy loading.


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    source_id: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    question: Optional[str] = None
    correct_answer: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]


def embed_query(query: str) -> List[float]:
    """
    Generates an embedding vector for a single query string.
    """
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        return response.data[0].embedding
    except OpenAIError as e:
        logger.error(f"OpenAI API error during query embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External embedding service is unavailable."
        )
    except Exception as e:
        logger.error(f"Unexpected error during query embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the query."
        )


def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """
    Extracts filters (like difficulty) from the query string using simple keywords.
    This helps the vector search target specific subsets of data.
    """
    query_lower = query.lower()
    filters = {}

    # Difficulty detection
    if "difficult" in query_lower or "hard" in query_lower:
        filters["difficulty"] = "hard"
    elif "medium" in query_lower:
        filters["difficulty"] = "medium"
    elif "easy" in query_lower:
        filters["difficulty"] = "easy"

    # Category detection could be added here similar to difficulty
    
    return filters


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Handles the search request: embeds the query, searches Pinecone with optional filters, and returns relevant results.
    """
    try:
        query_vector = embed_query(request.query)
        
        # Extract metadata filters (e.g., if user asks for "difficult questions")
        filters = extract_filters_from_query(request.query)
        logger.info(f"Applied filters for query '{request.query}': {filters}")

        pinecone_response = index.query(
            vector=query_vector,
            top_k=TOP_K,
            include_metadata=True,
            filter=filters if filters else None
        )

        results: List[SearchResult] = []

        if pinecone_response and "matches" in pinecone_response:
            for match in pinecone_response["matches"]:
                score = match["score"]
                if score < SIMILARITY_THRESHOLD:
                    continue

                metadata = match.get("metadata", {}) or {}
                results.append(
                    SearchResult(
                        id=match["id"],
                        score=score,
                        text=metadata.get("text", ""),
                        source_id=metadata.get("source_id"),
                        category=metadata.get("category"),
                        difficulty=metadata.get("difficulty"),
                        question=metadata.get("question"),
                        correct_answer=metadata.get("correct_answer"),
                    )
                )

        # Check if the top result is significantly better than the runner-up
        if len(results) >= 2:
            top_score = results[0].score
            second_score = results[1].score
            if (top_score - second_score) > 0.2:
                logger.info("Top result is significantly better (>0.2 diff). Returning only the top result.")
                results = [results[0]]

        return SearchResponse(results=results)

    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except PineconeException as e:
        logger.error(f"Pinecone search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database search service is unavailable."
        )
    except Exception as e:
        logger.error(f"Unexpected error in search endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred."
        )
