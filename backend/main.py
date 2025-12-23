from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
)

app = FastAPI()

# מאפשר ל-frontend בדפדפן לקרוא ל-API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # לפיתוח אפשר הכל, בפרודקשן כדאי לצמצם
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    source_id: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]


def embed_query(query: str) -> List[float]:
    """
    מייצר embedding לשאילתת משתמש יחידה.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    מקבל query, מחפש ב-Pinecone, ומחזיר את החתיכות הרלוונטיות.
    לא משתמש ב-LLM – רק אחזור (Retrieval).
    """
    query_vector = embed_query(request.query)

    # שאילתא ל-Pinecone
    pinecone_response = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )

    results: List[SearchResult] = []

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
            )
        )

    return SearchResponse(results=results)
