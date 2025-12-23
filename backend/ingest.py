# ingest.py
"""
סקריפט חד-פעמי:
1. קורא קובץ CSV קטן מקומי
2. יוצר חתיכות (chunks) מהטקסט
3. מייצר embeddings בעזרת OpenAI
4. מעלה את הווקטורים + הטקסט ל-Pinecone
"""

import uuid
from typing import List, Dict

import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# יוצרים לקוחות ל-OpenAI ול-Pinecone
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    חותך טקסט לחתיכות בגודל קבוע (לפי תווים, לא לפי טוקנים).
    overlap - כמה תווים לחפוף בין חתיכה לחתיכה.
    """
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # זזים קדימה עם חפיפה

        if start < 0:
            break

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    מקבל רשימת טקסטים ומחזיר רשימת וקטורים (embeddings)
    ע"י שימוש במודל text-embedding-3-small.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # כל אובייקט ב-data מכיל embedding אחד
    embeddings = [item.embedding for item in response.data]
    return embeddings


def create_index_if_not_exists():
    """
    יוצר אינדקס ב-Pinecone אם הוא לא קיים.
    dimension חייב להתאים למימד של המודל (1536 לטקסט-אמבדינג-3-סמול).
    """
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # dim של text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return pc.Index(PINECONE_INDEX_NAME)


def main():
    # 1. טוענים דאטה-סט קטן (לדוגמה CSV)
    #    פה אתה מחליט איזה Dataset לקחת מ-Kaggle, מוריד אותו ושם ב-backend/data.csv
    df = pd.read_csv("data.csv")

    # נניח שיש עמודת טקסט בשם "text" (אתה יכול להתאים את השם)
    all_chunks: List[Dict] = []

    for _, row in df.iterrows():
        source_id = str(row.get("id", uuid.uuid4()))
        # בוחרים את העמודה המרכזית (למשל "text" או מחברים כמה עמודות)
        text = str(row.get("text", ""))

        if not text.strip():
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{source_id}_{idx}",
                    "text": chunk,
                    "source_id": source_id,
                }
            )

    print(f"Total chunks: {len(all_chunks)}")

    # 2. יוצרים אינדקס (אם לא קיים) ומקבלים אובייקט Index
    index = create_index_if_not_exists()

    # 3. עושים Embedding לחתיכות בקבוצות (batchים)
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]

        vectors = embed_texts(texts)

        # מכינים את המידע ל-upsert ל-Pinecone
        pinecone_vectors = []
        for vec, item_id, item in zip(vectors, ids, batch):
            pinecone_vectors.append(
                {
                    "id": item_id,
                    "values": vec,
                    "metadata": {
                        "text": item["text"],
                        "source_id": item["source_id"],
                    },
                }
            )

        # 4. מעלים ל-Pinecone
        index.upsert(vectors=pinecone_vectors)
        print(f"Upserted {len(pinecone_vectors)} vectors...")

    print("Done ingesting data.")


if __name__ == "__main__":
    main()
