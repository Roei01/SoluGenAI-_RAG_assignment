# ingest.py
"""
סקריפט חד-פעמי:
1. טוען דאטה-סט של שאלות טריוויה מ-Kaggle
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

# === NEW: KaggleHub imports ===
import kagglehub
from kagglehub import KaggleDatasetAdapter

# יוצרים לקוחות ל-OpenAI ול-Pinecone
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


def load_quiz_dataset_from_kaggle() -> pd.DataFrame:
    """
    טוען את הדאטה-סט של Open Trivia Database מ-Kaggle כ-DataFrame.
    שים לב: file_path צריך להתאים לשם הקובץ בדאטה-סט.
    אם השם שונה, פשוט תעדכן פה.
    """
    # אם הורדת מהאתר קובץ בשם quiz_questions.csv –
    # זה כמעט בוודאות אותו שם גם ב-Kaggle.
    file_path = "quiz_questions.csv"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "shreyasur965/open-trivia-database-quiz-questions-all-categories",
        file_path,
    )

    print(f"Loaded {len(df)} quiz rows from Kaggle")
    return df


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
    ע"י שימוש במודל שמוגדר ב-EMBEDDING_MODEL.
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
    dimension חייב להתאים למימד של המודל (למשל 1536 ל-text-embedding-3-small).
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
    df = load_quiz_dataset_from_kaggle()
    df = df.head(200)
    print(f"Using only first {len(df)} rows out of full dataset")

    index = create_index_if_not_exists()

    # 🧹 ניקוי האינדקס לפני העלאה מחדש
    index.delete(delete_all=True)
    print("Index cleared before ingesting new data.")

    # מצפים לעמודות כמו: category, type, difficulty, question, correct_answer, incorrect_answers
    all_chunks: List[Dict] = []

    for idx, row in df.iterrows():
        # נשתמש באינדקס של השורה בתור source_id בסיסי
        source_id = f"quiz_{idx}"

        category = str(row.get("category", "")).strip()
        difficulty = str(row.get("difficulty", "")).strip()
        question = str(row.get("question", "")).strip()
        correct_answer = str(row.get("correct_answer", "")).strip()
        incorrect_answers = str(row.get("incorrect_answers", "")).strip()

        # אם אין שאלה – אין מה לאנדקס
        if not question:
            continue

        # בונים טקסט אחיד שיהיה ברור למודל
        # (אתה יכול לשנות את הפורמט איך שתרצה)
        base_text = (
            f"Category: {category}\n"
            f"Difficulty: {difficulty}\n"
            f"Question: {question}\n"
            f"Correct answer: {correct_answer}\n"
            f"Incorrect answers: {incorrect_answers}"
        )

        chunks = chunk_text(base_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for c_idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{source_id}_{c_idx}",
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
