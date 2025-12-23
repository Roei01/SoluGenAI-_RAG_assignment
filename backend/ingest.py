import logging
import uuid
from typing import List, Dict

import pandas as pd
from openai import OpenAI, OpenAIError
from pinecone import Pinecone, ServerlessSpec, PineconeException
import kagglehub
from kagglehub import KaggleDatasetAdapter

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize clients
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    logger.critical(f"Failed to initialize API clients: {e}")
    raise


def load_quiz_dataset_from_kaggle() -> pd.DataFrame:
    """
    Loads the Open Trivia Database dataset from Kaggle into a pandas DataFrame.
    """
    try:
        file_path = "quiz_questions.csv"
        # The path should match the file name in the dataset
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "shreyasur965/open-trivia-database-quiz-questions-all-categories",
            file_path,
        )
        logger.info(f"Successfully loaded {len(df)} rows from Kaggle.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset from Kaggle: {e}")
        raise


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits the input text into fixed-size chunks based on character count with overlap.
    """
    try:
        text = text.strip()
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

            if start < 0:
                break

        return chunks
    except Exception as e:
        logger.error(f"Error during text chunking: {e}")
        return []


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text strings using the OpenAI API.
    """
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings
    except OpenAIError as e:
        logger.error(f"OpenAI API error during embedding generation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {e}")
        raise


def create_index_if_not_exists():
    """
    Checks if the Pinecone index exists and creates it if necessary.
    """
    try:
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]

        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Index '{PINECONE_INDEX_NAME}' not found. Creating...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,  # Matches text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )
            logger.info(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
        
        return pc.Index(PINECONE_INDEX_NAME)
    except PineconeException as e:
        logger.error(f"Pinecone error while managing index: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while managing index: {e}")
        raise


def main():
    """
    Main execution flow: loads data, processes chunks, generates embeddings, and uploads to Pinecone.
    """
    try:
        df = load_quiz_dataset_from_kaggle()
        # Limit to first 200 rows for demonstration/testing
        df = df.head(200)
        logger.info(f"Processing a subset of {len(df)} rows.")

        index = create_index_if_not_exists()

        # Clear existing data in the index
        try:
            index.delete(delete_all=True)
            logger.info("Existing index data cleared.")
        except Exception as e:
            logger.warning(f"Could not clear index (it might be empty): {e}")

        all_chunks: List[Dict] = []

        for idx, row in df.iterrows():
            try:
                source_id = f"quiz_{idx}"
                category = str(row.get("category", "")).strip()
                difficulty = str(row.get("difficulty", "")).strip()
                question = str(row.get("question", "")).strip()
                correct_answer = str(row.get("correct_answer", "")).strip()
                incorrect_answers = str(row.get("incorrect_answers", "")).strip()

                if not question:
                    continue

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
            except Exception as row_err:
                logger.warning(f"Skipping row {idx} due to error: {row_err}")
                continue

        logger.info(f"Total text chunks prepared: {len(all_chunks)}")

        # Re-verify index connection
        index = create_index_if_not_exists()

        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            try:
                batch = all_chunks[i : i + batch_size]
                texts = [item["text"] for item in batch]
                ids = [item["id"] for item in batch]

                vectors = embed_texts(texts)

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

                index.upsert(vectors=pinecone_vectors)
                logger.info(f"Upserted batch starting at index {i} ({len(pinecone_vectors)} items).")

            except Exception as batch_err:
                logger.error(f"Failed to upsert batch starting at {i}: {batch_err}")
                # Continue with next batch instead of stopping entirely
                continue

        logger.info("Data ingestion process completed.")

    except Exception as e:
        logger.critical(f"Critical error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
