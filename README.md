# SoluGenAI-\_RAG_assignment

make .env file in backend/.env
in this struc:

PINECONE_API_KEY= xxxxxxx-api-key-xxxxxxxx
OPENAI_API_KEY= xxxxxxxxx-api-key-xxxxxxxx
EMBEDDING_MODEL=text-embedding-3-small
PINECONE_INDEX_NAME=my-rag-index
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.2

if you don't choose its the standart:
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

start project:

cd backend  
python -m venv venv
source venv/bin/activate # ב-Windows: venv\Scripts\activate
pip install -r requirements.txt

python ingest.py

uvicorn main:app --reload

open new terminal

cd frontend
npm i
npm run dev
