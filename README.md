# SoluGenAI-\_RAG_assignment

make .env file in backend/.env
in this structure: (must to be OPENAI_API_KEY and OPENAI_API_KEY)
״
PINECONE_API_KEY= xxxxxxx-api-key-xxxxxxxx
OPENAI_API_KEY= xxxxxxxxx-api-key-xxxxxxxx

EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.2
״

if you don't choose its the standart:
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3"

to start project:
open the terminal:
Git clone https://github.com/Roei01/SoluGenAI-_RAG_assignment.git
להוסיף את הapikey לפי קובץ readme
cd backend  
python -m venv venv
source venv/bin/activate # ב-Windows: venv\Scripts\activate
pip install -r requirements.txt

python ingest.py

uvicorn main:app --reload

open a new terminal:
cd frontend
npm i
npm run dev
