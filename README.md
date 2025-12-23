SoluGenAI – RAG Assignment

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using OpenAI and Pinecone, with a Python backend and a frontend client.

Environment Variables

Create a .env file inside the backend/ directory.

Required structure
PINECONE_API_KEY=xxxxxxxx-api-key-xxxxxxxx
OPENAI_API_KEY=xxxxxxxx-api-key-xxxxxxxx

EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.2

Important
PINECONE_API_KEY and OPENAI_API_KEY are mandatory.

Default Configuration

If you don’t explicitly set these values, the system will use the following defaults:

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

▶Getting Started
1️⃣ Clone the repository
git clone https://github.com/Roei01/SoluGenAI-_RAG_assignment.git

Add your API keys according to the instructions above.

2️⃣ Backend Setup
cd backend
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Ingest the data
python ingest.py

Run the backend server
uvicorn main:app --reload

3️⃣ Frontend Setup (New Terminal)
cd frontend
npm install
npm run dev
