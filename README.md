<h1>SoluGenAI – RAG Assignment</h1>

<p>
This project demonstrates a <strong>Retrieval-Augmented Generation (RAG)</strong> pipeline
using <strong>OpenAI</strong> and <strong>Pinecone</strong>, with a Python backend and a frontend client.
</p>

<hr />

<h2>🛠 Prerequisites</h2>

<ul>
  <li>
    <strong>Python</strong> must be installed on your machine
    (<code>Python 3.9+</code> recommended)
  </li>
  <li>
    A valid <strong>Pinecone account</strong> to obtain an API key
    (<a href="https://app.pinecone.io" target="_blank">https://app.pinecone.io</a>)
  </li>
  <li>
    A valid <strong>OpenAI account</strong> to obtain an API key
    (<a href="https://platform.openai.com" target="_blank">https://platform.openai.com</a>)
  </li>
</ul>

<p>
Make sure you have generated both API keys before continuing.
</p>

<hr />

<h2>Environment Variables</h2>

<p>
Create a <code>.env</code> file inside the <code>backend/</code> directory.
</p>

<h3>Required structure</h3>

<pre><code>
PINECONE_API_KEY=xxxxxxxx-api-key-xxxxxxxx
OPENAI_API_KEY=xxxxxxxx-api-key-xxxxxxxx

EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.2
</code></pre>

<p><strong>⚠ Important:</strong><br />
<code>PINECONE_API_KEY</code> and <code>OPENAI_API_KEY</code> are mandatory.
</p>

<hr />

<h2>Default Configuration</h2>

<p>
If you don’t explicitly set these values, the system will use the following defaults:
</p>

<pre><code>
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
</code></pre>

<hr />

<h2>▶ Getting Started</h2>

<h3>1️⃣ Clone the repository</h3>

<pre><code>
git clone https://github.com/Roei01/SoluGenAI-_RAG_assignment.git
</code></pre>

<p>
Add your API keys according to the instructions above.
</p>

<hr />

<h3>2️⃣ Backend Setup</h3>

<pre><code>
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
</code></pre>

<h4>Ingest the data</h4>

<pre><code>
python ingest.py
</code></pre>

<h4>Run the backend server</h4>

<pre><code>
uvicorn main:app --reload
</code></pre>

<hr />

<h3>3️⃣ Frontend Setup <small>(New Terminal)</small></h3>

<pre><code>
cd frontend
npm install
npm run dev
</code></pre>
