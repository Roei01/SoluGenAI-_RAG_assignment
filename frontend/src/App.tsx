import { useState } from "react";
import { searchQuery } from "./api";
import type { SearchResult } from "./api";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    const trimmed = query.trim();
    if (!trimmed) {
      alert("Please enter a query");
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const res = await searchQuery(trimmed);
      setResults(res);
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Unknown error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>RAG Retrieval</h1>

      <div className="card">
        <label htmlFor="query">Enter your question:</label>
        <textarea
          id="query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="For example: cats and dogs"
        />

        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>

        {loading && <p className="status">Sending request to the server...</p>}
        {!loading && error && <p className="status error">{error}</p>}
        {!loading && !error && results.length > 0 && (
          <p className="status ok">
            Found {results.length} result{results.length > 1 ? "s" : ""}.
          </p>
        )}
      </div>

      <div className="card">
        <h2>Results</h2>
        {results.length === 0 && !loading && !error && (
          <p className="no-results">
            No results yet. Try asking a question above.
          </p>
        )}

        <div className="results">
          {results.map((item) => (
            <div className="result" key={item.id}>
              <div className="score">
                <strong>ID:</strong> {item.id} | <strong>Score:</strong>{" "}
                {item.score.toFixed(3)} | <strong>Source:</strong>{" "}
                {item.source_id || "-"}
              </div>
              <p>{item.text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
