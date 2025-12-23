export interface SearchResult {
  id: string;
  score: number;
  text: string;
  source_id?: string | null;
  category?: string | null;
  difficulty?: string | null;
  question?: string | null;
  correct_answer?: string | null;
}

export interface SearchResponse {
  results: SearchResult[];
}

// Backend URL (your FastAPI)
const API_BASE_URL = "http://127.0.0.1:8000";

export async function searchQuery(query: string): Promise<SearchResult[]> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  const data: SearchResponse = await response.json();
  return data.results ?? [];
}
