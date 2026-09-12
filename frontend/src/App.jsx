import { useState } from "react";
import axios from "axios";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [explain, setExplain] = useState(false);
  const [loading, setLoading] = useState(false);

  const doSearch = async () => {
    if (!query) return;
    setLoading(true);
    try {
      const endpoint = explain ? "/explain" : "/search";
      const { data } = await axios.post(
        `http://127.0.0.1:8000${endpoint}`,
        { query, size: 10 }
      );
      setResults(explain ? data.explanations : data.results);
    } catch (e) {
      console.error(e);
      alert("Could not reach the API – is the FastAPI server running?");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 800, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>ETHS – Explainable Temporal Search</h1>
      <div style={{ marginBottom: "1rem" }}>
        <input
          type="text"
          placeholder="Enter a historical query…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: "70%", padding: "0.5rem" }}
        />
        <button onClick={doSearch} disabled={loading} style={{ marginLeft: "0.5rem", padding: "0.5rem 1rem" }}>
          {loading ? "…working" : "Search"}
        </button>
        <label style={{ marginLeft: "1rem" }}>
          <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
          Show Explainability
        </label>
      </div>

      {results.length > 0 && (
        <ol>
          {results.map((r, i) => (
            <li key={i} style={{ marginBottom: "1rem" }}>
              <div><strong>Snippet:</strong> {r.snippet || r.text?.slice(0, 150) + "…"}</div>
              {explain && (
                <div style={{ fontSize: "0.9em", marginTop: "0.5rem" }}>
                  <strong>Score breakdown:</strong>{" "}
                  Hybrid {(r["feature_contributions_%"] && r["feature_contributions_%"].hybrid) ?? (r.hybrid_score != null ? (r.hybrid_score * 100).toFixed(1) : "-")} |
                  {r.temporal_non_discriminating ? (
                    <span style={{ fontStyle: "italic", color: "#666" }}> Temporal relevance: not distinguishing for this query</span>
                  ) : (
                    <span> Temporal {(r["feature_contributions_%"] && r["feature_contributions_%"].temporal) ?? (r.temporal_score != null ? r.temporal_score.toFixed(4) : "-")}</span>
                  )}
                  {r.temporal_explanation && (
                    <span style={{ marginLeft: "0.5rem", color: "#555" }}>({r.temporal_explanation})</span>
                  )} |
                  {r.metadata_non_discriminating ? (
                    <span style={{ fontStyle: "italic", color: "#666" }}> Metadata: not distinguishing (neutral)</span>
                  ) : (
                    <span> Metadata {(r["feature_contributions_%"] && r["feature_contributions_%"].metadata) ?? (r.metadata_score != null ? r.metadata_score.toFixed(4) : "-")}</span>
                  )}
                </div>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
