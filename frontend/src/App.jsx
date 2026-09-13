import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route, Link, useParams, useNavigate, useSearchParams } from "react-router-dom";

const API_BASE = "http://127.0.0.1:8000";

function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [results, setResults] = useState([]);
  const [explain, setExplain] = useState(searchParams.get("explain") === "true");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Auto-search when URL query changes (handles back/forward navigation)
  useEffect(() => {
    const urlQuery = searchParams.get("q");
    const urlExplain = searchParams.get("explain") === "true";
    if (urlQuery && urlQuery !== query) {
      setQuery(urlQuery);
    }
    if (urlExplain !== explain) {
      setExplain(urlExplain);
    }
    if (urlQuery) {
      doSearch(urlQuery, urlExplain);
    }
  }, [searchParams]);

  const doSearch = async (searchQuery = query, searchExplain = explain) => {
    if (!searchQuery) return;
    setLoading(true);
    try {
      const endpoint = searchExplain ? "/explain" : "/search";
      const { data } = await axios.post(
        `${API_BASE}${endpoint}`,
        { query: searchQuery, size: 10 }
      );
      setResults(searchExplain ? data.explanations : data.results);
    } catch (e) {
      console.error(e);
      alert("Could not reach the API – is the FastAPI server running?");
    }
    setLoading(false);
    // Update URL with query (without triggering navigation)
    setSearchParams({ q: searchQuery, explain: searchExplain ? "true" : "false" });
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleExplainChange = (e) => {
    setExplain(e.target.checked);
  };

  const handleClick = (parentDocId, chunkId) => {
    // Update URL with current query before navigating
    setSearchParams({ q: query, explain: explain ? "true" : "false" });
    navigate(`/document/${parentDocId}?chunk=${chunkId}`);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      doSearch();
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>ETHS – Explainable Temporal Search</h1>
      <div style={{ marginBottom: "1rem" }}>
        <input
          type="text"
          placeholder="Enter a historical query…"
          value={query}
          onChange={handleQueryChange}
          onKeyDown={handleKeyDown}
          style={{ width: "70%", padding: "0.5rem" }}
        />
        <button onClick={doSearch} disabled={loading} style={{ marginLeft: "0.5rem", padding: "0.5rem 1rem" }}>
          {loading ? "…working" : "Search"}
        </button>
        <label style={{ marginLeft: "1rem" }}>
          <input type="checkbox" checked={explain} onChange={handleExplainChange} />
          Show Explainability
        </label>
      </div>

      {results.length > 0 && (
        <ol>
          {results.map((r, i) => (
            <li key={i} style={{ marginBottom: "1rem" }}>
              <div style={{ cursor: "pointer" }} onClick={() => handleClick(r.parent_doc_id, r.chunk_id)}>
                <strong>Snippet:</strong> {r.snippet || r.text?.slice(0, 150) + "…"}
              </div>
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

function DocumentView() {
  const { parent_doc_id } = useParams();
  const [document, setDocument] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [highlightChunkId, setHighlightChunkId] = useState(null);
  const highlightRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const chunk = urlParams.get("chunk");
    if (chunk) {
      setHighlightChunkId(chunk);
    }
  }, []);

  useEffect(() => {
    const fetchDocument = async () => {
      try {
        const { data } = await axios.get(`${API_BASE}/document/${parent_doc_id}`);
        setDocument(data);
      } catch (e) {
        setError(e.response?.data?.detail || "Failed to load document");
      } finally {
        setLoading(false);
      }
    };
    fetchDocument();
  }, [parent_doc_id]);

  // Scroll to highlighted chunk after render
  useEffect(() => {
    if (highlightChunkId && highlightRef.current) {
      const element = highlightRef.current.querySelector(`[data-chunk-id="${highlightChunkId}"]`);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "center" });
        element.classList.add("highlighted-chunk");
        setTimeout(() => element.classList.remove("highlighted-chunk"), 3000);
      }
    }
  }, [document, highlightChunkId]);

  const handleBack = () => {
    navigate(-1); // Go back in browser history, preserving search state
  };

  if (loading) {
    return (
      <div style={{ maxWidth: 800, margin: "2rem auto", fontFamily: "sans-serif", textAlign: "center" }}>
        <h1>Loading document…</h1>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ maxWidth: 800, margin: "2rem auto", fontFamily: "sans-serif", textAlign: "center" }}>
        <h1>Document Not Found</h1>
        <p style={{ color: "#666" }}>{error}</p>
        <p><button onClick={handleBack} style={{ color: "#0066cc", background: "none", border: "none", cursor: "pointer", fontSize: "1rem" }}>← Back to search</button></p>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <div style={{ marginBottom: "1rem" }}>
        <button onClick={handleBack} style={{ color: "#0066cc", background: "none", border: "none", cursor: "pointer", fontSize: "1rem", textDecoration: "underline" }}>← Back to search</button>
      </div>
      
      <div style={{ 
        border: "1px solid #ddd", 
        borderRadius: "8px", 
        padding: "1.5rem",
        backgroundColor: "#fafafa"
      }}>
        <h2 style={{ marginTop: 0, marginBottom: "0.5rem" }}>
          Document: {document.parent_doc_id}
        </h2>
        <div style={{ color: "#666", fontSize: "0.9em", marginBottom: "1rem" }}>
          {document.metadata.publication_year && <span>Year: {document.metadata.publication_year}</span>}
          {document.metadata.historical_start && document.metadata.historical_end && (
            <span style={{ marginLeft: "1rem" }}>
              Historical: {document.metadata.historical_start}–{document.metadata.historical_end}
            </span>
          )}
          {document.metadata.num_chunks && (
            <span style={{ marginLeft: "1rem" }}>Chunks: {document.metadata.num_chunks}</span>
          )}
          {document.metadata.provenance && (
            <div style={{ marginTop: "0.5rem" }}>
              <a href={document.metadata.provenance} target="_blank" rel="noopener noreferrer" style={{ color: "#0066cc" }}>
                Source: {document.metadata.provenance}
              </a>
            </div>
          )}
        </div>
        
        <div 
          ref={highlightRef}
          style={{ 
            whiteSpace: "pre-wrap", 
            lineHeight: "1.6",
            fontSize: "0.95rem",
            maxHeight: "70vh",
            overflowY: "auto",
            padding: "1rem",
            backgroundColor: "white",
            border: "1px solid #eee",
            borderRadius: "4px"
          }}
        >
          {document.chunks.map((chunk, idx) => (
            <div 
              key={chunk.chunk_id}
              data-chunk-id={chunk.chunk_id}
              style={{ 
                marginBottom: "1rem",
                padding: "0.5rem",
                borderLeft: highlightChunkId === chunk.chunk_id ? "4px solid #ffcc00" : "4px solid transparent",
                backgroundColor: highlightChunkId === chunk.chunk_id ? "#fffde7" : "transparent",
                transition: "all 0.3s ease"
              }}
            >
              <div style={{ fontSize: "0.75rem", color: "#999", marginBottom: "0.25rem" }}>
                Chunk: {chunk.chunk_id}
              </div>
              <div>{chunk.text}</div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .highlighted-chunk {
          animation: highlight-pulse 1s ease-in-out;
        }
        @keyframes highlight-pulse {
          0% { background-color: #fffde7; border-left-color: #ffcc00; }
          50% { background-color: #fff9c4; border-left-color: #f9a825; }
          100% { background-color: #fffde7; border-left-color: #ffcc00; }
        }
      `}</style>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/document/:parent_doc_id" element={<DocumentView />} />
      </Routes>
    </Router>
  );
}