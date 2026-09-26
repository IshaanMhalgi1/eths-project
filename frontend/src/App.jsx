import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route, Link, useParams, useNavigate, useSearchParams } from "react-router-dom";
import { fetchPageImage } from "./locImage";
import { useSpeechRecognition } from "./useSpeechRecognition";
import { useAutocomplete } from "./useAutocomplete";

// Requests go through the Vite dev proxy (see vite.config.js) so they are
// same-origin. Do not hardcode a backend host here.
const API_BASE = "/api";

function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [results, setResults] = useState([]);
  const [explain, setExplain] = useState(searchParams.get("explain") === "true");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  // Speech-to-text is strictly additive: when the browser lacks the Web Speech
  // API, `supported` is false and no microphone control is rendered at all.
  const speech = useSpeechRecognition({
    onFinal: (text) => {
      if (text) setQuery(text);
    },
  });

  // The query string, as a primitive. Depending on the `searchParams` object
  // itself is unsafe: useSearchParams returns a fresh object on every render,
  // so the effect would re-run forever, and because doSearch calls
  // setSearchParams it would issue an endless stream of duplicate searches.
  const searchStr = searchParams.toString();

  // Tracks the query+mode already requested, so a URL update caused by our own
  // search does not trigger a second identical request.
  const lastSearched = useRef(null);

  // Guarded against being used directly as an event handler. `onClick={doSearch}`
  // would pass the click event as searchQuery, and axios would then try to
  // JSON.stringify that SyntheticEvent -- which fails on its circular
  // references to the originating DOM node. Only accept a real query string.
  const doSearch = async (searchQuery, searchExplain) => {
    const q = typeof searchQuery === "string" ? searchQuery : query;
    const ex = typeof searchExplain === "boolean" ? searchExplain : explain;
    if (!q) return;
    lastSearched.current = `${q}|${ex}`;
    setLoading(true);
    setError(null);
    try {
      const endpoint = ex ? "/explain" : "/search";
      const { data } = await axios.post(
        `${API_BASE}${endpoint}`,
        { query: q, size: 10 }
      );
      setResults(ex ? data.explanations : data.results);
    } catch (e) {
      // Report what actually went wrong rather than always blaming the server:
      // a rejected or cancelled request looks identical to a dead backend from
      // the outside, but the remedy differs.
      const status = e?.response?.status;
      const detail = e?.response?.data?.detail;
      const message = status
        ? `Search failed (HTTP ${status})${detail ? `: ${detail}` : ""}`
        : e?.code === "ERR_CANCELED"
          ? "Search cancelled."
          : e?.request && !e?.response
            ? "Could not reach the API – is the FastAPI server running?"
            : e?.message || "Search failed.";
      setError(message);
      console.error(e);
    }
    setLoading(false);
    // Update URL with query (without triggering navigation)
    setSearchParams({ q, explain: ex ? "true" : "false" });
  };

  // Auto-search when the URL query changes (handles back/forward navigation).
  useEffect(() => {
    const params = new URLSearchParams(searchStr);
    const urlQuery = params.get("q");
    const urlExplain = params.get("explain") === "true";
    if (urlQuery && urlQuery !== query) {
      setQuery(urlQuery);
    }
    if (urlExplain !== explain) {
      setExplain(urlExplain);
    }
    if (urlQuery && lastSearched.current !== `${urlQuery}|${urlExplain}`) {
      doSearch(urlQuery, urlExplain);
    }
    // Primitive dependency: stable unless the URL genuinely changes.
  }, [searchStr]);

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

  // Autocomplete is corpus-derived, so it can only ever suggest a query the
  // index can actually answer.
  const ac = useAutocomplete({ value: query });

  // The dropdown owns ArrowUp/ArrowDown, Escape and Enter-while-open; anything
  // it does not claim falls through to "run the search".
  const handleKeyDown = (e) => {
    const adopted = ac.onKeyDown(e);
    if (adopted) {
      setQuery(adopted);
      inputRef.current?.focus();
      return;
    }
    if (e.defaultPrevented) return;
    if (e.key === "Enter") {
      doSearch();
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>ETHS – Explainable Temporal Search</h1>
      <div style={{ marginBottom: "1rem" }}>
        <div ref={ac.rootRef} style={{ position: "relative", display: "inline-block" }}>
          <input
            ref={inputRef}
            type="text"
            placeholder="Enter a historical query…"
            value={query}
            onChange={handleQueryChange}
            onKeyDown={handleKeyDown}
            role="combobox"
            aria-expanded={ac.open}
            aria-autocomplete="list"
            aria-controls="autocomplete-listbox"
            aria-activedescendant={
              ac.open && ac.activeIndex >= 0 ? `ac-opt-${ac.activeIndex}` : undefined
            }
            autoComplete="off"
            style={{ width: "70%", padding: "0.5rem" }}
          />
          {ac.open && ac.suggestions.length > 0 && (
            <ul
              id="autocomplete-listbox"
              role="listbox"
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                right: 0,
                margin: "2px 0 0",
                padding: 0,
                listStyle: "none",
                backgroundColor: "#fff",
                border: "1px solid #d1d5db",
                borderRadius: "4px",
                boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
                zIndex: 20,
                maxHeight: "240px",
                overflowY: "auto",
              }}
            >
              {ac.suggestions.map((s, i) => (
                <li
                  key={s}
                  id={`ac-opt-${i}`}
                  role="option"
                  aria-selected={i === ac.activeIndex}
                  // mousedown + preventDefault keeps focus in the input, so the
                  // click registers before the input can lose focus.
                  onMouseDown={(e) => {
                    e.preventDefault();
                    setQuery(ac.adopt(s));
                    inputRef.current?.focus();
                  }}
                  onMouseEnter={() => ac.setActiveIndex(i)}
                  style={{
                    padding: "0.4rem 0.6rem",
                    cursor: "pointer",
                    fontSize: "0.95em",
                    backgroundColor: i === ac.activeIndex ? "#e8f0fe" : "transparent",
                  }}
                >
                  {s}
                </li>
              ))}
            </ul>
          )}
        </div>
        <button onClick={() => doSearch()} disabled={loading} style={{ marginLeft: "0.5rem", padding: "0.5rem 1rem" }}>
          {loading ? "…working" : "Search"}
        </button>
        {speech.supported && (
          <button
            type="button"
            onClick={() => (speech.listening ? speech.stop() : speech.start())}
            aria-label={speech.listening ? "Stop dictation" : "Search by voice"}
            title={speech.listening ? "Stop dictation" : "Search by voice"}
            aria-pressed={speech.listening}
            style={{
              marginLeft: "0.5rem",
              padding: "0.45rem 0.6rem",
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: speech.listening ? "#dc2626" : "#f3f4f6",
              border: "1px solid #d1d5db",
              borderRadius: "4px",
              lineHeight: 0,
            }}
          >
            {speech.listening ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="#ffffff" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="1.5" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="#111827" aria-hidden="true">
                <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Z" />
                <path d="M18 11a1 1 0 1 0-2 0 4 4 0 0 1-8 0 1 1 0 1 0-2 0 6 6 0 0 0 5 5.9V19H9a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2h-2v-2.1A6 6 0 0 0 18 11Z" />
              </svg>
            )}
          </button>
        )}
        <label style={{ marginLeft: "1rem" }}>
          <input type="checkbox" checked={explain} onChange={handleExplainChange} />
          Show Explainability
        </label>
      </div>

      {error && (
        <div
          role="alert"
          style={{
            marginTop: "-0.5rem",
            marginBottom: "0.75rem",
            fontSize: "0.85em",
            color: "#b91c1c",
          }}
        >
          {error}
        </div>
      )}

      {/* Non-blocking: the search path is never gated on dictation succeeding. */}
      {speech.supported && (speech.status || speech.interim || speech.needsModel) && (
        <div
          role="status"
          aria-live="polite"
          style={{
            fontSize: "0.85em",
            marginTop: "-0.5rem",
            marginBottom: "0.75rem",
            color: speech.status?.kind === "error" ? "#b91c1c" : "#666",
          }}
        >
          {speech.status?.text}
          {speech.interim && (
            <span style={{ fontStyle: "italic" }}> {speech.interim}</span>
          )}
          {speech.needsModel && (
            <button
              type="button"
              onClick={() => speech.installModel()}
              disabled={speech.modelBusy}
              style={{
                marginLeft: "0.5rem",
                fontSize: "0.95em",
                textDecoration: "underline",
                background: "none",
                border: "none",
                color: "#0066cc",
                cursor: speech.modelBusy ? "default" : "pointer",
              }}
            >
              {speech.modelBusy ? "downloading…" : "download speech model"}
            </button>
          )}
        </div>
      )}

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

function SourceImagePanel({ sourceImage }) {
  const [imgState, setImgState] = useState("idle"); // idle | loading | ready | failed
  const [imgUrl, setImgUrl] = useState(null);
  const [imgReason, setImgReason] = useState(null);

  const page =
    sourceImage && sourceImage.available ? sourceImage.primary : null;
  const key = page ? `${page.lccn}|${page.date}|${page.edition}|${page.sequence}` : "";

  // Resolve the displayable JPEG via the backend, which can reach the LOC
  // manifest even when the browser cannot.
  useEffect(() => {
    if (!page) {
      setImgState("idle");
      setImgUrl(null);
      setImgReason(null);
      return undefined;
    }
    const ac = new AbortController();
    setImgState("loading");
    setImgUrl(null);
    setImgReason(null);
    fetchPageImage({ ...page, signal: ac.signal })
      .then((r) => {
        if (ac.signal.aborted) return;
        if (r.available) {
          setImgUrl(r.imageUrl);
          setImgState("ready");
        } else {
          setImgReason(r.reason);
          setImgState("failed");
        }
      })
      .catch(() => {
        if (!ac.signal.aborted) setImgState("failed");
      });
    return () => ac.abort();
  }, [key]);

  if (!sourceImage) return null;

  const linkStyle = {
    color: "#0066cc",
    fontSize: "0.9em",
    marginRight: "1rem",
    whiteSpace: "nowrap",
  };
  const wrap = {
    marginTop: "0.75rem",
    paddingTop: "0.75rem",
    borderTop: "1px solid #eee",
  };

  // Honest unavailable state: no fabricated placeholder, no broken image.
  if (!sourceImage.available) {
    return (
      <div style={wrap}>
        <div style={{ fontSize: "0.9em", color: "#888", fontStyle: "italic" }}>
          Page image unavailable for this document.
        </div>
      </div>
    );
  }

  const extraPages = sourceImage.pages.length - 1;

  return (
    <div style={wrap}>
      <div style={{ fontSize: "0.9em", color: "#444" }}>
        <strong>Page image:</strong>{" "}
        <span style={{ color: "#666" }}>
          Library of Congress, {page.label}
          {sourceImage.newspaper ? ` (${sourceImage.newspaper})` : ""}
        </span>
      </div>

      {imgState === "ready" && imgUrl && (
        <div style={{ marginTop: "0.6rem" }}>
          <img
            src={imgUrl}
            alt={`Scanned newspaper page: ${page.label}`}
            onError={() => setImgState("failed")}
            style={{
              display: "block",
              maxWidth: "100%",
              maxHeight: "60vh",
              border: "1px solid #ddd",
              borderRadius: "4px",
              backgroundColor: "#fff",
            }}
          />
        </div>
      )}

      {imgState === "loading" && (
        <div style={{ marginTop: "0.6rem", fontSize: "0.9em", color: "#888" }}>
          Loading page image&hellip;
        </div>
      )}

      {imgState === "failed" && (
        <div style={{ marginTop: "0.6rem", fontSize: "0.9em", color: "#888" }}>
          <em>
            The page scan could not be loaded from loc.gov. The links below open
            it on the Library of Congress site.
          </em>
          {imgReason && (
            <div style={{ fontSize: "0.8em", color: "#aaa", marginTop: "0.2rem" }}>
              reason: {imgReason}
            </div>
          )}
        </div>
      )}

      <div style={{ marginTop: "0.6rem" }}>
        <a href={page.resource_page} target="_blank" rel="noopener noreferrer" style={linkStyle}>
          Open on loc.gov &nearr;
        </a>
        <a href={page.legacy_jp2} target="_blank" rel="noopener noreferrer" style={linkStyle}>
          Download JP2
        </a>
        <a href={page.issue_gallery} target="_blank" rel="noopener noreferrer" style={linkStyle}>
          View full issue
        </a>
      </div>

      {extraPages > 0 && (
        <div style={{ marginTop: "0.4rem", fontSize: "0.85em", color: "#666" }}>
          This document spans {sourceImage.pages.length} scanned pages:{" "}
          {sourceImage.pages.map((pg, i) => (
            <span key={`${pg.date}-${pg.edition}-${pg.sequence}`}>
              {i > 0 && ", "}
              <a href={pg.resource_page} target="_blank" rel="noopener noreferrer" style={{ color: "#0066cc" }}>
                p.{pg.sequence}
              </a>
            </span>
          ))}
        </div>
      )}
      <div style={{ marginTop: "0.4rem", fontSize: "0.8em", color: "#999" }}>
        Scans are served by the Library of Congress.
      </div>
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

        <SourceImagePanel sourceImage={document.source_image} />
        
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