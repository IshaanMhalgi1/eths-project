import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useParams,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import { fetchPageImage } from "./locImage";
import Timeline from "./Timeline";
import { useSpeechRecognition } from "./useSpeechRecognition";
import { useAutocomplete } from "./useAutocomplete";

// Requests go through the Vite dev proxy (see vite.config.js) so they are
// same-origin. Do not hardcode a backend host here.
const API_BASE = "/api";

const THEME_KEY = "eths-theme";

// Upper bound on the page-image lookup before falling back to the direct LOC
// links. Cold resolutions that fall through to the Internet Archive have been
// observed at ~20s, which is long enough to read as a broken panel.
const IMAGE_LOOKUP_TIMEOUT_MS = 12000;

/* ---------------------------------------------------------------------------
 * Theme

 * The initial value is set by an inline script in index.html before first
 * paint, so there is no white flash on a dark reload. This hook only needs to
 * keep React's view of the theme in sync afterwards.
 * ------------------------------------------------------------------------- */

function currentTheme() {
  if (typeof document === "undefined") return "light";
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? "dark"
    : "light";
}

function useTheme() {
  const [theme, setTheme] = useState(currentTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // A blocked localStorage (private mode) must not break theming; the
      // attribute is already applied, so the toggle still works for this visit.
    }
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  return { theme, toggle };
}

/* ---------------------------------------------------------------------------
 * Shared chrome
 * ------------------------------------------------------------------------- */

function Rail({ brandSub, children, controls }) {
  return (
    <aside className="rail">
      <div>
        <h1 className="brand">ETHS</h1>
        <p className="brand-sub">{brandSub}</p>
      </div>
      {children}
      <div className="controls">{controls}</div>
    </aside>
  );
}

function MicButton({ speech }) {
  if (!speech.supported) return null;
  return (
    <button
      type="button"
      className="btn btn-icon"
      data-listening={speech.listening ? "true" : "false"}
      onClick={() => (speech.listening ? speech.stop() : speech.start())}
      aria-label={speech.listening ? "Stop dictation" : "Search by voice"}
      title={speech.listening ? "Stop dictation" : "Search by voice"}
      aria-pressed={speech.listening}
    >
      {speech.listening ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <rect x="6" y="6" width="12" height="12" rx="1.5" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Z" />
          <path d="M18 11a1 1 0 1 0-2 0 4 4 0 0 1-8 0 1 1 0 1 0-2 0 6 6 0 0 0 5 5.9V19H9a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2h-2v-2.1A6 6 0 0 0 18 11Z" />
        </svg>
      )}
    </button>
  );
}

/* ---------------------------------------------------------------------------
 * Search page
 * ------------------------------------------------------------------------- */

function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [results, setResults] = useState([]);
  const [overview, setOverview] = useState(null);
  const [overviewState, setOverviewState] = useState("idle");
  const [explain, setExplain] = useState(searchParams.get("explain") === "true");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Bumped on every completed search and used as a React key, so the reveal
  // sequence replays once per search instead of animating only on mount.
  const [revealId, setRevealId] = useState(0);
  // Year range chosen on the timeline, applied server-side as a retrieval
  // constraint. It lives in the URL so a shared link reproduces the same
  // result set, and so back/forward moves through range selections.
  const [yearRange, setYearRange] = useState(() => {
    const p = searchParams.get("y0");
    const q = searchParams.get("y1");
    return p && q ? { start: Number(p), end: Number(q) } : null;
  });
  // Precomputed decade density for the timeline's context track.
  const [density, setDensity] = useState(null);
  const densityLoaded = useRef(false);
  const resultRefs = useRef([]);
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const { theme, toggle } = useTheme();

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

  // Identifies the newest overview request so a slow reply for an old query
  // cannot overwrite a newer one.
  const overviewReq = useRef(0);

  const fetchOverview = (q, yr) => {
    const reqId = ++overviewReq.current;
    setOverviewState("loading");
    setOverview(null);
    axios
      .post(`${API_BASE}/overview`, {
        query: q,
        size: 10,
        ...(yr ? { year_start: yr.start, year_end: yr.end } : {}),
      })
      .then(({ data }) => {
        if (reqId !== overviewReq.current) return;
        setOverview(data);
        setOverviewState("ready");
      })
      .catch((e) => {
        if (reqId !== overviewReq.current) return;
        setOverview({
          status: "unavailable",
          reason: e?.response?.status
            ? `http-${e.response.status}`
            : "network-error",
        });
        setOverviewState("ready");
      });
  };

  const clearOverview = () => {
    overviewReq.current += 1;
    setOverview(null);
    setOverviewState("idle");
  };

  // Guarded against being used directly as an event handler. `onClick={doSearch}`
  // would pass the click event as searchQuery, and axios would then try to
  // JSON.stringify that SyntheticEvent -- which fails on its circular
  // references to the originating DOM node. Only accept a real query string.
  const doSearch = async (searchQuery, searchExplain, range) => {
    const q = typeof searchQuery === "string" ? searchQuery : query;
    const ex = typeof searchExplain === "boolean" ? searchExplain : explain;
    // `range` is passed explicitly by timeline interactions; otherwise fall
    // back to whatever range is currently applied.
    const yr = range === undefined ? yearRange : range;
    if (!q) return;
    lastSearched.current = `${q}|${ex}|${yr?.start ?? ""}-${yr?.end ?? ""}`;
    setLoading(true);
    setError(null);
    try {
      const endpoint = ex ? "/explain" : "/search";
      const { data } = await axios.post(`${API_BASE}${endpoint}`, {
        query: q,
        size: 10,
        // The range is sent to the server, never applied here: filtering the
        // results already in hand would misreport what the archive holds
        // outside the current top-K.
        ...(yr ? { year_start: yr.start, year_end: yr.end } : {}),
      });
      setResults(ex ? data.explanations : data.results);
      setRevealId((n) => n + 1);
      // The overview is a separate, non-blocking request: a failed or refused
      // summary must never turn a successful search into an error state.
      if (!ex) fetchOverview(q, yr);
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
            ? "Could not reach the API - is the FastAPI server running?"
            : e?.message || "Search failed.";
      setError(message);
      console.error(e);
    }
    setLoading(false);
    // Update URL with query (without triggering navigation)
    setSearchParams({
      q,
      explain: ex ? "true" : "false",
      ...(yr ? { y0: String(yr.start), y1: String(yr.end) } : {}),
    });
  };

  // Auto-search when the URL query changes (handles back/forward navigation).
  useEffect(() => {
    const params = new URLSearchParams(searchStr);
    const urlQuery = params.get("q");
    const urlExplain = params.get("explain") === "true";
    const y0 = params.get("y0");
    const y1 = params.get("y1");
    const urlRange = y0 && y1 ? { start: Number(y0), end: Number(y1) } : null;
    if (urlQuery && urlQuery !== query) {
      setQuery(urlQuery);
    }
    if (urlExplain !== explain) {
      setExplain(urlExplain);
    }
    if (
      urlQuery &&
      lastSearched.current !== `${urlQuery}|${urlExplain}|${y0 ?? ""}-${y1 ?? ""}`
    ) {
      setYearRange(urlRange);
      doSearch(urlQuery, urlExplain, urlRange);
    }
    // Primitive dependency: stable unless the URL genuinely changes.
  }, [searchStr]);

  // Decade density is precomputed and identical for every search, so it is
  // fetched once per session rather than per query.
  useEffect(() => {
    if (densityLoaded.current) return;
    densityLoaded.current = true;
    axios
      .get(`${API_BASE}/corpus/decade-density`)
      .then(({ data }) => setDensity(data))
      .catch(() => setDensity(null));
  }, []);

  // A range chosen on the timeline re-runs the search through the same server
  // path as a typed query, rather than narrowing the results already held.
  const handleSelectRange = (range) => {
    setYearRange(range);
    doSearch(undefined, undefined, range);
  };

  const handleClearRange = () => {
    setYearRange(null);
    doSearch(undefined, undefined, null);
  };

  // Clicking a marker scrolls its passage into view and flashes it, so the
  // chart and the list stay in step.
  const handleFocusResult = (index) => {
    const el = resultRefs.current[index];
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.remove("is-flash");
    // Force a reflow so re-adding the class restarts the flash animation.
    void el.offsetWidth;
    el.classList.add("is-flash");
    window.setTimeout(() => el.classList.remove("is-flash"), 1800);
  };

  const handleExplainChange = (e) => {
    setExplain(e.target.checked);
    // Explainability mode returns score breakdowns rather than passages, so
    // there is no text for the overview to summarise.
    if (e.target.checked) clearOverview();
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
    <div className="shell">
      <Rail
        brandSub="Explainable temporal search over 50,000 passages of 19th-century American newspapers."
        controls={
          <>
            <div className="control">
              <label htmlFor="explain-toggle">Show explainability</label>
              <input
                id="explain-toggle"
                type="checkbox"
                checked={explain}
                onChange={handleExplainChange}
              />
            </div>
            <button
              type="button"
              className="control-theme"
              onClick={toggle}
              aria-pressed={theme === "dark"}
            >
              {theme === "dark" ? "Light theme" : "Dark theme"}
            </button>
          </>
        }
      >
        <div className="composer" ref={ac.rootRef}>
          <div className="composer-row">
            <input
              ref={inputRef}
              type="text"
              placeholder="Search the archive"
              aria-label="Search query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              role="combobox"
              aria-expanded={ac.open}
              aria-autocomplete="list"
              aria-controls="autocomplete-listbox"
              aria-activedescendant={
                ac.open && ac.activeIndex >= 0 ? `ac-opt-${ac.activeIndex}` : undefined
              }
              autoComplete="off"
            />
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => doSearch()}
              disabled={loading}
            >
              {loading ? "Searching" : "Search"}
            </button>
            <MicButton speech={speech} />
          </div>

          {ac.open && ac.suggestions.length > 0 && (
            <ul id="autocomplete-listbox" role="listbox" className="suggest">
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
                >
                  {s}
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Non-blocking: the search path is never gated on dictation succeeding. */}
        {speech.supported &&
          (speech.status || speech.interim || speech.needsModel) && (
            <div
              className={`notice${
                speech.status?.kind === "error" ? " notice-error" : ""
              }`}
              role="status"
              aria-live="polite"
            >
              {speech.status?.text}
              {speech.interim && <em> {speech.interim}</em>}
              {speech.needsModel && (
                <button
                  type="button"
                  className="notice-btn"
                  onClick={() => speech.installModel()}
                  disabled={speech.modelBusy}
                >
                  {speech.modelBusy ? "downloading…" : "download speech model"}
                </button>
              )}
            </div>
          )}
      </Rail>

      <main className="reading">
        {error && (
          <div className="alert" role="alert">
            {error}
          </div>
        )}

        {/* Shown whenever a search has run, including one that returned
            nothing: a year range with no documents is exactly when the density
            band is most informative, so hiding it would hide the context that
            explains the empty result. */}
        {!explain && revealId > 0 && (
          <Timeline
            density={density}
            results={results}
            activeRange={yearRange}
            onSelectRange={handleSelectRange}
            onClearRange={handleClearRange}
            onFocusResult={handleFocusResult}
          />
        )}

        <OverviewPanel
          key={`summary-${revealId}`}
          revealing={revealId > 0}
          state={overviewState}
          overview={overview}
          results={results}
          onOpenDocument={handleClick}
        />

        {results.length > 0 ? (
          <ol className="result-list" key={`results-${revealId}`}>
            {results.map((r, i) => (
              <li
                key={i}
                ref={(el) => {
                  resultRefs.current[i] = el;
                }}
                className={`result${revealId > 0 ? " is-revealing" : ""}`}
                style={{ "--stagger-i": Math.min(i, 8) }}
              >
                <button
                  type="button"
                  className="result-btn"
                  onClick={() => handleClick(r.parent_doc_id, r.chunk_id)}
                >
                  <div className="result-rank">Passage {i + 1}</div>
                  <p className="result-text">
                    {r.snippet || r.text?.slice(0, 150) + "…"}
                  </p>
                </button>
                {explain && <ScoreBreakdown r={r} />}
              </li>
            ))}
          </ol>
        ) : (
          !loading &&
          !error && (
            <p className="empty">
              Search a topic, a year, or a phrase. Suggested terms come from the
              archive's own vocabulary, so anything offered here can be answered.
            </p>
          )
        )}
      </main>
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * Explainability score breakdown
 * ------------------------------------------------------------------------- */

function ScoreBreakdown({ r }) {
  const hybrid =
    (r["feature_contributions_%"] && r["feature_contributions_%"].hybrid) ??
    (r.hybrid_score != null ? (r.hybrid_score * 100).toFixed(1) : null);
  return (
    <div className="score">
      <b>Why this passage:</b> hybrid {hybrid ?? "–"}
      {r.temporal_non_discriminating ? (
        <span className="score-idle"> temporal not distinguishing here</span>
      ) : (
        <span>
          {" "}
          temporal{" "}
          {(r["feature_contributions_%"] &&
            r["feature_contributions_%"].temporal) ??
            (r.temporal_score != null ? r.temporal_score.toFixed(4) : "–")}
        </span>
      )}
      {r.temporal_explanation && <span> ({r.temporal_explanation})</span>}
      {r.metadata_non_discriminating ? (
        <span className="score-idle"> metadata neutral</span>
      ) : (
        <span>
          {" "}
          metadata{" "}
          {(r["feature_contributions_%"] &&
            r["feature_contributions_%"].metadata) ??
            (r.metadata_score != null ? r.metadata_score.toFixed(4) : "–")}
        </span>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * AI overview
 * ------------------------------------------------------------------------- */

const OVERVIEW_REASONS = {
  "not-configured":
    "No AI overview model is configured on this server, so no summary is shown. Search results below are unaffected.",
  timeout: "The overview model did not respond in time, so no summary is shown.",
  "rate-limited":
    "The overview model is rate limiting requests right now, so no summary is shown.",
  unauthorized:
    "The overview model rejected the configured API key, so no summary is shown.",
  "network-error":
    "The overview model could not be reached, so no summary is shown.",
  "malformed-response":
    "The overview model returned an unreadable response, so no summary is shown.",
  "output-too-long":
    "The overview model returned more text than the limit allows, so no summary is shown.",
  "reasoning-budget-exhausted":
    "The overview model used its whole response budget reasoning and returned no answer, so no summary is shown. Raising LLM_MAX_TOKENS usually fixes this.",
  "empty-response":
    "The overview model returned an empty response, so no summary is shown.",
  "no-results":
    "This query returned no readable passages to summarise.",
  "model-reported":
    "The passages retrieved for this query do not contain enough evidence to support a summary.",
  "no-attributable-claims":
    "A summary was produced, but no sentence in it could be tied to a source, so it was discarded.",
  uncited: "every sentence lacked a source citation",
  unsupported: "the summary repeated claims the source passages do not support",
  verbatim: "the summary copied the source text instead of paraphrasing it",
};

// The backend also emits comma-joined reasons (e.g. "uncited,unsupported") and
// dynamic "http-<status>" codes, neither of which is a key in the map above.
// Resolve those explicitly so users never see a raw internal string.
function reasonText(reason) {
  if (!reason) return null;
  if (OVERVIEW_REASONS[reason]) return OVERVIEW_REASONS[reason];
  const parts = String(reason)
    .split(",")
    .map((r) => OVERVIEW_REASONS[r.trim()])
    .filter(Boolean);
  if (parts.length) {
    return `The summary was discarded because ${parts.join(", and ")}.`;
  }
  if (String(reason).startsWith("http-")) {
    return `The overview service returned an error (${reason}), so no summary is shown.`;
  }
  return `The overview could not be produced (${reason}).`;
}

function OverviewPanel({
  state,
  overview,
  results,
  onOpenDocument,
  revealing,
}) {
  if (state === "loading") {
    return (
      <div className="summary" role="status" aria-live="polite">
        <h2 className="summary-title">AI summary</h2>
        <p className="summary-note">Reading the retrieved passages</p>
      </div>
    );
  }
  if (state !== "ready" || !overview) return null;

  // Match citations by document, not by list position. The overview endpoint
  // re-runs retrieval server-side so the model can only see server-retrieved
  // text, which means its sources are not guaranteed to be the ones rendered
  // below. A citation that is not on screen stays inert and says why, rather
  // than silently pointing at an unrelated result.
  const byDoc = new Map();
  results.forEach((r) => {
    if (r && r.parent_doc_id) byDoc.set(r.parent_doc_id, r);
  });
  const sourceByIndex = new Map(
    (overview.sources || []).map((s) => [s.index, s])
  );

  if (overview.status !== "ok" || !overview.claims?.length) {
    return (
      <div className="summary">
        <h2 className="summary-title">AI summary</h2>
        <p className="summary-note">
          {reasonText(overview.reason) ||
            "The overview could not be produced."}
        </p>
      </div>
    );
  }

  const v = overview.verification || {};
  const n = overview.sources?.length ?? 0;

  return (
    <section
      className={`summary${revealing ? " is-revealing" : ""}`}
      aria-label="AI summary of retrieved passages"
    >
      <div className="summary-head">
        <h2 className="summary-title">AI summary</h2>
        <span className="summary-meta">
          {n} {n === 1 ? "passage" : "passages"}
        </span>
      </div>

      {/* Reading content: Newsreader, and the claim text stays the model's own
          wording. Citations are rendered as buttons, not inline text, so they
          can be operated by keyboard. */}
      <p className="summary-body">
        {overview.claims.map((c, i) => (
          <span key={i}>
            {c.text}{" "}
            {c.citations.map((n2) => {
              const src = sourceByIndex.get(n2);
              const match = src && byDoc.get(src.parent_doc_id);
              const label = src?.year ? ` (${src.year})` : "";
              if (match) {
                return (
                  <button
                    key={n2}
                    type="button"
                    className="cite"
                    onClick={() =>
                      onOpenDocument(match.parent_doc_id, match.chunk_id)
                    }
                    title={`Open the cited passage${label}`}
                  >
                    [{n2}]
                  </button>
                );
              }
              return (
                <span
                  key={n2}
                  className="cite"
                  data-unlinked="true"
                  title={`Source ${n2}${label} was retrieved for the summary but is not in the results below`}
                >
                  [{n2}]
                </span>
              );
            })}{" "}
          </span>
        ))}
      </p>

      <details className="summary-details">
        <summary>How this summary was checked</summary>
        <ul className="audit">
          <li>
            {v.claims_kept} of {v.sentences} generated{" "}
            {v.sentences === 1 ? "sentence" : "sentences"} kept
            {v.claims_rejected > 0 && `, ${v.claims_rejected} discarded`}.
          </li>
          {v.unsupported_claims?.map((u, i) => (
            <li key={i}>
              Discarded as unsupported by its cited sources (word overlap{" "}
              {u.support}): <span className="audit-quote">“{u.text}”</span>
            </li>
          ))}
          {v.verbatim_claims?.map((c, i) => (
            <li key={`v${i}`}>
              Discarded as copied rather than paraphrased ({c.run} consecutive
              words shared): <span className="audit-quote">“{c.text}”</span>
            </li>
          ))}
          {v.uncited_claims > 0 && (
            <li>
              {v.uncited_claims} sentence{v.uncited_claims === 1 ? "" : "s"} had
              no citation and {v.uncited_claims === 1 ? "was" : "were"} discarded.
            </li>
          )}
          {v.invalid_citations?.length > 0 && (
            <li>
              Citations to non-existent sources were ignored:{" "}
              {v.invalid_citations.join(", ")}.
            </li>
          )}
        </ul>
        <p className="summary-note">
          Generated from these {n} retrieved {n === 1 ? "passage" : "passages"}{" "}
          only. The checks above are automated word-level filters, so a sentence
          that reuses source vocabulary while stating the opposite can still get
          through. Check anything that matters against the passage itself.
          {overview.model && <> Model: {overview.model}.</>}
          {v.thresholds && (
            <>
              {" "}
              Thresholds used: word overlap {v.thresholds.support_min}, copied-run{" "}
              {v.thresholds.verbatim_max_run} words.
            </>
          )}
        </p>
      </details>
    </section>
  );
}

/* ---------------------------------------------------------------------------
 * Page image
 * ------------------------------------------------------------------------- */

function SourceImagePanel({ sourceImage }) {
  const [imgState, setImgState] = useState("idle"); // idle | loading | ready | failed
  const [imgUrl, setImgUrl] = useState(null);
  const [imgReason, setImgReason] = useState(null);

  const page = sourceImage && sourceImage.available ? sourceImage.primary : null;
  const key = page ? `${page.lccn}|${page.date}|${page.edition}|${page.sequence}` : "";

  // Resolve the displayable JPEG via the backend, which can reach the LOC
  // manifest even when the browser cannot. A cache miss can take ~20s while the
  // Internet Archive fallback is tried, so the lookup is bounded: without this
  // the panel can sit on "Loading" indefinitely, which is worse than showing the
  // honest fallback links that are already rendered either way.
  useEffect(() => {
    if (!page) {
      setImgState("idle");
      setImgUrl(null);
      setImgReason(null);
      return undefined;
    }
    const ac = new AbortController();
    let settled = false;
    setImgState("loading");
    setImgUrl(null);
    setImgReason(null);
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      ac.abort();
      setImgReason("image-lookup-timeout");
      setImgState("failed");
    }, IMAGE_LOOKUP_TIMEOUT_MS);
    fetchPageImage({ ...page, signal: ac.signal })
      .then((r) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (r.available) {
          setImgUrl(r.imageUrl);
          setImgState("ready");
        } else {
          setImgReason(r.reason);
          setImgState("failed");
        }
      })
      .catch(() => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        setImgState("failed");
      });
    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [key]);

  if (!sourceImage) return null;

  // Honest unavailable state: no fabricated placeholder, no broken image.
  if (!sourceImage.available) {
    return (
      <div className="scan-status">
        Page image unavailable for this document.
      </div>
    );
  }

  const extraPages = sourceImage.pages.length - 1;

  return (
    <section className="scan" aria-label="Scanned newspaper page">
      <div className="scan-frame">
        {imgState === "ready" && imgUrl && (
          <img
            src={imgUrl}
            alt={`Scanned newspaper page: ${page.label}`}
            onError={() => setImgState("failed")}
          />
        )}
        {imgState === "loading" && (
          <p className="scan-status">Loading page image</p>
        )}
        {imgState === "failed" && (
          <p className="scan-status">
            <em>
              The page scan could not be loaded from loc.gov. The links below open
              it on the Library of Congress site.
            </em>
            {imgReason && <div>reason: {imgReason}</div>}          </p>
        )}

        <div className="scan-meta">
          <span>
            Library of Congress, {page.label}
            {sourceImage.newspaper ? ` (${sourceImage.newspaper})` : ""}
          </span>
          <span>Scans are served by the Library of Congress.</span>
        </div>

        <div className="scan-links">
          <a href={page.resource_page} target="_blank" rel="noopener noreferrer">
            Open on loc.gov
          </a>
          <a href={page.legacy_jp2} target="_blank" rel="noopener noreferrer">
            Download JP2
          </a>
          <a href={page.issue_gallery} target="_blank" rel="noopener noreferrer">
            View full issue
          </a>
        </div>

        {extraPages > 0 && (
          <p className="scan-extra">
            This document spans {sourceImage.pages.length} scanned pages:{" "}
            {sourceImage.pages.map((pg, i) => (
              <span key={`${pg.date}-${pg.edition}-${pg.sequence}`}>
                {i > 0 && ", "}
                <a
                  href={pg.resource_page}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  page {pg.sequence}
                </a>
              </span>
            ))}
          </p>
        )}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------------------
 * Document view
 * ------------------------------------------------------------------------- */

function DocumentView() {
  const { parent_doc_id } = useParams();
  const [document, setDocument] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [highlightChunkId, setHighlightChunkId] = useState(null);
  const highlightRef = useRef(null);
  const navigate = useNavigate();
  const { theme, toggle } = useTheme();

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

  // Scroll to the matched chunk and let its one-time pulse play. The pulse is a
  // single non-repeating animation, so no class teardown is needed afterwards.
  useEffect(() => {
    if (highlightChunkId && highlightRef.current) {
      const element = highlightRef.current.querySelector(
        `[data-chunk-id="${highlightChunkId}"]`
      );
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [document, highlightChunkId]);

  const handleBack = () => {
    navigate(-1); // Go back in browser history, preserving search state
  };

  if (loading) {
    return (
      <div className="shell">
        <Rail
          brandSub="Explainable temporal search over 19th-century American newspapers."
          controls={
            <button type="button" className="control-theme" onClick={toggle}>
              {theme === "dark" ? "Light theme" : "Dark theme"}
            </button>
          }
        >
          <button type="button" className="back-link" onClick={handleBack}>
            Back to search
          </button>
        </Rail>
        <main className="reading">
          <p className="doc-loading">Loading document</p>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="shell">
        <Rail
          brandSub="Explainable temporal search over 19th-century American newspapers."
          controls={
            <button type="button" className="control-theme" onClick={toggle}>
              {theme === "dark" ? "Light theme" : "Dark theme"}
            </button>
          }
        >
          <button type="button" className="back-link" onClick={handleBack}>
            Back to search
          </button>
        </Rail>
        <main className="reading">
          <h1 className="doc-title">Document not found</h1>
          <p className="summary-note">{error}</p>
        </main>
      </div>
    );
  }

  const m = document.metadata || {};
  const facts = [
    m.publication_year && { label: "Year", value: m.publication_year },
    m.historical_start &&
      m.historical_end && {
        label: "Historical range",
        value: `${m.historical_start}–${m.historical_end}`,
      },
    m.num_chunks && { label: "Passages", value: m.num_chunks },
  ].filter(Boolean);

  return (
    <div className="shell">
      <Rail
        brandSub="Document view"
        controls={
          <>
            <button type="button" className="back-link" onClick={handleBack}>
              Back to search
            </button>
            <button type="button" className="control-theme" onClick={toggle}>
              {theme === "dark" ? "Light theme" : "Dark theme"}
            </button>
          </>
        }
      >
        {facts.length > 0 && (
          <dl className="rail-facts">
            {facts.map((f) => (
              <div className="rail-fact" key={f.label}>
                <dt>{f.label}</dt>
                <dd>{f.value}</dd>
              </div>
            ))}
          </dl>
        )}
        {m.provenance && (
          <p className="rail-note">
            <a
              href={m.provenance}
              target="_blank"
              rel="noopener noreferrer"
            >
              Source record
            </a>
          </p>
        )}
      </Rail>

      <main className="reading">
        <header className="doc-head">
          <h1 className="doc-title">{document.parent_doc_id}</h1>
        </header>

        <SourceImagePanel sourceImage={document.source_image} />

        <div ref={highlightRef} className="doc-text">
          {document.chunks.map((chunk) => (
            <div
              key={chunk.chunk_id}
              data-chunk-id={chunk.chunk_id}
              className={`chunk${
                highlightChunkId === chunk.chunk_id ? " is-match is-pulsing" : ""
              }`}
            >
              <div className="chunk-tag">Passage {chunk.chunk_id}</div>
              <div>{chunk.text}</div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * Routes
 * ------------------------------------------------------------------------- */

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
