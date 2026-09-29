import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

/* ---------------------------------------------------------------------------
 * Timeline: where this search's results fall across the corpus's date range.
 *
 * The whole point of this chart is to let a reader tell two different things
 * apart:
 *
 *   - how many RESULTS this search returned in each year
 *   - how many DOCUMENTS the corpus holds in each decade
 *
 * Those are drawn as two separate tracks on a shared time axis, each with its
 * own baseline and its own stated scale, rather than as one overlay. An overlay
 * invites the reading "this decade is important", which would be a claim about
 * significance that neither layer actually supports.
 *
 * Every hook is called before any early return. The component waits on two
 * things that arrive independently -- the density payload and the container
 * width -- so a guard placed above a hook would change the hook count between
 * renders and crash the tree.
 * ------------------------------------------------------------------------- */

const PAD_X = 10;
const TOP_PAD = 6;
const LABEL_H = 15;
const TRACK_H = 52;
const TRACK_GAP = 18;
const AXIS_H = 20;
const HEIGHT =
  TOP_PAD + (LABEL_H + TRACK_H) + TRACK_GAP + (LABEL_H + TRACK_H) + AXIS_H;

const DOT_R = 4.5;
const DOT_GAP = 12;
const MAX_STACKED_DOTS = 4;
const BAR_INSET = 0.3; // fraction of the decade slot left as gutter

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Render at real pixel size so axis labels stay legible instead of scaling. */
function useMeasure() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect?.width ?? 0;
      setWidth(Math.floor(w));
    });
    ro.observe(el);
    setWidth(Math.floor(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, []);

  return [ref, width];
}

function formatCount(n) {
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k`;
  return String(n);
}

export default function Timeline({
  density,
  results,
  activeRange,
  onSelectRange,
  onClearRange,
  onFocusResult,
  corpus = "expanded",
}) {
  const [wrapRef, width] = useMeasure();
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);
  const [drag, setDrag] = useState(null);
  const [entering, setEntering] = useState(false);
  // Timestamp of the last drag release. A drag ending over a band still lets
  // the browser fire that band's click, so the click is ignored -- but only
  // briefly. A sticky boolean broke the *next* real click whenever the drag's
  // own click did not land on a band, which silently disabled decade
  // selection until the page was reloaded.
  const lastDragEnd = useRef(0);
  // Live drag values for the window listeners, which close over the first
  // render and would otherwise read a stale `drag` state.
  const dragRef = useRef(null);

  const rows = results || [];
  const ready = !!density && !!density.decades?.length && width > 0;
  const start = ready ? density.range.start : 0;
  const end = ready ? density.range.end : 1;

  const commit = useCallback(
    (a, b) => {
      const lo = clamp(Math.round(Math.min(a, b)), start, end);
      const hi = clamp(Math.round(Math.max(a, b)), start, end);
      // A drag that never left its origin is a click, not a range.
      if (hi - lo < 1) return;
      onSelectRange({ start: lo, end: hi });
    },
    [end, onSelectRange, start]
  );

  // The entrance plays once per result set, not on every re-render.
  useEffect(() => {
    setEntering(false);
    if (!ready) return undefined;
    const id = requestAnimationFrame(() => setEntering(true));
    return () => cancelAnimationFrame(id);
  }, [ready, rows]);

  if (!ready) {
    // Deliberately the same element and ref as the populated branch below.
    // Swapping the ref onto a different node would leave the ResizeObserver
    // in useMeasure watching a detached element, and the chart would then
    // measure zero width forever.
    return (
      <section
        className="timeline"
        ref={wrapRef}
        aria-label="Result distribution versus corpus density by decade"
      >
        {!density && <p className="timeline-note">Timeline unavailable.</p>}
      </section>
    );
  }

  const { decades } = density;
  const span = Math.max(1, end - start);
  const innerW = Math.max(1, width - PAD_X * 2);
  const xOf = (year) => PAD_X + ((clamp(year, start, end) - start) / span) * innerW;
  const yearOf = (px) =>
    start + ((clamp(px, PAD_X, width - PAD_X) - PAD_X) / innerW) * span;

  const trackATop = TOP_PAD + LABEL_H;
  const trackABase = TOP_PAD + LABEL_H + TRACK_H;
  const trackBTop = trackABase + TRACK_GAP + LABEL_H;
  const trackBBase = trackBTop + TRACK_H;

  // Group results by year so repeated years stack rather than overlap.
  const byYear = new Map();
  rows.forEach((r, i) => {
    const y = r.publication_year ?? r.historical_start;
    if (y == null) return;
    if (!byYear.has(y)) byYear.set(y, []);
    byYear.get(y).push({ ...r, _i: i });
  });
  const yearGroups = [...byYear.entries()].sort((a, b) => a[0] - b[0]);

  const maxResults = Math.max(1, ...yearGroups.map(([, v]) => v.length));
  const maxCorpus = Math.max(1, density.max_documents);

  // Pointer -> year, relative to the SVG box.
  const localX = (e) => e.clientX - svgRef.current.getBoundingClientRect().left;

  // Deliberately not using setPointerCapture. Capturing the pointer on
  // pointerdown retargets the subsequent click event to the capturing element,
  // which meant a plain click on a decade band never reached the band's own
  // handler and decade selection silently stopped working. Tracking the drag
  // with window listeners keeps the drag robust when the pointer leaves the
  // chart, and leaves click dispatch alone.
  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    const from = yearOf(localX(e));
    dragRef.current = { from, to: from };
    setDrag({ from, to: from });

    const onMove = (ev) => {
      const d = dragRef.current;
      if (!d) return;
      d.to = yearOf(
        ev.clientX - svgRef.current.getBoundingClientRect().left
      );
      setDrag({ from: d.from, to: d.to });
    };

    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      const d = dragRef.current;
      dragRef.current = null;
      setDrag(null);
      if (!d) return;
      // A drag releases over whatever band it started or ended on, and the
      // browser may still fire that band's click. Mark the gesture as a drag so
      // the band handler does not also treat the release as a decade click.
      if (Math.abs(d.to - d.from) >= 1) lastDragEnd.current = performance.now();
      commit(d.from, d.to);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const bandW = innerW / decades.length;
  const barW = Math.max(2, bandW * (1 - BAR_INSET));

  const preview =
    drag && Math.abs(drag.to - drag.from) >= 1
      ? {
          start: clamp(Math.round(Math.min(drag.from, drag.to)), start, end),
          end: clamp(Math.round(Math.max(drag.from, drag.to)), start, end),
        }
      : null;
  const shownRange = preview || activeRange;

  return (
    <section
      className="timeline"
      ref={wrapRef}
      aria-label="Result distribution versus corpus density by decade"
    >
      <header className="timeline-head">
        <p className="timeline-caption">
          Result distribution vs. overall corpus density by decade. The two
          tracks are counted separately and are not to each other&rsquo;s scale.
        </p>
        {activeRange && (
          <button type="button" className="timeline-clear" onClick={onClearRange}>
            Clear {activeRange.start}&ndash;{activeRange.end} range
          </button>
        )}
      </header>

      <div className="timeline-plot">
        <svg
          ref={svgRef}
          className={`timeline-svg${entering ? " is-entering" : ""}`}
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Results by year against corpus density by decade, ${start} to ${end}. ${rows.length} results shown.`}
          onPointerDown={onPointerDown}
        >
          {/* Decade bands: click to isolate, drag across to choose a range.

              Rendered first so they sit behind the markers. SVG hit-testing
              follows paint order, and a transparent fill still receives events,
              so bands drawn on top would swallow every hover and drag aimed at
              a result dot. */}
          {decades.map((d, k) => (
            <rect
              key={`hit-${d.decade}`}
              className="timeline-band"
              x={PAD_X + k * bandW}
              y={trackATop}
              width={bandW}
              height={trackBBase - trackATop}
              onClick={(e) => {
                e.stopPropagation();
                // Swallow only the click that immediately follows a drag.
                if (performance.now() - lastDragEnd.current < 250) return;
                onSelectRange({ start: d.decade, end: d.decade + 9 });
              }}
            />
          ))}

          {/* Track A: this search's results, in the accent. */}
          <text className="timeline-track-label" x={PAD_X} y={TOP_PAD + 10}>
            results ({rows.length})
          </text>
          <line
            className="timeline-baseline"
            x1={PAD_X}
            y1={trackABase}
            x2={width - PAD_X}
            y2={trackABase}
          />
          {yearGroups.length === 0 && (
            <text
              className="timeline-empty"
              x={PAD_X + innerW / 2}
              y={trackATop + TRACK_H / 2}
              textAnchor="middle"
            >
              no results in this search
            </text>
          )}
          {yearGroups.map(([year, items]) => {
            const cx = xOf(year);
            return (
              <g key={year}>
                {items.length > MAX_STACKED_DOTS ? (
                  // Too many in one year for dots to stay legible: a short bar
                  // says "many" without pretending to resolve individuals.
                  <rect
                    className="timeline-result-bar"
                    x={cx - 3}
                    y={trackABase - 20}
                    width={6}
                    height={20}
                    rx={3}
                    onMouseEnter={() => setHover({ x: cx, y: trackABase - 20, year, items })}
                    onMouseLeave={() => setHover(null)}
                  />
                ) : (
                  items.map((it, k) => {
                    const cy = trackABase - DOT_R - 4 - k * DOT_GAP;
                    return (
                      <circle
                        key={`${year}-${it._i}`}
                        className="timeline-dot"
                        cx={cx}
                        cy={cy}
                        r={DOT_R}
                        tabIndex={0}
                        role="button"
                        aria-label={`${year}: passage ${it._i + 1}`}
                        onMouseEnter={() => setHover({ x: cx, y: cy, year, items: [it] })}
                        onMouseLeave={() => setHover(null)}
                        onFocus={() => setHover({ x: cx, y: cy, year, items: [it] })}
                        onBlur={() => setHover(null)}
                        onClick={() => onFocusResult(it._i)}
                      />
                    );
                  })
                )}
              </g>
            );
          })}
          <text
            className="timeline-scale"
            x={width - PAD_X}
            y={TOP_PAD + 10}
            textAnchor="end"
          >
            max {maxResults} in one year
          </text>

          {/* Track B: corpus density, deliberately recessive. */}
          <text className="timeline-track-label" x={PAD_X} y={trackBTop - 5}>
            corpus documents ({formatCount(density.total_documents)})
          </text>
          <line
            className="timeline-baseline"
            x1={PAD_X}
            y1={trackBBase}
            x2={width - PAD_X}
            y2={trackBBase}
          />
          {decades.map((d, k) => {
            const h = (d.documents / maxCorpus) * TRACK_H;
            const bx = PAD_X + k * bandW + (bandW - barW) / 2;
            return (
              <rect
                key={d.decade}
                className={`timeline-corpus-bar${d.complete ? "" : " is-partial"}`}
                x={bx}
                y={trackBBase - h}
                width={barW}
                height={Math.max(1, h)}
              >
                <title>
                  {`${d.label}: ${d.documents.toLocaleString()} documents (${
                    (d.share * 100).toFixed(1)
                  }% of corpus)${d.complete ? "" : " - partial decade"}`}
                </title>
              </rect>
            );
          })}
          <text
            className="timeline-scale"
            x={width - PAD_X}
            y={trackBTop - 5}
            textAnchor="end"
          >
            max {formatCount(maxCorpus)}
          </text>

          {/* Applied range, drawn behind the interaction layer. */}
          {shownRange && (
            <rect
              className="timeline-selection"
              x={xOf(shownRange.start)}
              y={trackATop}
              width={Math.max(1, xOf(shownRange.end) - xOf(shownRange.start))}
              height={trackBBase - trackATop}
            />
          )}

          {/* Axis labels are rendered as real buttons just below this SVG, so a
              decade is reachable by keyboard as well as by pointer. Keeping the
              text out of the SVG avoids drawing each label twice. */}
        </svg>

        <div
          className="timeline-axis-buttons"
          style={{ paddingLeft: PAD_X, paddingRight: PAD_X }}
        >
          {decades.map((d) => (
            <button
              key={`btn-${d.decade}`}
              type="button"
              className="timeline-decade-btn"
              style={{ flex: "1 1 0" }}
              onClick={() => onSelectRange({ start: d.decade, end: d.decade + 9 })}
              aria-label={`Restrict search to the ${d.label}, ${d.documents.toLocaleString()} documents`}
            >
              {d.label}
            </button>
          ))}
        </div>

        {hover && (
          <div
            className="timeline-tip"
            style={{
              left: `${clamp(hover.x, 90, width - 90)}px`,
              top: `${hover.y - 10}px`,
            }}
            role="status"
          >
            <b>{hover.year}</b>
            <span>
              {hover.items.length > 1
                ? `${hover.items.length} passages in ${hover.year}`
                : `Passage ${hover.items[0]._i + 1}: ${
                    (hover.items[0].snippet || hover.items[0].text || "")
                      .slice(0, 90)
                      .trim()
                  }…`}
            </span>
          </div>
        )}
      </div>

      <p className="timeline-hint">
        Click a decade, or drag across the chart, to re-run the search within
        those years. Selecting a range re-queries the archive rather than
        trimming the passages already shown.
      </p>
      <span className="visually-hidden">
        {`Showing corpus ${corpus} range ${start} to ${end}.`}
      </span>
    </section>
  );
}
