import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Debounced prefix autocomplete against the corpus vocabulary index.
 *
 * Suggestions are only ever corpus-derived, so the dropdown cannot propose a
 * query the retrieval system is unable to answer.
 *
 * Keyboard contract:
 *   ArrowDown / ArrowUp  move the active row (opens the list if closed)
 *   Enter                adopt the active suggestion and close; it does NOT
 *                        auto-submit, so a suggestion can be reviewed first
 *   Escape               dismiss
 * A pointer click outside the list also dismisses.
 */

const DEBOUNCE_MS = 150;
const MIN_CHARS = 2;

export function useAutocomplete({ value, apiBase = "/api", limit = 8 }) {
  const [suggestions, setSuggestions] = useState([]);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);

  const timerRef = useRef(null);
  const abortRef = useRef(null);
  const rootRef = useRef(null);
  // Distinguishes a value change from an adopted suggestion so adopting one
  // does not immediately trigger a second lookup.
  const suppressNext = useRef(false);

  const reset = useCallback(() => {
    setSuggestions([]);
    setOpen(false);
    setActiveIndex(-1);
  }, []);

  // Debounced lookup, cancelled on every keystroke.
  useEffect(() => {
    const q = (value || "").trim();

    if (suppressNext.current) {
      suppressNext.current = false;
      return undefined;
    }
    if (q.length < MIN_CHARS) {
      reset();
      return undefined;
    }

    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      if (abortRef.current) abortRef.current.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      try {
        const params = new URLSearchParams({ q, limit: String(limit) });
        const res = await fetch(`${apiBase}/autocomplete?${params}`, { signal: ac.signal });
        if (!res.ok) {
          reset();
          return;
        }
        const data = await res.json();
        const items = Array.isArray(data?.suggestions) ? data.suggestions : [];
        setSuggestions(items);
        setOpen(items.length > 0);
        setActiveIndex(items.length > 0 ? 0 : -1);
      } catch (err) {
        if (err?.name !== "AbortError") reset();
      }
    }, DEBOUNCE_MS);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
    // `value` is the only meaningful input; limit/apiBase are static config.
  }, [value, limit, apiBase, reset]);

  // Dismiss on outside pointer press.
  useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) {
        setOpen(false);
        setActiveIndex(-1);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open]);

  const adopt = useCallback((text) => {
    suppressNext.current = true;
    setSuggestions([]);
    setOpen(false);
    setActiveIndex(-1);
    return text;
  }, []);

  const onKeyDown = useCallback(
    (e) => {
      if (e.key === "Escape") {
        if (open) {
          e.preventDefault();
          setOpen(false);
          setActiveIndex(-1);
        }
        return undefined;
      }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        if (suggestions.length === 0) return undefined;
        e.preventDefault();
        if (!open) {
          setOpen(true);
          setActiveIndex(e.key === "ArrowDown" ? 0 : suggestions.length - 1);
          return undefined;
        }
        const delta = e.key === "ArrowDown" ? 1 : -1;
        setActiveIndex((prev) => {
          const next = prev + delta;
          if (next < 0) return suggestions.length - 1;
          if (next >= suggestions.length) return 0;
          return next;
        });
        return undefined;
      }
      if (e.key === "Enter" && open && activeIndex >= 0) {
        e.preventDefault();
        return adopt(suggestions[activeIndex]);
      }
      return undefined;
    },
    [open, suggestions, activeIndex, adopt]
  );

  return {
    suggestions,
    open,
    activeIndex,
    rootRef,
    onKeyDown,
    adopt,
    setActiveIndex,
    close: () => {
      setOpen(false);
      setActiveIndex(-1);
    },
  };
}
