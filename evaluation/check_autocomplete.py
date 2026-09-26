"""Spot-check: do autocomplete suggestions actually return relevant results?

The spec requires that suggestions come from real corpus content and that a
sample of them "actually return relevant results when searched". This exercises
the live endpoint end to end and verifies, for each suggestion, that:

  1. the search returns at least one result, and
  2. the suggestion's content words actually occur in the returned snippets.

The second check is what makes this a relevance test rather than a smoke test:
a term can be frequent in the corpus yet retrieve nothing for itself, and that
would be a real weakness in the suggestion index.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "http://127.0.0.1:8000"
# Diverse prefixes chosen to cover the three suggestion families the index
# produces: common terms, multi-word entities, and years.
PREFIXES = [
    "gol", "tele", "washi", "abrah", "linc", "miss", "rail", "gold",
    "confed", "presid", "senat", "elect", "govern", "colonel", "harvest",
    "wom", "strik", "panic", "exposition", "arbitr", "graduat", "temper",
    "nas", "chi", "san", "new", "180", "183", "185", "186",
]

STOP = set("a an the of and or in on at to for by with from is was are".split())
TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]{1,24}")


def get(path, timeout=30):
    req = urllib.request.Request(API + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_search(query, timeout=120):
    body = json.dumps({"query": query, "size": 10}).encode()
    req = urllib.request.Request(
        API + "/search", data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def content_words(suggestion):
    return [w.lower() for w in TOKEN.findall(suggestion) if w.lower() not in STOP]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    prefixes = PREFIXES[:n]

    tested = 0
    with_results = 0
    grounded = 0
    empty = []
    ungrounded = []
    rows = []
    t0 = time.time()

    for prefix in prefixes:
        try:
            data = get("/autocomplete?q=" + urllib.parse.quote(prefix) + "&limit=1")
        except urllib.error.URLError as e:
            print(f"autocomplete endpoint unreachable: {e}")
            return 1
        sugg = (data.get("suggestions") or [None])[0]
        if not sugg:
            rows.append((prefix, "-", 0, "-"))
            continue

        tested += 1
        try:
            res = post_search(sugg)
        except Exception as e:
            rows.append((prefix, sugg, 0, f"error {type(e).__name__}"))
            empty.append(sugg)
            continue

        results = res.get("results") or []
        if results:
            with_results += 1
        else:
            empty.append(sugg)

        words = content_words(sugg)
        blob = " ".join((r.get("text") or "") for r in results).lower()

        if not words and sugg.isdigit():
            # Date suggestions are validated differently: a bare year rarely
            # appears verbatim in OCR text, so check the temporal metadata the
            # ranker actually returned instead.
            year = int(sugg)
            near = 0
            for r in results:
                vals = [r.get("historical_start"), r.get("historical_end"),
                        r.get("publication_year")]
                yrs = [v for v in vals if isinstance(v, int)]
                if any(abs(v - year) <= 2 for v in yrs):
                    near += 1
            ok = near > 0
            note = f"{near}/{len(results)} results within 2yr"
        else:
            hit = [w for w in words if w in blob]
            ok = bool(results) and bool(words) and len(hit) == len(words)
            note = f"{len(hit)}/{len(words)} terms"

        if ok:
            grounded += 1
        elif results:
            ungrounded.append((sugg, note))
        else:
            ungrounded.append((sugg, note))

        rows.append((prefix, sugg, len(results), note))

    elapsed = time.time() - t0
    print("=" * 74)
    print(f"AUTOCOMPLETE SPOT-CHECK  ({tested} suggestions, {elapsed:.0f}s)")
    print("=" * 74)
    print(f"{'prefix':10} {'suggestion':34} {'hits':>5}  grounded-terms")
    print("-" * 74)
    for prefix, sugg, hits, note in rows:
        print(f"{prefix:10} {sugg[:33]:34} {hits:5}  {note}")

    print()
    print(f"  suggestions tested              : {tested}")
    print(f"  returned at least one result    : {with_results}/{tested}")
    print(f"  all suggestion terms in snippets: {grounded}/{tested}")
    if empty:
        print(f"  returned nothing                : {empty}")
    if ungrounded:
        print("  not fully grounded             :")
        for s, note in ungrounded:
            print(f"      {s}  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
