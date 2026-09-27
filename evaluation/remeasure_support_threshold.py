"""Re-measure the support threshold against a LIVE model.

Why this exists
---------------
`SUPPORT_MIN = 0.34` was calibrated with extractive claims only: sentences built
by copying words straight out of the cited snippets, which score 1.00 by
construction. Real paraphrases -- the thing the threshold actually has to pass in
production -- were never measured, because no live model was available. The
measured noise ceiling for period vocabulary drawn from unrelated documents is
0.33, leaving a 0.01 margin. That is too thin to ship on, because:

  * if genuine paraphrases score below 0.34, the feature rejects almost all
    real output and silently shows nothing;
  * if contradictions also score above 0.34, the feature shows fabrications.

Both failure modes look identical from the UI: no summary appears.

Usage
-----
    # configure in .env at the repo root (api/main.py loads it; this script
    # does too), or set the variables in the shell:
    #   LLM_API_KEY=gsk_...
    #   LLM_BASE_URL=https://api.groq.com/openai/v1
    #   LLM_MODEL=openai/gpt-oss-120b
    python evaluation/remeasure_support_threshold.py

Then read the verdict and set LLM_SUPPORT_MIN accordingly.

Note: this spends real tokens. It makes a few dozen short model calls.
"""
import os
import random
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass

from api.overview import (  # noqa: E402
    SUPPORT_MIN,
    _content_words,
    build_sources,
    call_model,
    verify,
)
from ranking.ranker import final_search  # noqa: E402

QUERIES = [
    "cotton prices",
    "railroad opening",
    "yellow fever epidemic",
    "patent medicine advertisement",
    "parliament debate",
    "steamship arrival",
]

# Seconds between model calls. Free tiers throttle bursts hard, and a throttled
# run yields a smaller sample, which makes the verdict less trustworthy rather
# than more. Raise this if you see "rate limited" lines.
CALL_GAP_SECONDS = float(os.environ.get("REMEASURE_GAP", 10))

# Queries deliberately outside the corpus's thematic coverage, to sample
# paraphrases over awkward, low-coherence OCR rather than only clean passages.
ADVERSARIAL_QUERIES = [
    "why did the war in Bulgaria end so badly",
    "quarterly earnings of the Baltic Trading Company",
    "the 1987 hurricane landfall",
]

# Contradiction probes: each reuses a lot of a real snippet's vocabulary while
# asserting the opposite, which is exactly what a bag-of-words check cannot see.
CONTRADICTION_TEMPLATE = (
    "The {topic} in this passage rose sharply and caused panic among traders, "
    "prompting an immediate ban [1]."
)
TOPIC_WORDS = ["price", "price", "arrival", "meeting", "cargo", "trade"]


def support_of(claim_text, sources, citations):
    """Recompute the same support score verify() uses, for a single claim.

    `citations` must be the sources the model actually cited. Scoring against a
    guessed source produces meaningless numbers, so rejected claims carry their
    real citations through from verify().
    """
    valid = {s["index"] for s in sources}
    good = [n for n in (citations or []) if n in valid]
    if not good:
        return None
    words = _content_words(claim_text)
    if not words:
        return None
    by_index = {s["index"]: s for s in sources}
    blob = " ".join(
        w for n in good for w in _content_words(by_index[n]["text"])
    )
    if not blob:
        return None
    return sum(1 for w in words if w in blob) / len(words)


def call_with_backoff(query, sources, attempts=4):
    """Call the model, backing off on rate limits.

    Free-tier providers throttle aggressively; a burst of nine calls gets most
    of them rejected, which silently shrinks the sample and biases the result
    toward whatever survived.
    """
    delay = 6
    for i in range(attempts):
        text, err = call_model(query, sources)
        if err != "rate-limited":
            return text, err
        if i < attempts - 1:
            print(f"    rate limited, retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None, "rate-limited"


def measure(query, label):
    results = final_search(query, size=20)
    sources = build_sources(results, 6)
    if not sources:
        return []
    # Space out calls: free-tier limits throttle bursts, and a throttled run
    # silently produces a smaller, differently-composed sample.
    time.sleep(CALL_GAP_SECONDS)
    text, err = call_with_backoff(query, sources)
    if err:
        print(f"  [{label}] {query!r}: call failed ({err})")
        return []
    v = verify(text, sources)
    kept, rejected = [], []
    for c in v["claims"]:
        kept.append(c["support"])
    # Score claims enforcement threw away: if genuine paraphrases are being
    # dropped, that shows up here and nowhere else.
    for r in v["rejected"]:
        if r["reason"] in ("unsupported", "verbatim"):
            s = support_of(r["text"], sources, r.get("citations"))
            if s is not None:
                rejected.append((r["reason"], s, r["text"]))
    if kept or rejected:
        print(f"  [{label}] {query!r}: kept={len(kept)} {[round(s, 2) for s in kept]}")
        for reason, score, text in rejected:
            print(f"      REJECTED {reason} score={score:.2f}: {text[:160]}")
    return kept + [s for _, s, _ in rejected]


def contradiction_scores():
    """Score contradiction probes against real snippets, no model needed."""
    results = final_search("cotton prices", size=20)
    sources = build_sources(results, 6)
    if not sources:
        return []
    random.seed(3)
    words = _content_words(" ".join(s["text"] for s in sources))
    out = []
    for topic in TOPIC_WORDS:
        claim = CONTRADICTION_TEMPLATE.format(topic=topic)
        out.append(support_of(claim, sources, [1]))
    # And a control: genuinely on-topic text drawn from the snippets.
    for i, s in enumerate(sources[:3], start=1):
        clause = " ".join(_content_words(s["text"])[:12])
        out.append(support_of(clause, sources, [i]))
    return [x for x in out if x is not None]


def main():
    if not os.environ.get("LLM_API_KEY"):
        print("LLM_API_KEY is not set. This script needs a live model.")
        print("Set it in .env at the repo root (or the environment):")
        print("  LLM_API_KEY=gsk_...")
        print("  LLM_BASE_URL=https://api.groq.com/openai/v1")
        print("  LLM_MODEL=openai/gpt-oss-120b")
        print("Check available models with: python evaluation/list_available_models.py")
        return 2

    print("=== genuine paraphrase scores (live model) ===")
    genuine = []
    for q in QUERIES:
        genuine += measure(q, "normal")
    for q in ADVERSARIAL_QUERIES:
        genuine += measure(q, "adversarial")

    print("\n=== contradiction vs on-topic scores (no model) ===")
    contra = contradiction_scores()
    print(f"  scores: {[round(x, 2) for x in contra]}")

    print(f"\ncurrent SUPPORT_MIN = {SUPPORT_MIN}")
    if genuine:
        g = sorted(genuine)
        print(f"genuine paraphrase scores: n={len(g)} "
              f"min={g[0]:.2f} p10={g[len(g)//10]:.2f} median={g[len(g)//2]:.2f} max={g[-1]:.2f}")
        below = [round(s, 2) for s in g if s < SUPPORT_MIN]
        if below:
            print(f"  !! {len(below)} genuine paraphrase(s) score BELOW the threshold: {below}")
    if contra:
        c = sorted(contra)
        print(f"contradiction/on-topic scores: min={c[0]:.2f} median={c[len(c)//2]:.2f} max={c[-1]:.2f}")

    if not genuine:
        print("\nNo genuine scores captured (every call failed or returned nothing).")
        print("If calls were rate limited, wait and re-run; a truncated sample is not evidence.")
        return 1

    lo, hi = min(genuine), max(genuine)
    print("\nVERDICT")
    if lo >= SUPPORT_MIN:
        print(f"  Genuine paraphrases clear the threshold (lowest {lo:.2f} >= {SUPPORT_MIN}). "
              "Keep it, or raise it toward the contradiction ceiling if there is room.")
    else:
        print(f"  GENUINE PARAPHRASES ARE BEING REJECTED (lowest {lo:.2f} < {SUPPORT_MIN}).")
        print(f"  Set LLM_SUPPORT_MIN to about {max(0.05, round(lo - 0.02, 2))} and re-run.")
    if contra:
        print(f"  Contradiction ceiling is {max(contra):.2f}. Any threshold below that "
              "cannot separate contradictions from genuine text; that is a limitation "
              "of lexical matching, not a tuning problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
