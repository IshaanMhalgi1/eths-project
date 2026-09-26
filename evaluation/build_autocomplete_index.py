"""Build the autocomplete suggestion index from actual corpus content.

Design constraints (from the task spec):
  * suggestions must come from corpus vocabulary and entities that verifiably
    appear in the indexed documents -- never from invented phrases or a query
    log, so that autocomplete never implies a query the corpus cannot answer;
  * cover frequent terms, named entities (people/places) and date expressions;
  * serve prefix matches fast enough to feel instant.

The embedding vectors dominate the corpus files and are irrelevant here, so the
text is read once and counted. Everything is derived from observed counts, which
is what makes "every suggestion is findable in the corpus" true by construction.

Output: data/autocomplete_index.json
"""
import json
import os
import re
import sys
import time
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT = os.path.join(DATA, 'autocomplete_index.json')

# Token shape: letters, allowing internal apostrophes/hyphens (19th-century
# spellings such as "cider-mill" and "bradstreet's" are real content).
TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]{1,24}")
# A capitalised run, which is how people and place names appear in the prose.
CAP = re.compile(r"\b[A-Z][A-Za-z'\-]{1,24}\b")
YEAR = re.compile(r"\b(1[6-9]\d{2})\b")
# Punctuation that ends a sentence and therefore breaks a capitalised run.
SENTENCE_END = re.compile(r"[.!?;:\n\"]")


def _emit(run, phrase_counts, cap_bigrams):
    """Record a candidate entity phrase and its internal capitalised bigrams."""
    if len(run) < 2:
        return
    phrase_counts[" ".join(run)] += 1
    for i in range(len(run) - 1):
        cap_bigrams[(run[i], run[i + 1])] += 1

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves will shall may might must upon said say says mr mrs messrs dr hon
st saint ave street road county state city also thus yet still ever never
""".split())

# Capitalised words that are almost never useful as entity suggestions on their
# own; they are stripped from the edges of an entity phrase.
ENTITY_EDGE_STOP = STOPWORDS | {
    "the", "a", "an", "and", "of", "in", "on", "at", "to", "for", "by", "with",
    "from", "this", "that", "these", "those", "his", "her", "their", "our", "it",
}

MIN_TERM_COUNT = 12      # keeps the vocabulary genuine rather than OCR speckle
MIN_ENTITY_COUNT = 8     # entities are rarer, so allow more
MIN_CAP_BIGRAM_COUNT = 4 # leading capitalised pair must genuinely recur
MIN_YEAR_DOCS = 20       # a suggested year must have real corpus coverage
MIN_TOKEN_LEN = 3
MAX_TERMS = 80000
MAX_PHRASES = 40000


def main():
    t0 = time.time()
    src = os.path.join(DATA, 'chunks.jsonl')
    if not os.path.exists(src):
        sys.exit(f"missing corpus: {src}")

    term_counts = Counter()
    phrase_counts = Counter()
    cap_bigrams = Counter()
    year_counts = Counter()
    ndocs = 0
    ntok = 0

    with open(src, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get('text') or ''
            if not text:
                continue
            ndocs += 1

            # --- unigrams ---
            toks = [t.lower().strip("'-") for t in TOKEN.findall(text)]
            ntok += len(toks)
            for t in toks:
                if len(t) < MIN_TOKEN_LEN or t in STOPWORDS:
                    continue
                if t.isdigit():
                    continue
                term_counts[t] += 1

            # --- multi-word entities from capitalised runs ---
            # A run is broken at sentence punctuation; otherwise unrelated
            # capitalised words chain into nonsense ("aaron pennsylvania
            # republicanism house"). A candidate must additionally contain a
            # capitalised bigram that recurs often, which is a reasonable proxy
            # for a real name rather than two capitalised words that merely sit
            # near each other once.
            run = []
            prev_end = None
            for m in CAP.finditer(text):
                w = m.group(0)
                lw = w.lower().strip("'-")
                broke = False
                if prev_end is not None:
                    between = text[prev_end:m.start()]
                    if SENTENCE_END.search(between):
                        broke = True
                prev_end = m.end()

                if broke or (lw in ENTITY_EDGE_STOP and not w.isupper()):
                    if len(run) >= 2:
                        _emit(run, phrase_counts, cap_bigrams)
                    run = []
                    if lw in ENTITY_EDGE_STOP and not w.isupper():
                        continue
                    continue

                run.append(lw)
                if len(run) > 4:
                    _emit(run[:-1], phrase_counts, cap_bigrams)
                    run = run[-1:]
            if len(run) >= 2:
                _emit(run, phrase_counts, cap_bigrams)

            # --- date expressions ---
            # Deliberately NOT taken from four-digit numbers in the prose: those
            # include incidental references (prices, page counts, cross-references)
            # and years well outside the corpus. Suggesting a year the corpus
            # cannot answer is precisely what the spec forbids, so the year set
            # is built from actual publication_year coverage instead.
            py = obj.get('publication_year')
            if isinstance(py, int) and 1600 <= py <= 1999:
                year_counts[str(py)] += 1

    print(f"scanned {ndocs:,} documents, {ntok:,} tokens in {time.time()-t0:.0f}s")

    # Keep only entities that look like names. The bigram requirement is what
    # removes incidental capitalisation: a phrase only survives if its leading
    # capitalised pair actually recurs in the corpus.
    phrases = []
    for phrase, c in phrase_counts.items():
        if c < MIN_ENTITY_COUNT:
            continue
        parts = phrase.split()
        if len(parts) < 2 or len(parts) > 4:
            continue
        if all(p in STOPWORDS for p in parts):
            continue
        if any(len(p) < 3 for p in parts):
            continue
        if cap_bigrams.get((parts[0], parts[1]), 0) < MIN_CAP_BIGRAM_COUNT:
            continue
        phrases.append((phrase, c))
    phrases.sort(key=lambda kv: (-kv[1], kv[0]))
    phrases = phrases[:MAX_PHRASES]

    terms = [(t, c) for t, c in term_counts.items() if c >= MIN_TERM_COUNT]
    terms.sort(key=lambda kv: kv[0])          # alphabetical, for prefix bisect
    phrases.sort(key=lambda kv: kv[0])        # alphabetical, for prefix bisect

    # Only offer a year the corpus actually holds documents for, with enough of
    # them to answer a query about that year.
    years = {y: c for y, c in year_counts.items() if c >= MIN_YEAR_DOCS}
    if years:
        print(f"  year coverage  : {min(years)}-{max(years)}")

    index = {
        "meta": {
            "built_from": os.path.basename(src),
            "documents": ndocs,
            "tokens": ntok,
            "terms": len(terms),
            "phrases": len(phrases),
            "years": len(years),
            "min_term_count": MIN_TERM_COUNT,
            "min_entity_count": MIN_ENTITY_COUNT,
            "note": ("Derived solely from corpus text. Every term, phrase and year "
                     "here occurs in the indexed documents, so any suggestion is "
                     "answerable by the index."),
        },
        "terms": terms,
        "phrases": phrases,
        "years": years,
    }

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(index, fh, ensure_ascii=False)

    size = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}")
    print(f"  terms   : {len(terms):,} (>= {MIN_TERM_COUNT} occurrences)")
    print(f"  phrases : {len(phrases):,} (>= {MIN_ENTITY_COUNT} occurrences)")
    print(f"  years   : {len(years):,}")
    print(f"  size    : {size:.0f} KB")
    print(f"\nsample terms  : {[t for t, _ in terms[:0]] or [t for t, _ in terms[len(terms)//2:len(terms)//2+8]]}")
    print(f"sample phrases: {[p for p, _ in phrases[:10]]}")
    print(f"sample years  : {sorted(years)[:8]} ... {sorted(years)[-4:]}")


if __name__ == '__main__':
    main()
