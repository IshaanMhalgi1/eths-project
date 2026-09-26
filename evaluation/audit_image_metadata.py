"""Does the indexed corpus retain the fields needed to build a
Chronicling America page-image URL?

Needed: LCCN (newspaper id), issue date, edition, page sequence number.
Also records what the id/date fields DO look like, so the URL can be
reconstructed or shown to be impossible.
"""
import json
import os
import re
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Candidate keys that could carry each needed field.
WANT = {
    'lccn': ['lccn', 'newspaper_id', 'paper_id', 'title_lccn', 'batch'],
    'date': ['date', 'issue_date', 'publication_date', 'pub_date', 'day'],
    'edition': ['edition', 'edition_seq', 'edition_sequence'],
    'seq': ['seq', 'sequence', 'page_seq', 'page_sequence', 'page', 'image_id',
            'sequence_number'],
    'url': ['url', 'source_url', 'image_url', 'jp2', 'jp2_url', 'pdf'],
    'place': ['place', 'location', 'city', 'state'],
}


def audit(name, obj, key):
    """Recursively collect candidate field names and sample values."""
    found, samples = {}, {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            for want, keys in WANT.items():
                if kl in keys or any(w in kl for w in keys):
                    if kl not in found:
                        found.setdefault(want, Counter())[kl] += 1
                        samples.setdefault(f"{want}.{kl}", v)
    elif isinstance(obj, list) and obj:
        return audit(name, obj[0], key)
    return found, samples


for fname in ('chunks.jsonl',):
    p = os.path.join(DATA, fname)
    if not os.path.exists(p):
        print(f"{fname}: NOT FOUND")
        continue
    n = 0
    allkeys = Counter()
    sample = None
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            n += 1
            if sample is None:
                sample = o
            for k in o.keys():
                allkeys[str(k).lower()] += 1
            if n >= 20000:
                break
    print("=" * 70)
    print(f"{fname}: {n} records sampled")
    print("=" * 70)
    print("top-level fields present:")
    for k, c in allkeys.most_common():
        print(f"  {k:<28} {c}")
    print()
    if sample:
        print("sample record (non-text):")
        for k, v in sample.items():
            if k == 'text':
                continue
            sv = str(v)
            print(f"  {k:<24} = {sv[:110]}")
