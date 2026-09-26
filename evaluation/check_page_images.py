"""End-to-end check: resolve a sample of real corpus documents to a JPEG that
actually serves, and report the honest success rate.

Reports separately on whether resolution succeeded and whether the resulting
IIIF URL actually returns image bytes, because those are different claims.
"""
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'api'))
from loc_images import document_references, resolve_page_image  # noqa: E402

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

# Sample documents straight from the corpus so the sample is real, not curated.
docs = []
with open(os.path.join(DATA, 'chunks.jsonl'), encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        docs.append(o.get('provenance'))
        if len(docs) >= 4000:
            break
random.seed(7)
sample = random.sample(docs, N)

ok_resolve = 0
ok_image = 0
reasons = {}
sources = {}
t0 = time.time()

for prov in sample:
    ref = document_references([prov])["primary"]
    r = resolve_page_image(ref['lccn'], ref['date'], ref['edition'], ref['sequence'])
    if not r['available']:
        reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        continue
    ok_resolve += 1
    sources[r['source']] = sources.get(r['source'], 0) + 1
    try:
        req = urllib.request.Request(r['image_url'], headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=90) as resp:
            head = resp.read(3)
            if resp.status == 200 and head == bytes([0xff, 0xd8, 0xff]):
                ok_image += 1
    except Exception as e:
        reasons['image-fetch-failed'] = reasons.get('image-fetch-failed', 0) + 1

elapsed = time.time() - t0
print("=" * 66)
print(f"SAMPLED {N} real corpus documents  ({elapsed:.1f}s total)")
print("=" * 66)
print(f"  resolved to a displayable URL : {ok_resolve}/{N}")
print(f"  URL actually served a JPEG    : {ok_image}/{N}")
print(f"  manifest sources              : {sources}")
print(f"  failures                      : {reasons or 'none'}")
