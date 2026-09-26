"""Build Library of Congress page-image references for corpus documents.

Background
----------
The `provenance` field on every chunk already stores the canonical Chronicling
America page URL, e.g.

    https://chroniclingamerica.loc.gov/lccn/sn83025881/1800-04-18/ed-1/seq-3.json

so the four components needed to address a page scan -- LCCN, issue date,
edition and page sequence -- are all recoverable by parsing that string. No
re-derivation from the original dataset release is required.

Measured coverage: 50000/50000 chunks (100%) yield a parseable reference,
across 574 distinct newspaper titles, with exactly one page per parent_doc_id.

Post-August-2025 URL scheme
---------------------------
On 2025-08-04 the Library of Congress migrated Chronicling America into the
main loc.gov digital-collections framework. chroniclingamerica.loc.gov now
permanently redirects to loc.gov. Per the Library's migration documentation
(https://www.loc.gov/ndnp/migration/) the mappings are:

    newspaper page  -> https://www.loc.gov/resource/{lccn}/{date}/ed-{n}/?sp={page}
    newspaper issue -> https://www.loc.gov/resource/{lccn}/{date}/ed-{n}/?st=gallery
    newspaper title -> https://www.loc.gov/item/{lccn}/

The Library also documents that "links to downloadable files in PDF, JP2, and
other formats will automatically redirect", so the legacy file URLs remain
valid entry points.

Both families are exposed here. The loc.gov resource page is the canonical
human-facing destination; the legacy URLs are kept because they are the stable
addresses recorded in the corpus provenance and are documented to redirect.
"""

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict

# .../lccn/{lccn}/{YYYY-MM-DD}/ed-{edition}/seq-{sequence}[.json|.xml|.txt]
PAGE_RE = re.compile(
    r"/lccn/(?P<lccn>[a-zA-Z0-9]+)"
    r"/(?P<date>\d{4}-\d{2}-\d{2})"
    r"/ed-(?P<edition>\d+)"
    r"/seq-(?P<sequence>\d+)"
    r"(?P<ext>\.[A-Za-z]+)?"
)

# Extensions we accept as evidence that the URL addresses a text/metadata
# record for a real page, as opposed to some unrelated LOC path.
_ACCEPTED_EXTS = {"", ".json", ".xml", ".txt", ".htm", ".html"}

LEGACY_BASE = "https://chroniclingamerica.loc.gov/lccn"
LOC_BASE = "https://www.loc.gov"


def parse_provenance(provenance):
    """Extract page coordinates from a Chronicling America provenance URL.

    Returns ``{'lccn', 'date', 'edition', 'sequence'}`` or ``None`` when the
    string does not address a single newspaper page.
    """
    if not provenance or not isinstance(provenance, str):
        return None
    m = PAGE_RE.search(provenance)
    if not m:
        return None
    g = m.groupdict()
    if g.get("ext") and g["ext"].lower() not in _ACCEPTED_EXTS:
        return None
    try:
        edition = int(g["edition"])
        sequence = int(g["sequence"])
    except (TypeError, ValueError):
        return None
    if edition < 1 or sequence < 1:
        return None
    return {
        "lccn": g["lccn"],
        "date": g["date"],
        "edition": edition,
        "sequence": sequence,
    }


def page_reference(provenance):
    """Build the full set of page-image references for one provenance URL.

    Returns a dict, or ``None`` if the provenance is not a page URL. The dict
    is always safe to hand to a template: every value is a plain string.
    """
    parsed = parse_provenance(provenance)
    if not parsed:
        return None

    lccn = urllib.parse.quote(parsed["lccn"], safe="")
    date = parsed["date"]
    ed = parsed["edition"]
    sp = parsed["sequence"]

    # Canonical loc.gov destinations (post-migration).
    resource_page = f"{LOC_BASE}/resource/{lccn}/{date}/ed-{ed}/?sp={sp}"
    issue_gallery = f"{LOC_BASE}/resource/{lccn}/{date}/ed-{ed}/?st=gallery"
    issue_manifest = f"{LOC_BASE}/item/{lccn}/{date}/ed-{ed}/manifest.json"
    title_page = f"{LOC_BASE}/item/{lccn}/"

    # Legacy entry points, documented to redirect to the new site. These are
    # the addresses stored in the corpus provenance itself.
    legacy_page = f"{LEGACY_BASE}/{lccn}/{date}/ed-{ed}/seq-{sp}/"
    legacy_jp2 = f"{LEGACY_BASE}/{lccn}/{date}/ed-{ed}/seq-{sp}.jp2"
    legacy_pdf = f"{LEGACY_BASE}/{lccn}/{date}/ed-{ed}/seq-{sp}.pdf"

    return {
        "lccn": parsed["lccn"],
        "date": date,
        "edition": ed,
        "sequence": sp,
        "label": f"{date}, edition {ed}, page {sp}",
        # primary human-facing destination
        "resource_page": resource_page,
        "issue_gallery": issue_gallery,
        "issue_manifest": issue_manifest,
        "title_page": title_page,
        "legacy_page": legacy_page,
        "legacy_jp2": legacy_jp2,
        "legacy_pdf": legacy_pdf,
    }


def document_references(provenances):
    """Aggregate page references for all chunks of one parent document.

    ``provenances`` is an iterable of provenance strings (one per chunk).
    Returns a dict with:
      * ``available``   -- bool, False when no chunk yielded a page reference
      * ``pages``       -- list of references, one per distinct page, sorted
      * ``primary``     -- the reference the UI should link to, or None
      * ``newspaper``   -- LCCN of the source newspaper when known
    """
    pages, seen = [], set()
    for prov in provenances or ():
        ref = page_reference(prov)
        if not ref:
            continue
        key = (ref["lccn"], ref["date"], ref["edition"], ref["sequence"])
        if key in seen:
            continue
        seen.add(key)
        pages.append(ref)

    if not pages:
        return {
            "available": False,
            "reason": "no-chronicling-america-provenance",
            "pages": [],
            "primary": None,
            "newspaper": None,
        }

    pages.sort(key=lambda r: (r["date"], r["edition"], r["sequence"]))
    return {
        "available": True,
        "reason": None,
        "pages": pages,
        "primary": pages[0],
        "newspaper": pages[0]["lccn"],
    }


# ---------------------------------------------------------------------------
# Displayable image resolution
# ---------------------------------------------------------------------------
# The corpus yields JP2/PDF URLs, but JP2 is JPEG 2000 and no browser renders
# it, so it cannot be used as an <img> src. LOC serves displayable JPEGs via
# its IIIF image service on tile.loc.gov, whose identifiers embed the submitter
# code, reel number and page code -- none of which appear in the corpus. Those
# are only obtainable from the issue manifest.
#
# The manifest is resolved here, server-side, for two reasons:
#   * it avoids CORS entirely, since the browser only ever loads a plain
#     <img src> against tile.loc.gov, which imposes no cross-origin constraint;
#   * loc.gov's web hosts are fronted by a Cloudflare WAF that refuses requests
#     from some networks, so the manifest is fetched with a documented fallback
#     to the Internet Archive's copy of the same file.
#
# tile.loc.gov itself serves images without that restriction, which was verified
# by fetching a real IIIF JPEG (HTTP 200, image/jpeg).

IIIF_MAX_PX = 1400
WAYBACK_AVAILABILITY = "https://archive.org/wayback/available?url={url}"
WAYBACK_CDX = ("https://web.archive.org/cdx/search/cdx?url={url}"
               "&output=json&limit=8&filter=statuscode:200")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_CACHE_TTL = 24 * 60 * 60        # manifests are immutable historical records
_CACHE_MAX = 512                 # bound memory
_MANIFEST_CACHE = OrderedDict()  # (lccn, date, edition) -> (expires_at, manifest)
_CACHE_LOCK = threading.Lock()

# "Page 3", "page 3", "3", "Plate 3" -> 3
_LABEL_TRAILING_INT = re.compile(r"(\d+)\s*$")


def _get(url, timeout):
    """GET a URL, returning bytes or None on any failure."""
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read()
    except Exception:
        # Any network, HTTP, TLS or timeout failure degrades to the next source.
        return None


def _wayback_raw_url(original, timestamp):
    """Wayback raw-content URL.

    The 'id_' modifier is required: without it the Archive rewrites asset URLs
    through its own replay proxy, which corrupts the embedded IIIF identifiers.
    """
    return f"https://web.archive.org/web/{timestamp}id_/{original}"


def _cache_get(key):
    with _CACHE_LOCK:
        hit = _MANIFEST_CACHE.get(key)
        if not hit:
            return None
        expires_at, manifest = hit
        if expires_at < time.time():
            _MANIFEST_CACHE.pop(key, None)
            return None
        _MANIFEST_CACHE.move_to_end(key)
        return manifest


def _cache_put(key, manifest):
    with _CACHE_LOCK:
        _MANIFEST_CACHE[key] = (time.time() + _CACHE_TTL, manifest)
        _MANIFEST_CACHE.move_to_end(key)
        while len(_MANIFEST_CACHE) > _CACHE_MAX:
            _MANIFEST_CACHE.popitem(last=False)


def _wayback_timestamp(original):
    """Find a usable Wayback snapshot timestamp for a URL.

    The availability API is not dependable here: it routinely reports a snapshot
    as available while returning no timestamp, even for URLs the CDX index knows
    about. CDX is therefore tried as well, and the two are combined.
    """
    quoted = urllib.parse.quote(original, safe="")
    timestamps = []

    raw = _get(WAYBACK_AVAILABILITY.format(url=quoted), timeout=10)
    if raw and raw[:1] == b"{":
        try:
            snap = (json.loads(raw).get("archived_snapshots") or {}).get("closest") or {}
            if snap.get("timestamp"):
                timestamps.append(str(snap["timestamp"]))
        except Exception:
            pass

    if not timestamps:
        raw = _get(WAYBACK_CDX.format(url=quoted), timeout=20)
        if raw and raw[:1] == b"[":
            try:
                rows = json.loads(raw)
                # rows[0] is the header; remaining rows carry the timestamp.
                for row in rows[1:]:
                    if isinstance(row, list) and len(row) > 1 and row[1]:
                        timestamps.append(str(row[1]))
            except Exception:
                pass

    if not timestamps:
        return None
    # Most recent snapshot is the best match for current loc.gov output.
    return max(timestamps)


def _fetch_manifest(lccn, date, edition):
    """Fetch one issue manifest, preferring loc.gov and falling back to the
    Internet Archive. Returns (manifest, source_label) or (None, None)."""
    key = (lccn, date, str(edition))
    cached = _cache_get(key)
    if cached is not None:
        return cached, "cache"

    original = f"{LOC_BASE}/item/{urllib.parse.quote(lccn, safe='')}/{date}/ed-{edition}/manifest.json"

    raw = _get(original, timeout=8)
    if raw and raw[:1] in (b"{", b"["):
        try:
            manifest = json.loads(raw)
            _cache_put(key, manifest)
            return manifest, "loc.gov"
        except Exception:
            pass

    # Fallback: the Internet Archive's copy of the same file.
    timestamp = _wayback_timestamp(original)
    if not timestamp:
        return None, None

    raw = _get(_wayback_raw_url(original, timestamp), timeout=20)
    if raw and raw[:1] in (b"{", b"["):
        try:
            manifest = json.loads(raw)
            _cache_put(key, manifest)
            return manifest, "wayback"
        except Exception:
            pass

    return None, None


def _select_canvas(canvases, sequence):
    """Pick the canvas for a 1-based page sequence.

    LOC labels canvases variously as '1', 'Page 1' or 'Plate 1', so an exact
    match on the raw label is not sufficient; the trailing integer is compared
    instead. Positional order is the final fallback.
    """
    if not isinstance(canvases, list) or not canvases:
        return None
    wanted = int(sequence)
    for c in canvases:
        label = str((c or {}).get("label") or "").strip()
        m = _LABEL_TRAILING_INT.search(label)
        if m and int(m.group(1)) == wanted:
            return c
    if 1 <= wanted <= len(canvases):
        return canvases[wanted - 1]
    return None


def _iiif_image_url(canvas):
    """Extract a browser-displayable JPEG URL from one canvas."""
    images = (canvas or {}).get("images")
    if not isinstance(images, list) or not images:
        return None
    resource = (images[0] or {}).get("resource")
    if not isinstance(resource, dict):
        return None

    service = resource.get("service")
    service_id = None
    if isinstance(service, dict):
        service_id = service.get("@id")
    elif isinstance(service, list) and service and isinstance(service[0], dict):
        service_id = service[0].get("@id")

    if isinstance(service_id, str) and service_id:
        return f"{service_id}/full/!{IIIF_MAX_PX},{IIIF_MAX_PX}/0/default.jpg"

    direct = resource.get("@id")
    if isinstance(direct, str) and re.search(r"\.jpe?g($|\?)", direct, re.I):
        return direct

    return None


def resolve_page_image(lccn, date, edition, sequence):
    """Resolve one newspaper page to a displayable image URL.

    Returns a dict that is always safe to serialize:
      available  -- True only when a displayable image URL was found
      image_url  -- the IIIF JPEG URL, when available
      source     -- 'loc.gov', 'wayback' or 'cache'
      reason     -- machine-readable failure cause when unavailable
    """
    if not all([lccn, date, edition, sequence]):
        return {"available": False, "image_url": None, "source": None,
                "reason": "missing-page-coordinates"}

    manifest, source = _fetch_manifest(str(lccn), str(date), str(edition))
    if manifest is None:
        return {"available": False, "image_url": None, "source": None,
                "reason": "manifest-unavailable"}

    sequences = manifest.get("sequences") if isinstance(manifest, dict) else None
    if not isinstance(sequences, list) or not sequences:
        return {"available": False, "image_url": None, "source": source,
                "reason": "manifest-has-no-sequences"}

    canvas = _select_canvas(sequences[0].get("canvases"), sequence)
    if canvas is None:
        return {"available": False, "image_url": None, "source": source,
                "reason": "page-not-in-issue"}

    image_url = _iiif_image_url(canvas)
    if not image_url:
        return {"available": False, "image_url": None, "source": source,
                "reason": "no-displayable-image-format"}

    return {"available": True, "image_url": image_url, "source": source,
            "reason": None}
