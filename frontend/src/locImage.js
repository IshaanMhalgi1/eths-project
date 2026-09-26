/**
 * Client for displayable page-image resolution.
 *
 * The browser cannot resolve the image itself: loc.gov's web hosts are served
 * behind a Cloudflare WAF that refuses cross-origin and non-browser requests,
 * and the IIIF identifiers live in the issue manifest rather than in the corpus
 * provenance. The backend resolves the manifest (with an Internet Archive
 * fallback) and returns a tile.loc.gov IIIF URL, which the browser loads as a
 * plain <img> and is not subject to CORS.
 *
 * This module never throws: a failed lookup resolves to a reason string so the
 * UI can show an honest "unavailable" state and keep the direct LOC links.
 */

const API_BASE = '/api';

/**
 * @returns {Promise<{available: boolean, imageUrl: string|null, reason: string|null}>}
 */
export async function fetchPageImage({ lccn, date, edition, sequence, signal }) {
  if (!lccn || !date || !edition || !sequence) {
    return { available: false, imageUrl: null, reason: 'missing-page-coordinates' };
  }
  const params = new URLSearchParams({
    lccn: String(lccn),
    date: String(date),
    edition: String(edition),
    sequence: String(sequence),
  });
  try {
    const res = await fetch(`${API_BASE}/page-image?${params}`, { signal });
    if (!res.ok) {
      return { available: false, imageUrl: null, reason: `http-${res.status}` };
    }
    const data = await res.json();
    return {
      available: Boolean(data?.available && data?.image_url),
      imageUrl: data?.image_url ?? null,
      reason: data?.reason ?? null,
    };
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    return { available: false, imageUrl: null, reason: 'network-error' };
  }
}
