"""Blog source — RSS / sitemap discovery + trafilatura extraction.

Mirrors `app/ingest/sources/twitter.py` shape: an async-generator
`fetch()` that yields one `RawItem` per post (NOT per chunk —
chunking happens later in `app/ingest/chunker.chunk_blog_paragraphs`).
The CLI is responsible for stitching fetch() → chunker → upsert.

Discovery cascades through several known endpoints, stopping at the
first one that returns ≥1 URL:

  1. {scheme}://{domain}/index.xml          (Hugo-style RSS)
  2. {scheme}://{domain}/feed/              (WordPress-style RSS)
  3. {scheme}://{domain}/rss.xml            (generic)
  4. {scheme}://{domain}/sitemap.xml        (sitemap)
  5. {scheme}://{domain}/sitemap_index.xml  (sitemap index)

If none of these work we'd need archive-page scraping; that's not
implemented today and would need per-blog selectors anyway, so we
log + abort. Most operator blogs expose at least one of these so the
discovery cascade is sufficient in practice.

Per-post fetch:
  - Cached at /srv/data/cache/blog/{domain}/{md5(url)}.html.
    Re-runs are free unless --refresh busts the cache.
  - 1.0 s + jitter between fetches; User-Agent identifies the bot.
  - trafilatura extracts title + body; we additionally parse
    headings ourselves so chunker can prepend section context.

Budget caps mirror Twitter's pattern: max_posts (default 500) and
max_cost_usd (default off — bandwidth-bound, not API-bound).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urljoin, urlparse

import httpx

from app.config import settings
from app.ingest import RawItem
from app.ingest.sources.base import Source

log = logging.getLogger(__name__)

_USER_AGENT = "council-bot/0.1 (research; +https://github.com/council-brain/council)"
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_REQUEST_DELAY = 1.0
_DEFAULT_JITTER = 0.3
_DEFAULT_MAX_POSTS = 500


class BlogBudgetExceeded(Exception):
    """Mirror of TwitterBudgetExceeded — raised when the post or cost
    cap trips. Caller stitches what's already been emitted, then
    propagates the abort."""


class BlogSource(Source):
    source_type = "blog"

    _DISCOVERY_PATHS = (
        "/index.xml",          # Hugo / static-site RSS
        "/feed",               # Substack RSS (no trailing slash)
        "/feed/",              # WordPress RSS (with trailing slash)
        "/rss.xml",            # generic
        "/sitemap.xml",        # WordPress (sitemap-index) + Substack (flat urlset)
        "/sitemap_index.xml",  # WordPress alternate
    )

    # If the first successful discovery probe returns fewer than this
    # many posts, treat it as suspicious (RSS truncation is the most
    # common cause — Substack caps /feed at ~20) and union with later
    # probes. Stops as soon as the union crosses the threshold.
    _DISCOVERY_SPARSE_THRESHOLD = 30

    def __init__(
        self,
        *,
        domain: str,
        since: datetime,
        rss_url: str | None = None,
        cache_dir: Path | None = None,
        max_posts: int | None = _DEFAULT_MAX_POSTS,
        max_cost_usd: float | None = None,
        refresh: bool = False,
        request_delay_seconds: float = _DEFAULT_REQUEST_DELAY,
        request_jitter_seconds: float = _DEFAULT_JITTER,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.domain = domain.strip().lower().lstrip("/").rstrip("/")
        if "://" in self.domain:
            self.domain = urlparse(self.domain).netloc or self.domain
        self.base_url = f"https://{self.domain}"
        self.since = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        # Explicit RSS / sitemap URL override — tried first, before the
        # standard discovery cascade. Useful when the canonical feed
        # lives on an unrelated host (e.g. paulgraham.com's RSS at
        # aaronsw.com) or at a non-default path.
        self.rss_url = rss_url
        self.max_posts = max_posts
        self.max_cost_usd = max_cost_usd  # parity with Twitter; bandwidth ~free
        self.refresh = refresh
        self.request_delay_seconds = request_delay_seconds
        self.request_jitter_seconds = request_jitter_seconds
        self.timeout_seconds = timeout_seconds

        self.cache_dir = (
            cache_dir
            if cache_dir is not None
            else settings.data_dir / "cache" / "blog" / self.domain
        )

        self._external_client = http_client
        self.stats = BlogStats()

    # ------------------------------------------------------------------
    # Public fetch — async generator of one RawItem per post
    # ------------------------------------------------------------------

    async def fetch(self) -> AsyncIterator[RawItem]:
        client, owns = self._acquire_client()
        try:
            urls_with_dates = await self._discover_post_urls(client)
            log.info(
                "blog: discovered %d posts at %s via %s",
                len(urls_with_dates),
                self.domain,
                self.stats.discovery_method or "(none)",
            )

            kept_for_since = []
            for url, dt in urls_with_dates:
                if dt is not None and dt < self.since:
                    self.stats.posts_below_since += 1
                    continue
                kept_for_since.append((url, dt))

            log.info(
                "blog: %d posts pass since-filter (cutoff %s)",
                len(kept_for_since),
                self.since.date().isoformat(),
            )

            for url, dt in kept_for_since:
                self._enforce_caps()
                try:
                    html = await self._fetch_with_cache(client, url)
                except httpx.HTTPError as e:
                    self.stats.fetch_errors += 1
                    log.warning("blog: fetch failed for %s: %s", url, e)
                    continue
                if not html:
                    continue

                post = _extract_post(html, url=url, fallback_date=dt)
                if post is None:
                    self.stats.extract_errors += 1
                    log.warning("blog: extraction returned no body for %s", url)
                    continue

                self.stats.posts_emitted += 1
                yield post

            log.info(
                "blog: done. discovered=%d emitted=%d cache_hits=%d "
                "cache_misses=%d below_since=%d fetch_errors=%d "
                "extract_errors=%d",
                self.stats.posts_discovered,
                self.stats.posts_emitted,
                self.stats.cache_hits,
                self.stats.cache_misses,
                self.stats.posts_below_since,
                self.stats.fetch_errors,
                self.stats.extract_errors,
            )
        finally:
            if owns:
                await client.aclose()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def _discover_post_urls(
        self, client: httpx.AsyncClient
    ) -> list[tuple[str, datetime | None]]:
        """Probe each known discovery endpoint in order; union URLs
        across probes when the first hit is sparse (< threshold).

        Substack /feed truncates to ~20 most recent posts but
        /sitemap.xml exposes the full archive. The union strategy
        catches that without per-host special-casing — when /feed
        returns 20 we keep going to /sitemap.xml and merge."""
        seen: dict[str, datetime | None] = {}
        successful_paths: list[str] = []

        # Try the explicit override URL first, if configured. This is
        # the only way to point at a feed hosted on a different domain
        # than the blog's canonical host (e.g. paulgraham.com essays
        # via aaronsw.com's RSS).
        probe_paths: list[tuple[str, str]] = []
        if self.rss_url:
            probe_paths.append((self.rss_url, self.rss_url))
        for path in self._DISCOVERY_PATHS:
            probe_paths.append((path, urljoin(self.base_url, path)))

        for label, url in probe_paths:
            urls = await self._discover_at(client, url, depth=0)
            if not urls:
                continue
            successful_paths.append(label)
            for u, d in urls:
                # Prefer dated entries (sitemap lastmod / RSS pubDate
                # are reliable; archive pages may have None).
                if u not in seen or (d is not None and seen[u] is None):
                    seen[u] = d
            if len(seen) >= self._DISCOVERY_SPARSE_THRESHOLD:
                break

        if seen:
            self.stats.discovery_method = " + ".join(successful_paths)
            self.stats.posts_discovered = len(seen)
            return list(seen.items())

        log.error(
            "blog: no working RSS / sitemap discovery endpoint at %s; "
            "tried %s",
            self.base_url,
            ", ".join(self._DISCOVERY_PATHS),
        )
        return []

    async def _discover_at(
        self,
        client: httpx.AsyncClient,
        url: str,
        depth: int,
    ) -> list[tuple[str, datetime | None]]:
        """Fetch one discovery URL and interpret it. Returns post URLs
        for RSS / sitemap; recursively follows sitemap-index entries
        (WordPress's `/sitemap_index.xml` or `/sitemap.xml` is usually
        an index pointing at `/post-sitemap.xml`, `/page-sitemap.xml`,
        etc., and only the leaves contain `<urlset>`)."""
        if depth > 3:
            return []
        try:
            resp = await client.get(url)
        except httpx.HTTPError as e:
            log.debug("blog: discovery probe failed %s: %s", url, e)
            return []
        if resp.status_code != 200 or not resp.content:
            return []
        text = resp.text

        head = text[:2048].lower()
        if "<sitemapindex" in head:
            child_urls = _parse_sitemap_index(text)
            merged: list[tuple[str, datetime | None]] = []
            for child in child_urls:
                # Skip page-only / non-post sitemaps to avoid crawling
                # nav pages. The substring filter is permissive — any
                # child sitemap whose URL contains "post" is included.
                if "page-sitemap" in child or "category-sitemap" in child:
                    continue
                merged.extend(await self._discover_at(client, child, depth + 1))
            return merged

        if _looks_like_rss(text):
            return _parse_rss(text)
        return _parse_sitemap(text)

    # ------------------------------------------------------------------
    # Per-post fetch + cache
    # ------------------------------------------------------------------

    async def _fetch_with_cache(
        self, client: httpx.AsyncClient, url: str
    ) -> str | None:
        cache_path = self._cache_path_for(url)
        if not self.refresh and cache_path.exists():
            self.stats.cache_hits += 1
            try:
                return cache_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.warning("blog: cache read failed %s: %s", cache_path, e)

        self.stats.cache_misses += 1
        # polite jitter before going to the network
        delay = self.request_delay_seconds + random.uniform(
            0, max(0.0, self.request_jitter_seconds)
        )
        if delay > 0:
            await asyncio.sleep(delay)

        resp = await client.get(url)
        if resp.status_code != 200:
            log.warning("blog: %s returned status %d", url, resp.status_code)
            return None
        text = resp.text

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        except OSError as e:
            log.warning("blog: cache write failed %s: %s", cache_path, e)
        return text

    def _cache_path_for(self, url: str) -> Path:
        h = hashlib.md5(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.html"

    # ------------------------------------------------------------------
    # HTTP plumbing + budget
    # ------------------------------------------------------------------

    def _acquire_client(self) -> tuple[httpx.AsyncClient, bool]:
        if self._external_client is not None:
            return self._external_client, False
        client = httpx.AsyncClient(
            headers={
                "user-agent": _USER_AGENT,
                "accept": "text/html, application/xml, application/rss+xml, */*",
            },
            follow_redirects=True,
            timeout=self.timeout_seconds,
        )
        return client, True

    def _enforce_caps(self) -> None:
        if (
            self.max_posts is not None
            and self.stats.posts_emitted >= self.max_posts
        ):
            raise BlogBudgetExceeded(
                f"max_posts cap ({self.max_posts}) reached at "
                f"posts_emitted={self.stats.posts_emitted}"
            )
        if (
            self.max_cost_usd is not None
            and self.stats.estimated_cost_usd >= self.max_cost_usd
        ):
            raise BlogBudgetExceeded(
                f"max_cost_usd cap (${self.max_cost_usd:.2f}) reached "
                f"at est_cost=${self.stats.estimated_cost_usd:.4f}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class BlogStats:
    discovery_method: str | None
    posts_discovered: int
    posts_emitted: int
    posts_below_since: int
    cache_hits: int
    cache_misses: int
    fetch_errors: int
    extract_errors: int

    def __init__(self) -> None:
        self.discovery_method = None
        self.posts_discovered = 0
        self.posts_emitted = 0
        self.posts_below_since = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.fetch_errors = 0
        self.extract_errors = 0

    @property
    def estimated_cost_usd(self) -> float:
        # Bandwidth-bound; we don't price requests.
        return 0.0


# ---------------------------------------------------------------------------
# Discovery helpers (module-level so tests can call directly)
# ---------------------------------------------------------------------------

_RSS_HINT = re.compile(r"<rss\b|<feed\b", re.IGNORECASE)
_SITEMAP_HINT = re.compile(r"<urlset\b|<sitemapindex\b", re.IGNORECASE)


def _looks_like_rss(body: str) -> bool:
    head = body[:512]
    if _RSS_HINT.search(head):
        return True
    if _SITEMAP_HINT.search(head):
        return False
    return False


def _parse_rss(body: str) -> list[tuple[str, datetime | None]]:
    """Return [(post_url, pub_date)]. Handles both RSS 2.0 (<item>
    + <link>) and Atom (<entry> + <link href="...">) shapes."""
    out: list[tuple[str, datetime | None]] = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return out

    # RSS 2.0
    for item in root.iter():
        tag = _strip_ns(item.tag).lower()
        if tag != "item":
            continue
        link = _first_text_child(item, ("link",))
        if not link:
            continue
        date_text = _first_text_child(item, ("pubDate", "pubdate"))
        out.append((link.strip(), _parse_loose_date(date_text)))

    # Atom
    for entry in root.iter():
        tag = _strip_ns(entry.tag).lower()
        if tag != "entry":
            continue
        link = None
        for child in entry:
            if _strip_ns(child.tag).lower() == "link":
                link = child.attrib.get("href") or (child.text or "").strip()
                if link:
                    break
        if not link:
            continue
        date_text = _first_text_child(entry, ("published", "updated"))
        out.append((link.strip(), _parse_loose_date(date_text)))

    return out


def _parse_sitemap(body: str) -> list[tuple[str, datetime | None]]:
    out: list[tuple[str, datetime | None]] = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return out

    for url_el in root.iter():
        tag = _strip_ns(url_el.tag).lower()
        if tag != "url":
            continue
        loc = _first_text_child(url_el, ("loc",))
        if not loc:
            continue
        date_text = _first_text_child(url_el, ("lastmod",))
        out.append((loc.strip(), _parse_loose_date(date_text)))

    return out


def _parse_sitemap_index(body: str) -> list[str]:
    """Return the child-sitemap URLs from a `<sitemapindex>` document."""
    out: list[str] = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return out
    for sm in root.iter():
        if _strip_ns(sm.tag).lower() != "sitemap":
            continue
        loc = _first_text_child(sm, ("loc",))
        if loc:
            out.append(loc.strip())
    return out


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _first_text_child(parent: ET.Element, names: tuple[str, ...]) -> str | None:
    targets = {n.lower() for n in names}
    for child in parent:
        if _strip_ns(child.tag).lower() in targets and child.text:
            return child.text
    return None


def _parse_loose_date(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Extraction (trafilatura + handcrafted heading parse)
# ---------------------------------------------------------------------------

def _extract_post(
    html: str,
    *,
    url: str,
    fallback_date: datetime | None = None,
) -> RawItem | None:
    """Run trafilatura over the HTML and produce a RawItem with a
    paragraph-/heading-block list in metadata. Chunker pulls those
    blocks at a later stage to build properly-bounded chunks."""
    import trafilatura  # heavy import; defer

    extracted = trafilatura.extract(
        html,
        url=url,
        favor_precision=True,
        include_comments=False,
        include_tables=False,
        include_formatting=True,   # preserves heading markers in output
        output_format="markdown",
    )
    if not extracted:
        return None
    extracted = extracted.strip()
    if not extracted:
        return None

    title, blocks = _markdown_to_blocks(extracted)

    # Try to pull a publish date from trafilatura's metadata layer.
    metadata = trafilatura.extract_metadata(html, default_url=url)
    date_iso: str = ""
    if metadata is not None and getattr(metadata, "date", None):
        # trafilatura returns 'YYYY-MM-DD' for date.
        try:
            dt = datetime.fromisoformat(metadata.date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            date_iso = dt.isoformat()
        except (TypeError, ValueError):
            date_iso = ""
    if not date_iso and fallback_date is not None:
        date_iso = fallback_date.isoformat()

    if not title and metadata is not None and getattr(metadata, "title", None):
        title = metadata.title.strip()

    body_for_dedup = "\n\n".join(b["text"] for b in blocks if b["text"]).strip()

    return RawItem(
        source_type="blog_post",
        source_url=url,
        date=date_iso,
        title=title or None,
        body=body_for_dedup,
        metadata={
            "domain": urlparse(url).netloc.lower(),
            "blocks": blocks,  # paragraph-aware chunker reads this
        },
    )


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _markdown_to_blocks(md: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse trafilatura's markdown into title + ordered block list.
    Each block is {type: "heading"|"paragraph", level: int, text: str}.
    First H1 is treated as the post title and dropped from the block
    list. Blank lines split paragraphs."""
    title = ""
    blocks: list[dict[str, Any]] = []

    paragraph_buf: list[str] = []

    def _flush_paragraph() -> None:
        nonlocal paragraph_buf
        text = " ".join(line.strip() for line in paragraph_buf if line.strip()).strip()
        if text:
            blocks.append({"type": "paragraph", "level": 0, "text": text})
        paragraph_buf = []

    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        m = _HEADING_RE.match(line)
        if m:
            _flush_paragraph()
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1 and not title and not blocks:
                title = text
                continue
            blocks.append({"type": "heading", "level": level, "text": text})
            continue
        if not line.strip():
            _flush_paragraph()
            continue
        paragraph_buf.append(line)

    _flush_paragraph()
    return title, blocks
