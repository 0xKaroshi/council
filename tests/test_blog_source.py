"""Unit tests for app.ingest.sources.blog.

No real network. The blog source's `http_client` is injected so
tests pass an `httpx.AsyncClient` backed by `httpx.MockTransport`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from app.ingest import RawItem
from app.ingest.chunker import chunk_blog_paragraphs
from app.ingest.sources.blog import (
    BlogSource,
    _parse_rss,
    _parse_sitemap,
)

_RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Blog</title>
    <item>
      <title>Post One</title>
      <link>https://example.test/posts/one/</link>
      <pubDate>Mon, 12 Feb 2024 08:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Post Two</title>
      <link>https://example.test/posts/two/</link>
      <pubDate>Tue, 20 Aug 2024 14:30:00 +0000</pubDate>
    </item>
    <item>
      <title>Old Post</title>
      <link>https://example.test/posts/old/</link>
      <pubDate>Sat, 02 Jan 2010 00:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""

_SITEMAP_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.test/posts/alpha/</loc>
    <lastmod>2024-03-15</lastmod>
  </url>
  <url>
    <loc>https://example.test/posts/beta/</loc>
    <lastmod>2024-04-10T12:00:00Z</lastmod>
  </url>
</urlset>"""


# Minimal HTML that trafilatura can extract a title + body from.
def _post_html(title: str, body_paragraphs: list[str], h2: str | None = None) -> str:
    paras = "\n".join(f"<p>{p}</p>" for p in body_paragraphs)
    h2_html = f"<h2>{h2}</h2>" if h2 else ""
    return f"""<!DOCTYPE html>
<html><head><title>{title}</title></head>
<body>
  <article>
    <h1>{title}</h1>
    <time datetime="2024-02-12">February 12, 2024</time>
    {h2_html}
    {paras}
  </article>
</body></html>"""


# ---------------------------------------------------------------------------
# Test 1 — RSS parser extracts URL + date pairs
# ---------------------------------------------------------------------------


def test_rss_parser_extracts_links_and_dates() -> None:
    """Pure parser test — no network. Both pubDate-style and missing-
    date items return the right shape; sitemap parser handles both
    plain-date and ISO-with-tz lastmod."""
    rss = _parse_rss(_RSS_FIXTURE)
    urls = [u for u, _ in rss]
    assert urls == [
        "https://example.test/posts/one/",
        "https://example.test/posts/two/",
        "https://example.test/posts/old/",
    ]
    # Each parsed date is tz-aware UTC.
    for _, dt in rss:
        assert dt is not None
        assert dt.tzinfo is not None

    sm = _parse_sitemap(_SITEMAP_FIXTURE)
    sm_urls = [u for u, _ in sm]
    assert sm_urls == [
        "https://example.test/posts/alpha/",
        "https://example.test/posts/beta/",
    ]
    # ISO-without-tz still gets coerced to UTC.
    assert all(dt is not None and dt.tzinfo is not None for _, dt in sm)


# ---------------------------------------------------------------------------
# Test 2 — trafilatura extraction yields title + paragraph blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extraction_produces_title_and_blocks(tmp_path: Path) -> None:
    """Spin up a BlogSource backed by a MockTransport that serves
    one /index.xml plus one HTML post. Verify the emitted RawItem
    has a title, ISO date, and a `blocks` metadata list with at
    least the post's paragraphs surfaced."""
    rss = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item><link>https://example.test/posts/p1/</link>
            <pubDate>Mon, 12 Feb 2024 08:00:00 +0000</pubDate></item>
    </channel></rss>"""
    html = _post_html(
        "How to Price Your SaaS",
        body_paragraphs=[
            "Customer acquisition cost matters more than you think.",
            "Lifetime value is the actual answer to most pricing questions.",
        ],
        h2="The numbers",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/index.xml"):
            return httpx.Response(200, text=rss, headers={"content-type": "application/xml"})
        if url == "https://example.test/posts/p1/":
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    source = BlogSource(
        domain="example.test",
        since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        cache_dir=tmp_path / "cache",
        request_delay_seconds=0.0,
        request_jitter_seconds=0.0,
        http_client=client,
    )

    items = [item async for item in source.fetch()]
    await client.aclose()

    assert len(items) == 1
    item = items[0]
    assert item.source_type == "blog_post"
    assert item.source_url == "https://example.test/posts/p1/"
    assert item.title == "How to Price Your SaaS"
    assert item.date.startswith("2024-02-12")

    blocks = item.metadata["blocks"]
    block_types = [b["type"] for b in blocks]
    block_texts = [b["text"] for b in blocks]
    # An H2 should land as a heading block; the two paragraphs as
    # paragraph blocks.
    assert "heading" in block_types
    assert any("Customer acquisition cost" in t for t in block_texts)
    assert any("Lifetime value" in t for t in block_texts)


# ---------------------------------------------------------------------------
# Test 3 — paragraph chunker prepends "{title} > {h2}: " and respects target
# ---------------------------------------------------------------------------


def test_chunk_blog_paragraphs_prefixes_and_segments() -> None:
    """Hand-build a RawItem with metadata['blocks']: 1 H2 followed by
    several long paragraphs. With a tiny target_tokens the chunker
    must emit multiple chunks; each chunk text starts with the
    "{title} > {h2}: " prefix and respects paragraph boundaries (no
    chunk text begins mid-sentence)."""
    paragraphs = [
        "Pricing is positioning. " * 30,  # ~120 tokens
        "Retention is the only metric that matters at scale. " * 30,
        "ICP work is what survives the next platform shift. " * 30,
        "Test the offer before building the product. " * 30,
    ]
    blocks = [{"type": "heading", "level": 2, "text": "Operator economics"}]
    blocks += [{"type": "paragraph", "level": 0, "text": p} for p in paragraphs]

    item = RawItem(
        source_type="blog_post",
        source_url="https://example.test/posts/x/",
        date="2024-02-12T00:00:00+00:00",
        title="Test Title",
        body="\n\n".join(paragraphs),
        metadata={"blocks": blocks, "domain": "example.test"},
    )

    chunks = chunk_blog_paragraphs(item, "testmentor", target_tokens=300, overlap_tokens=50)

    # >1 chunk because the body exceeds target_tokens.
    assert len(chunks) >= 2

    # Every chunk carries the title + section prefix and starts
    # exactly with it.
    for c in chunks:
        assert c.text.startswith("Test Title > Operator economics: ")
        # The substring after the prefix must begin at the start of
        # one of the paragraphs (paragraph-boundary respect).
        body = c.text[len("Test Title > Operator economics: ") :]
        assert any(body.startswith(p[:30]) for p in paragraphs), (
            f"chunk does not begin at a paragraph boundary: {body[:60]!r}"
        )

    # source_priority is canonical (3) by default for blog chunker.
    assert all(c.source_priority == 3 for c in chunks)


# ---------------------------------------------------------------------------
# Test 4 — cache hits avoid the network on the second fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_fetch_hits_cache_and_skips_network(tmp_path: Path) -> None:
    """First run discovers + fetches; second run with the same
    cache_dir (and refresh=False) re-discovers but every per-post
    fetch is a cache hit. Counter on the mock transport proves it."""
    rss = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item><link>https://example.test/posts/p1/</link>
            <pubDate>Mon, 12 Feb 2024 08:00:00 +0000</pubDate></item>
      <item><link>https://example.test/posts/p2/</link>
            <pubDate>Tue, 13 Feb 2024 08:00:00 +0000</pubDate></item>
    </channel></rss>"""

    def html_for(u):
        return _post_html(f"Post at {u}", ["Body paragraph for the test."])

    post_fetch_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/index.xml"):
            return httpx.Response(200, text=rss, headers={"content-type": "application/xml"})
        if "/posts/" in url:
            post_fetch_count["n"] += 1
            return httpx.Response(200, text=html_for(url), headers={"content-type": "text/html"})
        return httpx.Response(404)

    cache_dir = tmp_path / "cache"

    # First run.
    client1 = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    s1 = BlogSource(
        domain="example.test",
        since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        cache_dir=cache_dir,
        request_delay_seconds=0.0,
        request_jitter_seconds=0.0,
        http_client=client1,
    )
    items1 = [i async for i in s1.fetch()]
    await client1.aclose()
    assert len(items1) == 2
    assert post_fetch_count["n"] == 2
    assert s1.stats.cache_misses == 2
    assert s1.stats.cache_hits == 0

    # Second run, fresh source, same cache_dir.
    client2 = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    s2 = BlogSource(
        domain="example.test",
        since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        cache_dir=cache_dir,
        request_delay_seconds=0.0,
        request_jitter_seconds=0.0,
        http_client=client2,
    )
    items2 = [i async for i in s2.fetch()]
    await client2.aclose()
    assert len(items2) == 2
    # No new per-post fetches — only the discovery /index.xml.
    assert post_fetch_count["n"] == 2
    assert s2.stats.cache_hits == 2
    assert s2.stats.cache_misses == 0


# ---------------------------------------------------------------------------
# Test 5 — content_hash dedup across two domains hosting the same essay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_substack_sparse_rss_unions_with_sitemap(tmp_path: Path) -> None:
    """Substack /feed truncates to ~20 most recent posts but
    /sitemap.xml exposes the full archive. The discovery layer's
    sparse-RSS-fallback unions URLs from /feed with later probes
    until the threshold (30) is crossed.

    Mock /feed → 5 items, /sitemap.xml → 50 items (overlap of 5).
    Result must be 50 unique URLs and discovery_method must list
    both probes."""
    feed_items = "\n".join(
        f"<item><link>https://sub.test/p/feed-only-{i}/</link>"
        f"<pubDate>Mon, 01 Apr 2024 00:00:00 +0000</pubDate></item>"
        for i in range(5)
    )
    rss = f"""<?xml version="1.0"?><rss version="2.0"><channel>
      {feed_items}
    </channel></rss>"""

    sitemap_urls = []
    # 5 URLs that overlap with the RSS feed + 45 unique ones.
    for i in range(5):
        sitemap_urls.append(
            f"<url><loc>https://sub.test/p/feed-only-{i}/</loc><lastmod>2024-04-01</lastmod></url>"
        )
    for i in range(45):
        sitemap_urls.append(
            f"<url><loc>https://sub.test/p/sitemap-only-{i}/</loc>"
            f"<lastmod>2024-03-15</lastmod></url>"
        )
    sitemap = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(sitemap_urls)
        + "</urlset>"
    )

    fetched_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        fetched_paths.append(path)
        if path in ("/feed", "/feed/"):
            return httpx.Response(
                200,
                text=rss,
                headers={"content-type": "application/xml"},
            )
        if path == "/sitemap.xml":
            return httpx.Response(
                200,
                text=sitemap,
                headers={"content-type": "application/xml"},
            )
        # Per-post fetches stub out — we only care about discovery here.
        if path.startswith("/p/"):
            return httpx.Response(
                200,
                text="<html><body><article><h1>x</h1><p>x</p></article></body></html>",
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=True)

    # Use a since cutoff that excludes everything so we don't burn time
    # extracting the 50 stub posts — discovery is what we're testing.
    source = BlogSource(
        domain="sub.test",
        since=datetime(2030, 1, 1, tzinfo=timezone.utc),
        cache_dir=tmp_path / "cache",
        request_delay_seconds=0.0,
        request_jitter_seconds=0.0,
        http_client=client,
    )

    [_ async for _ in source.fetch()]
    await client.aclose()

    # /feed (no slash) hit first, gave 5; /sitemap.xml then hit, gave 50;
    # union = 50 distinct URLs. The discovery_method label should mention
    # both probes that contributed.
    assert source.stats.posts_discovered == 50
    assert "/feed" in source.stats.discovery_method
    assert "/sitemap.xml" in source.stats.discovery_method

    # Both endpoints actually requested during discovery.
    assert "/feed" in fetched_paths
    assert "/sitemap.xml" in fetched_paths


@pytest.mark.asyncio
async def test_substack_post_extraction_from_fixture(tmp_path: Path) -> None:
    """A representative Substack post HTML — header h1, byline,
    multiple body paragraphs, footer subscribe widget — should
    extract via trafilatura into a RawItem with a non-empty title
    and at least 2 paragraph blocks. The footer widget shouldn't
    leak into the body."""
    # Substack post page shape — simplified but realistic. trafilatura
    # uses readability heuristics; <article> + <h1> + body <p>s extract
    # cleanly.
    html = """<!DOCTYPE html>
<html><head>
<title>Build Once Sell Twice — Visualize Value</title>
<meta name="description" content="A primer on leveraging design.">
<meta property="article:published_time" content="2024-08-15T12:00:00Z">
</head><body>
<header>SUBSCRIBE NOW · NAV · MENU</header>
<article>
  <h1>Build Once, Sell Twice</h1>
  <div class="byline">Test Author · Aug 15, 2024</div>
  <p>The simplest leverage move in modern business is to package
  what you know once and sell it many times. The internet collapsed
  the cost of distribution to zero — most operators haven't updated
  their mental model.</p>
  <p>Service businesses trade time for money. Product businesses
  trade leverage for money. The question isn't whether to build a
  product — it's how to design one that emerges naturally from the
  service work you already do.</p>
  <p>Personal monopoly is the destination. The path is articulating
  what you uniquely see, then productizing the artifact.</p>
</article>
<footer>
  <div class="subscribe-widget">Get future essays in your inbox</div>
  <div class="footer-links">About · Archive · Contact</div>
</footer>
</body></html>"""

    rss = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item><link>https://sub.test/p/build-once-sell-twice/</link>
            <pubDate>Thu, 15 Aug 2024 12:00:00 +0000</pubDate></item>
    </channel></rss>"""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in ("/feed", "/feed/", "/index.xml"):
            return httpx.Response(
                200,
                text=rss,
                headers={"content-type": "application/xml"},
            )
        if path == "/p/build-once-sell-twice/":
            return httpx.Response(
                200,
                text=html,
                headers={"content-type": "text/html"},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    source = BlogSource(
        domain="sub.test",
        since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        cache_dir=tmp_path / "cache",
        request_delay_seconds=0.0,
        request_jitter_seconds=0.0,
        http_client=client,
    )

    items = [item async for item in source.fetch()]
    await client.aclose()

    assert len(items) == 1
    item = items[0]
    assert item.source_type == "blog_post"
    assert item.title == "Build Once, Sell Twice"
    assert item.date.startswith("2024-08-15")

    # Body paragraphs extracted; subscribe widget did NOT leak in.
    paragraphs = [b["text"] for b in item.metadata["blocks"] if b["type"] == "paragraph"]
    assert len(paragraphs) >= 2
    body = " ".join(paragraphs)
    assert "leverage" in body.lower()
    assert "personal monopoly" in body.lower()
    assert "subscribe" not in body.lower()
    assert "subscribe-widget" not in body.lower()


def test_content_hash_dedup_across_domains() -> None:
    """Same essay text under two different URLs / domains hashes to
    the same content_hash so upsert_chunks's UNIQUE constraint
    collapses them into one row. We don't need a DB to verify —
    just that the chunker assigns identical hashes."""
    paragraphs = [
        "Bootstrapped SaaS economics start with a clean ICP. " * 10,
        "Retention is operator math, not a marketing problem. " * 10,
    ]

    def make(item_url: str, domain: str) -> RawItem:
        blocks = [{"type": "paragraph", "level": 0, "text": p} for p in paragraphs]
        return RawItem(
            source_type="blog_post",
            source_url=item_url,
            date="2024-01-01T00:00:00+00:00",
            title="The Same Essay",
            body="\n\n".join(paragraphs),
            metadata={"blocks": blocks, "domain": domain},
        )

    chunks_a = chunk_blog_paragraphs(
        make("https://blog.example.test/post-x/", "blog.example.test"),
        "testmentor",
        target_tokens=400,
    )
    chunks_b = chunk_blog_paragraphs(
        make("https://longform.example.test/republished/post-x/", "longform.example.test"),
        "testmentor",
        target_tokens=400,
    )

    hashes_a = {c.content_hash for c in chunks_a}
    hashes_b = {c.content_hash for c in chunks_b}

    # Identical text → identical content_hash sets, so upsert with
    # IGNORE-on-conflict collapses them to a single DB row each.
    assert hashes_a == hashes_b
    assert len(hashes_a) == len(chunks_a)  # within a single post, hashes are distinct
