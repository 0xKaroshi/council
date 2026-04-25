"""Twitter / X source — TwitterAPI.io backend.

TwitterAPI.io docs — user timeline endpoint:
    https://docs.twitterapi.io/api-reference/endpoint/get_user_last_tweets
User info (handle → user record):
    https://docs.twitterapi.io/api-reference/endpoint/get_user_info

At current pricing (~$0.15 per 1,000 tweets), a typical 2,000–5,000
tweet archive backfills for under $1. Recommend a separate API key
per deployment so rate limiting or rotation doesn't blast radius
into other projects sharing the same provider.

High-level contract:

    source = TwitterSource(
        handle="paulg",
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        api_key=settings.twitter_api_key,
    )
    async for thread_item in source.fetch():
        # one RawItem per thread, body = concatenated tweet texts

fetch() paginates the timeline newest-first, accumulates tweets into
threads keyed by `conversationId`, and flushes one RawItem per thread
at the end of pagination. We page until either the API runs out of
pages or the oldest tweet we've seen drops below `since`.

Thread assembly happens in this source rather than being deferred to
chunker.chunk_tweets(): we already have to buffer tweets to build
`raw_tweets` / `thread_length` / aggregated engagement counts, so
emitting a pre-joined body is essentially free from here.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from app.ingest import RawItem
from app.ingest.sources.base import Source

log = logging.getLogger(__name__)

_BASE_URL = "https://api.twitterapi.io"
_COST_PER_TWEET_USD = 0.00015  # $0.15 / 1k tweets
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_REQUEST_DELAY_SECONDS = 0.1


@dataclass
class TwitterSourceStats:
    api_calls: int = 0
    tweets_fetched: int = 0          # everything the API returned (we paid for it)
    replies_dropped: int = 0         # filtered out as chatter (reply to non-self)
    tweets_kept: int = 0             # passed reply filter AND since filter
    threads_emitted: int = 0
    last_cursor: str | None = None

    @property
    def estimated_cost_usd(self) -> float:
        return self.tweets_fetched * _COST_PER_TWEET_USD


class TwitterBudgetExceeded(Exception):
    """Raised mid-stream when --max-tweets or --max-cost-usd trips. The
    CLI catches it, prints what was fetched so far, and exits non-zero
    so the user knows to either raise the cap or accept the abort."""


@dataclass
class BatchInfo:
    """Passed to the on_batch callback after each successful page.

    Lets the ingest CLI persist the cursor to the meta table so a
    crash doesn't force a full re-pull.
    """

    cursor: str | None
    total_tweets_fetched: int
    total_tweets_kept: int


OnBatch = Callable[[BatchInfo], Awaitable[None] | None]


class TwitterSource(Source):
    source_type = "twitter"

    def __init__(
        self,
        *,
        handle: str,
        since: datetime,
        api_key: str,
        cursor: str | None = None,
        on_batch: OnBatch | None = None,
        request_delay_seconds: float = _DEFAULT_REQUEST_DELAY_SECONDS,
        max_retries: int = 5,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_tweets: int | None = None,
        max_cost_usd: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("TwitterSource requires a non-empty api_key")
        self.handle = handle.lstrip("@")
        self.since = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        self.api_key = api_key
        self.cursor = cursor
        self.on_batch = on_batch
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.max_tweets = max_tweets
        self.max_cost_usd = max_cost_usd
        # Resolved at fetch() start by _resolve_handle. Used to keep
        # only the mentor's own replies (thread bodies) and drop
        # conversational chatter (replies to other people).
        self.user_id: str | None = None
        # If the caller supplies a client (tests do), we won't close it;
        # otherwise we create + manage one ourselves.
        self._external_client = http_client
        self.stats = TwitterSourceStats()

    # ------------------------------------------------------------------
    # Public fetch — async generator
    # ------------------------------------------------------------------

    async def fetch(self) -> AsyncIterator[RawItem]:
        client, owns_client = self._acquire_client()
        try:
            await self._resolve_handle(client)
            threads: dict[str, list[dict[str, Any]]] = {}

            # Only TwitterBudgetExceeded is shielded. Any other exception
            # (httpx errors, CancelledError, etc.) propagates unchanged
            # — we should not silently yield half-accumulated state on
            # genuinely fatal failures.
            pending_raise: TwitterBudgetExceeded | None = None
            try:
                async for batch in self._paginate_timeline(client):
                    for tw in batch:
                        cid = str(tw.get("conversationId") or tw.get("id") or "")
                        if not cid:
                            continue
                        threads.setdefault(cid, []).append(tw)
            except TwitterBudgetExceeded as exc:
                pending_raise = exc

            for cid, tweets in threads.items():
                item = self._thread_to_raw_item(tweets)
                if item is None:
                    continue
                self.stats.threads_emitted += 1
                yield item

            log.info(
                "twitter: done. api_calls=%d tweets_fetched=%d tweets_kept=%d "
                "threads=%d est_cost=$%.4f",
                self.stats.api_calls,
                self.stats.tweets_fetched,
                self.stats.tweets_kept,
                self.stats.threads_emitted,
                self.stats.estimated_cost_usd,
            )

            if pending_raise is not None:
                raise pending_raise
        finally:
            if owns_client:
                await client.aclose()

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _acquire_client(self) -> tuple[httpx.AsyncClient, bool]:
        if self._external_client is not None:
            return self._external_client, False
        client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "x-api-key": self.api_key,
                "accept": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        return client, True

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await client.request(method, url, **kwargs)
            except httpx.HTTPError as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
                delay = min(30.0, 2 ** attempt)
                log.warning(
                    "twitter: HTTP error on %s %s: %s (retry %d/%d in %.1fs)",
                    method, url, e, attempt + 1, self.max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self.max_retries:
                    resp.raise_for_status()
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                delay = retry_after if retry_after > 0 else min(30.0, 2 ** attempt)
                log.warning(
                    "twitter: status %d on %s %s (retry %d/%d in %.1fs)",
                    resp.status_code, method, url, attempt + 1, self.max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue

            resp.raise_for_status()
            return resp

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Handle resolution
    # ------------------------------------------------------------------

    async def _resolve_handle(self, client: httpx.AsyncClient) -> dict[str, Any]:
        resp = await self._request_with_retry(
            client,
            "GET",
            "/twitter/user/info",
            params={"userName": self.handle},
        )
        self.stats.api_calls += 1
        body = resp.json()
        # /twitter/user/info returns the user object under "data" in live
        # responses; fall back to the top-level body in case the docs-
        # described flat shape ever ships.
        inner = body.get("data") if isinstance(body, dict) else None
        user = inner if isinstance(inner, dict) else (body if isinstance(body, dict) else {})
        uid = user.get("id") or user.get("userId")
        if uid:
            self.user_id = str(uid)
        log.info("twitter: resolved @%s → user_id=%s", self.handle, self.user_id)
        return user

    # ------------------------------------------------------------------
    # Timeline pagination
    # ------------------------------------------------------------------

    async def _paginate_timeline(
        self, client: httpx.AsyncClient
    ) -> AsyncIterator[list[dict[str, Any]]]:
        while True:
            # `includeReplies=true` is required to capture reply-based
            # threads — the endpoint defaults to replies-excluded and
            # would otherwise drop ~half the meaningful content for
            # mentors who post most of their long-form ideas as
            # threads.
            params: dict[str, str] = {
                "userName": self.handle,
                "includeReplies": "true",
            }
            if self.cursor:
                params["cursor"] = self.cursor

            resp = await self._request_with_retry(
                client,
                "GET",
                "/twitter/user/last_tweets",
                params=params,
            )
            self.stats.api_calls += 1

            # Live /last_tweets response shape:
            #   { status, msg, code, data: { pin_tweet, tweets }, has_next_page, next_cursor }
            # `tweets` is under data; the pagination fields are AT THE TOP
            # LEVEL. An earlier helper tried to unwrap everything through
            # `data` and silently killed pagination after page 1.
            body: dict[str, Any] = resp.json() or {}
            data_block = body.get("data") if isinstance(body.get("data"), dict) else {}
            raw_tweets: list[dict[str, Any]] = (
                data_block.get("tweets")
                or body.get("tweets")  # tolerate the docs-flat shape
                or []
            )
            has_next: bool = bool(body.get("has_next_page"))
            next_cursor: str | None = body.get("next_cursor") or None

            if not raw_tweets and not has_next:
                break

            # Reply filter — drop conversational chatter (replies to
            # other users) before the date check. Keeps:
            #   - originals (inReplyToUserId is None), AND
            #   - self-replies (inReplyToUserId == self.user_id) — these
            #     form the bodies of the mentor's own threads.
            # Skipped if user_id resolution failed (rare); in that case
            # we keep everything to avoid silent over-filtering.
            self_uid = self.user_id
            filtered_tweets: list[dict[str, Any]] = []
            batch_replies_dropped = 0
            for tw in raw_tweets:
                reply_uid = tw.get("inReplyToUserId")
                if (
                    reply_uid is not None
                    and self_uid is not None
                    and str(reply_uid) != self_uid
                ):
                    batch_replies_dropped += 1
                    continue
                filtered_tweets.append(tw)

            kept: list[dict[str, Any]] = []
            oldest_dt: datetime | None = None
            for tw in filtered_tweets:
                dt = _parse_tweet_date(tw.get("createdAt"))
                if dt is None:
                    continue
                oldest_dt = dt if oldest_dt is None or dt < oldest_dt else oldest_dt
                if dt >= self.since:
                    kept.append(tw)

            self.stats.tweets_fetched += len(raw_tweets)
            self.stats.replies_dropped += batch_replies_dropped
            self.stats.tweets_kept += len(kept)
            log.info(
                "twitter: batch fetched=%d kept=%d cumulative=%d "
                "oldest=%s est_cost=$%.4f",
                len(raw_tweets),
                len(kept),
                self.stats.tweets_fetched,
                oldest_dt.strftime("%Y-%m-%d") if oldest_dt else "n/a",
                self.stats.estimated_cost_usd,
            )

            if kept:
                yield kept

            self.cursor = next_cursor
            self.stats.last_cursor = next_cursor

            if self.on_batch is not None:
                result = self.on_batch(
                    BatchInfo(
                        cursor=next_cursor,
                        total_tweets_fetched=self.stats.tweets_fetched,
                        total_tweets_kept=self.stats.tweets_kept,
                    )
                )
                if asyncio.iscoroutine(result):
                    await result

            self._enforce_caps()

            # Stop if we're past the cutoff or the API has no more pages.
            if oldest_dt is not None and oldest_dt < self.since:
                break
            if not has_next or not next_cursor:
                break

            if self.request_delay_seconds > 0:
                await asyncio.sleep(self.request_delay_seconds)

    def _enforce_caps(self) -> None:
        if (
            self.max_tweets is not None
            and self.stats.tweets_fetched >= self.max_tweets
        ):
            raise TwitterBudgetExceeded(
                f"max_tweets cap ({self.max_tweets}) reached at "
                f"tweets_fetched={self.stats.tweets_fetched}"
            )
        if (
            self.max_cost_usd is not None
            and self.stats.estimated_cost_usd >= self.max_cost_usd
        ):
            raise TwitterBudgetExceeded(
                f"max_cost_usd cap (${self.max_cost_usd:.2f}) reached at "
                f"est_cost=${self.stats.estimated_cost_usd:.4f}"
            )

    # ------------------------------------------------------------------
    # Thread → RawItem
    # ------------------------------------------------------------------

    def _thread_to_raw_item(self, tweets: list[dict[str, Any]]) -> RawItem | None:
        if not tweets:
            return None

        dated: list[tuple[datetime, dict[str, Any]]] = []
        for tw in tweets:
            dt = _parse_tweet_date(tw.get("createdAt"))
            if dt is None:
                continue
            dated.append((dt, tw))
        if not dated:
            return None

        dated.sort(key=lambda pair: pair[0])
        ordered = [tw for _, tw in dated]
        root_dt, root = dated[0]
        root_id = str(root.get("id") or "")
        if not root_id:
            return None

        body = "\n\n".join(
            (tw.get("text") or "").strip()
            for tw in ordered
            if (tw.get("text") or "").strip()
        )
        if not body:
            return None

        like_total = sum(int(tw.get("likeCount") or 0) for tw in ordered)
        rt_total = sum(int(tw.get("retweetCount") or 0) for tw in ordered)
        is_reply = bool(root.get("isReply") or root.get("inReplyToId"))
        is_quote = bool(root.get("quotedTweet") or root.get("isQuote"))
        reply_to_user = root.get("inReplyToUsername") or root.get("inReplyToUserId")

        return RawItem(
            source_type="twitter",
            source_url=f"https://x.com/{self.handle}/status/{root_id}",
            date=root_dt.isoformat(),
            title=None,
            body=body,
            metadata={
                "conversation_id": str(root.get("conversationId") or root_id),
                "root_tweet_id": root_id,
                "thread_length": len(ordered),
                "like_count_total": like_total,
                "retweet_count_total": rt_total,
                "is_reply": is_reply,
                "is_quote": is_quote,
                "reply_to_user": reply_to_user,
                "raw_tweets": ordered,
            },
        )


# ---------------------------------------------------------------------------
# Helpers (module-level so tests can exercise them directly)
# ---------------------------------------------------------------------------

def _parse_tweet_date(s: str | None) -> datetime | None:
    """Accept Twitter's legacy "Mon Jan 01 00:00:00 +0000 2024" format
    and ISO 8601 as a fallback. Always returns tz-aware UTC."""
    if not s:
        return None
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


def _parse_retry_after(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0
