"""Unit tests for TwitterSource.

No real network. The httpx.AsyncClient is injected via the `http_client`
constructor param (the production code path creates and owns its own
client; tests hand in an AsyncMock). Responses are a hand-rolled
FakeResponse class so we don't pull in respx just for three tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.ingest.sources.twitter import TwitterBudgetExceeded, TwitterSource


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"test response was {self.status_code}, not OK")


def _user_info_response(user_id: str = "u1") -> FakeResponse:
    return FakeResponse(payload={"data": {"id": user_id, "userName": "test_handle"}})


def _tweet(
    tid: str,
    *,
    conv: str,
    created: str,
    text: str,
    likes: int = 0,
    rts: int = 0,
    is_reply: bool = False,
    in_reply_to_user: str | None = None,
    in_reply_to_user_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": tid,
        "conversationId": conv,
        "createdAt": created,
        "text": text,
        "likeCount": likes,
        "retweetCount": rts,
        "isReply": is_reply,
        "inReplyToUsername": in_reply_to_user,
        "inReplyToUserId": in_reply_to_user_id,
    }


def _timeline_response(
    tweets: list[dict[str, Any]],
    *,
    has_next: bool = False,
    next_cursor: str | None = None,
) -> FakeResponse:
    """Real /twitter/user/last_tweets response shape: `tweets` live under
    `data`, but `has_next_page` / `next_cursor` are at the TOP LEVEL."""
    return FakeResponse(
        payload={
            "status": "success",
            "msg": "success",
            "data": {"pin_tweet": None, "tweets": tweets},
            "has_next_page": has_next,
            "next_cursor": next_cursor,
        }
    )


def _make_source(
    client: AsyncMock,
    *,
    since: datetime | None = None,
    cursor: str | None = None,
) -> TwitterSource:
    return TwitterSource(
        handle="test_handle",
        since=since or datetime(2023, 1, 1, tzinfo=timezone.utc),
        api_key="sk-test",
        cursor=cursor,
        request_delay_seconds=0.0,
        http_client=client,
    )


# ---------------------------------------------------------------------------
# Test 1 — thread reassembly groups by conversation_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thread_reassembly_groups_by_conversation_id() -> None:
    """Two conversations spread across two pages should produce two
    RawItems, each carrying exactly the tweets that belong to it and
    ordered chronologically within the thread."""
    page1 = _timeline_response(
        [
            _tweet(
                "101", conv="100", created="Mon Mar 04 10:00:00 +0000 2024", text="root of thread A"
            ),
            _tweet(
                "201", conv="200", created="Mon Mar 04 11:00:00 +0000 2024", text="standalone B"
            ),
        ],
        has_next=True,
        next_cursor="cursor-page-2",
    )
    page2 = _timeline_response(
        [
            _tweet(
                "102",
                conv="100",
                created="Mon Mar 04 10:05:00 +0000 2024",
                text="reply 1 in thread A",
            ),
            _tweet(
                "103",
                conv="100",
                created="Mon Mar 04 10:07:00 +0000 2024",
                text="reply 2 in thread A",
            ),
        ],
        has_next=False,
        next_cursor=None,
    )

    client = AsyncMock()
    client.request.side_effect = [
        _user_info_response(),
        page1,
        page2,
    ]

    source = _make_source(client)
    items = [item async for item in source.fetch()]

    assert len(items) == 2
    by_conv = {item.metadata["conversation_id"]: item for item in items}

    thread_a = by_conv["100"]
    assert thread_a.metadata["thread_length"] == 3
    assert thread_a.body.splitlines()[::2] == [  # every other line = text lines
        "root of thread A",
        "reply 1 in thread A",
        "reply 2 in thread A",
    ]
    assert thread_a.metadata["root_tweet_id"] == "101"
    assert thread_a.source_url == "https://x.com/test_handle/status/101"

    thread_b = by_conv["200"]
    assert thread_b.metadata["thread_length"] == 1
    assert thread_b.body == "standalone B"

    # Exactly 3 HTTP calls: user info + 2 timeline pages.
    assert client.request.call_count == 3


# ---------------------------------------------------------------------------
# Test 2 — RawItem construction preserves metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_item_preserves_metadata_and_aggregates() -> None:
    """Engagement counts sum across the thread; root-level flags
    (is_reply, reply_to_user) come from the root tweet; raw_tweets
    carries the full API blobs in chronological order."""
    reply_root = _tweet(
        "1001",
        conv="1000",
        created="Tue Apr 09 12:00:00 +0000 2024",
        text="hello world",
        likes=10,
        rts=2,
        is_reply=True,
        in_reply_to_user="someone_else",
    )
    reply_child = _tweet(
        "1002",
        conv="1000",
        created="Tue Apr 09 12:01:00 +0000 2024",
        text="follow-up",
        likes=5,
        rts=1,
    )

    client = AsyncMock()
    client.request.side_effect = [
        _user_info_response(),
        _timeline_response([reply_root, reply_child], has_next=False),
    ]

    source = _make_source(client)
    items = [item async for item in source.fetch()]

    assert len(items) == 1
    item = items[0]

    assert item.source_type == "twitter"
    assert item.title is None
    assert item.source_url == "https://x.com/test_handle/status/1001"
    assert item.date.startswith("2024-04-09")

    md = item.metadata
    assert md["conversation_id"] == "1000"
    assert md["root_tweet_id"] == "1001"
    assert md["thread_length"] == 2
    assert md["like_count_total"] == 15
    assert md["retweet_count_total"] == 3
    assert md["is_reply"] is True
    assert md["is_quote"] is False
    assert md["reply_to_user"] == "someone_else"

    raw = md["raw_tweets"]
    assert [t["id"] for t in raw] == ["1001", "1002"], "raw_tweets must be chronological"
    assert raw[0] is reply_root  # identity preserved — we don't copy


# ---------------------------------------------------------------------------
# Test 3 — since filter stops iteration at cutoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_since_cutoff_stops_pagination() -> None:
    """Timeline returns newest-first. Once the oldest tweet in a batch
    is older than `since`, we keep only the ones that passed the filter
    and then stop — even if has_next_page is still True."""
    page1 = _timeline_response(
        [
            _tweet(
                "301", conv="300", created="Fri Jun 07 10:00:00 +0000 2024", text="recent and kept"
            ),
            _tweet(
                "302",
                conv="302",
                created="Sat Dec 31 23:59:00 +0000 2022",
                text="old — below cutoff",
            ),
        ],
        has_next=True,
        next_cursor="would-not-be-fetched",
    )

    client = AsyncMock()
    client.request.side_effect = [
        _user_info_response(),
        page1,
        # No page 2 wired on purpose — if it's ever fetched, AsyncMock
        # returns a default MagicMock and the test blows up accessing
        # .json(), failing loudly rather than silently.
    ]

    since = datetime(2023, 1, 1, tzinfo=timezone.utc)
    source = _make_source(client, since=since)
    items = [item async for item in source.fetch()]

    # Only the kept tweet becomes a thread.
    assert len(items) == 1
    assert items[0].metadata["root_tweet_id"] == "301"

    # Two HTTP calls total: user info + one timeline page. Pagination
    # halted because oldest in batch was below cutoff.
    assert client.request.call_count == 2

    # Stats reflect both tweets seen but only one kept post-filter.
    assert source.stats.tweets_fetched == 2
    assert source.stats.tweets_kept == 1
    assert source.stats.threads_emitted == 1


# ---------------------------------------------------------------------------
# Test 4 — pagination follows top-level has_next_page (regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pagination_follows_top_level_has_next_page() -> None:
    """Regression for the bug that capped the dry-run path at 20 tweets.

    `/twitter/user/last_tweets` returns `has_next_page` and `next_cursor`
    at the TOP LEVEL of the response (not inside `data`). An earlier
    parser unconditionally unwrapped `body["data"]` and then looked for
    pagination fields there — finding none — so the loop terminated
    after page 1. If this test starts failing, the same bug class is
    back.
    """
    # Manually construct responses so we can assert specifically that
    # has_next_page / next_cursor are ONLY at the top level, never
    # mirrored inside `data`. If the buggy code ever returns, it would
    # read None from `data.has_next_page` and stop after page 1 —
    # making client.request.call_count == 2 instead of 3.
    page1 = FakeResponse(
        payload={
            "status": "success",
            "msg": "success",
            "data": {
                "pin_tweet": None,
                "tweets": [
                    _tweet(
                        "101",
                        conv="101",
                        created="Mon Mar 04 10:00:00 +0000 2024",
                        text="page 1 only",
                    ),
                ],
            },
            "has_next_page": True,
            "next_cursor": "cursor-for-page-2",
        }
    )
    page2 = FakeResponse(
        payload={
            "status": "success",
            "msg": "success",
            "data": {
                "pin_tweet": None,
                "tweets": [
                    _tweet(
                        "202",
                        conv="202",
                        created="Mon Mar 04 11:00:00 +0000 2024",
                        text="page 2 only",
                    ),
                ],
            },
            "has_next_page": False,
            "next_cursor": None,
        }
    )

    client = AsyncMock()
    client.request.side_effect = [
        _user_info_response(),
        page1,
        page2,
    ]

    source = _make_source(client)
    items = [item async for item in source.fetch()]

    # user info + page 1 + page 2 = 3 HTTP calls. If this is 2, the
    # pagination-field parser regressed.
    assert client.request.call_count == 3

    # One thread per page (distinct conversation_ids) → 2 RawItems.
    assert len(items) == 2

    # Page 2 must have been called with the cursor from page 1.
    page2_call = client.request.call_args_list[2]
    assert page2_call.kwargs["params"]["cursor"] == "cursor-for-page-2"
    # And includeReplies must be on every timeline call.
    assert page2_call.kwargs["params"]["includeReplies"] == "true"


# ---------------------------------------------------------------------------
# Test 5 — reply filter keeps originals + self-replies, drops chatter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_filter_keeps_originals_and_self_replies() -> None:
    """`includeReplies=true` floods the response with one-line replies
    the mentor sent to other people. Those are noise. Keep:
       - originals (no inReplyToUserId), and
       - self-replies (inReplyToUserId == own user_id) — the bodies of
         his own threads.
    Drop everything else.
    """
    own_user_id = "u1"  # _user_info_response defaults to user_id="u1"

    original = _tweet(
        "100",
        conv="100",
        created="Mon Mar 04 10:00:00 +0000 2024",
        text="original tweet",
    )
    self_reply = _tweet(
        "101",
        conv="100",
        created="Mon Mar 04 10:05:00 +0000 2024",
        text="self-reply continuing the thread",
        is_reply=True,
        in_reply_to_user_id=own_user_id,
    )
    other_reply = _tweet(
        "102",
        conv="999",
        created="Mon Mar 04 10:10:00 +0000 2024",
        text="reply to someone else's tweet",
        is_reply=True,
        in_reply_to_user_id="some-other-uid",
    )

    client = AsyncMock()
    client.request.side_effect = [
        _user_info_response(user_id=own_user_id),
        _timeline_response([original, self_reply, other_reply], has_next=False),
    ]

    source = _make_source(client)
    items = [item async for item in source.fetch()]

    # Two surviving tweets share conversationId="100" → one thread.
    # The other_reply (conv=999) was dropped before threading, so no
    # phantom second thread.
    assert len(items) == 1
    thread = items[0]
    assert thread.metadata["conversation_id"] == "100"
    assert thread.metadata["thread_length"] == 2

    # Stats reflect the API gave us 3 (we paid for 3), 1 was dropped
    # by the reply filter, and 2 passed both filters.
    assert source.stats.tweets_fetched == 3
    assert source.stats.replies_dropped == 1
    assert source.stats.tweets_kept == 2


# ---------------------------------------------------------------------------
# Test 6 — budget cap flushes accumulated threads before re-raising
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_cap_flushes_accumulated_threads_before_reraising() -> None:
    """When --max-tweets (or --max-cost-usd) trips mid-stream, the
    already-accumulated threads must be yielded to the caller BEFORE
    TwitterBudgetExceeded surfaces. Otherwise an aborted run loses
    every tweet it paid for.

    Setup: three pages × two tweets, spread across two conversations.
    max_tweets=6 trips on page 3 (after tweets_fetched reaches 6).
    The caller should receive both threads (3 tweets each) and THEN
    observe TwitterBudgetExceeded.
    """
    b1 = _timeline_response(
        [
            _tweet("1a", conv="a", created="Mon Mar 04 10:00:00 +0000 2024", text="a1"),
            _tweet("1b", conv="b", created="Mon Mar 04 10:00:01 +0000 2024", text="b1"),
        ],
        has_next=True,
        next_cursor="c2",
    )
    b2 = _timeline_response(
        [
            _tweet("2a", conv="a", created="Mon Mar 04 10:01:00 +0000 2024", text="a2"),
            _tweet("2b", conv="b", created="Mon Mar 04 10:01:01 +0000 2024", text="b2"),
        ],
        has_next=True,
        next_cursor="c3",
    )
    b3 = _timeline_response(
        [
            _tweet("3a", conv="a", created="Mon Mar 04 10:02:00 +0000 2024", text="a3"),
            _tweet("3b", conv="b", created="Mon Mar 04 10:02:01 +0000 2024", text="b3"),
        ],
        has_next=True,
        next_cursor="should-not-be-fetched",
    )

    client = AsyncMock()
    client.request.side_effect = [
        _user_info_response(),
        b1,
        b2,
        b3,
        # No 4th timeline response wired; if the cap fails to trip and
        # a 4th request is made, AsyncMock's default object has no
        # working .json() and the test blows up loudly.
    ]

    source = TwitterSource(
        handle="test_handle",
        since=datetime(2023, 1, 1, tzinfo=timezone.utc),
        api_key="sk-test",
        request_delay_seconds=0.0,
        max_tweets=6,  # trips after batch 3: tweets_fetched=6 >= 6
        http_client=client,
    )

    yielded: list[Any] = []
    with pytest.raises(TwitterBudgetExceeded):
        async for item in source.fetch():
            yielded.append(item)

    # Both threads flushed to the caller BEFORE the exception surfaced.
    assert len(yielded) == 2
    conv_ids = {item.metadata["conversation_id"] for item in yielded}
    assert conv_ids == {"a", "b"}

    # Each thread carries all three tweets from its conversation —
    # proves the flush assembled full threads, not fragments.
    for item in yielded:
        assert item.metadata["thread_length"] == 3

    # Stats at the moment the exception surfaced: 2 threads emitted,
    # 6 tweets fetched across 3 timeline pages + 1 user-info call.
    assert source.stats.threads_emitted == 2
    assert source.stats.tweets_fetched == 6
    assert client.request.call_count == 4
