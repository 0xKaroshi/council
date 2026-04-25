"""council — single-binary CLI surface.

Self-contained: no MCP server, no OAuth, no Docker required. Just
Python + an OpenAI key. Optional Twitter API key if Twitter sources
are configured.

Commands:

    council init                       Scaffold ./config + ./user_context
    council list-mentors               Show every mentor in the YAML config
    council ingest <slug>              Backfill one mentor's archive
    council ingest --all               Backfill every configured mentor
    council embed <slug>               Embed pending chunks for one mentor
    council embed --all                Embed pending chunks across the lineup
    council ask <slug> "question"      Query a single mentor
    council convene "question"         Fan a question out to every mentor
    council status                     Per-mentor corpus snapshot
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import textwrap
from pathlib import Path

import click

from app.ingest.mentors import MentorConfig, load_mentors

log = logging.getLogger("council.cli")


def _bundled_dir(name: str) -> Path | None:
    """Locate a top-level bundled package dir (config / user_context /
    examples) via importlib.resources, falling back to the
    walk-up-from-this-file path used by editable installs."""
    try:
        from importlib.resources import files
        try:
            traversable = files(name)
            real_path = Path(str(traversable))
            if real_path.is_dir():
                return real_path
        except (ModuleNotFoundError, FileNotFoundError, AttributeError):
            pass
    except ImportError:
        pass
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / name
    if candidate.is_dir():
        return candidate
    return None


_TEMPLATES_DIR = _bundled_dir("user_context") or (
    Path(__file__).resolve().parent.parent / "user_context"
)
_BUNDLED_CONFIG = _bundled_dir("config")
_BUNDLED_EXAMPLES = _bundled_dir("examples")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _err(msg: str) -> None:
    click.echo(click.style(msg, fg="red"), err=True)


def _info(msg: str) -> None:
    click.echo(msg)


def _safe_load_mentors() -> dict[str, MentorConfig]:
    try:
        return load_mentors()
    except FileNotFoundError as exc:
        _err(str(exc))
        sys.exit(2)


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------

@click.group(help="council — multi-mentor knowledge base CLI")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    _configure_logging(verbose)
    ctx.ensure_object(dict)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@cli.command(help="Scaffold ./config/mentors.yaml and ./user_context/ from templates.")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def init(force: bool) -> None:
    config_dir = Path("./config")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "mentors.yaml"
    template_path = (
        (_BUNDLED_CONFIG / "mentors.yaml")
        if _BUNDLED_CONFIG is not None
        else Path(__file__).resolve().parent.parent / "config" / "mentors.yaml"
    )

    if config_path.exists() and not force:
        _info(f"  · {config_path} already exists (pass --force to overwrite)")
    elif template_path.exists() and template_path.resolve() == config_path.resolve():
        # Running from inside the source repo — config_path IS the
        # template. Nothing to copy.
        _info(f"  · {config_path} is the bundled template (running in-repo)")
    else:
        if template_path.exists():
            shutil.copy(template_path, config_path)
        else:
            config_path.write_text(_FALLBACK_YAML)
        _info(f"  ✔ wrote {config_path}")

    ctx_dir = Path("./user_context")
    ctx_dir.mkdir(exist_ok=True)
    if _TEMPLATES_DIR.exists() and _TEMPLATES_DIR.resolve() != ctx_dir.resolve():
        copied = 0
        for tmpl in _TEMPLATES_DIR.glob("*.example.md"):
            target = ctx_dir / tmpl.name
            if target.exists() and not force:
                continue
            shutil.copy(tmpl, target)
            copied += 1
        _info(f"  ✔ {copied} user_context templates available at {ctx_dir}/")
    elif _TEMPLATES_DIR.resolve() == ctx_dir.resolve():
        _info(f"  · {ctx_dir} is the bundled templates dir (running in-repo)")
    else:
        _info("  ! no built-in user_context templates found; you can create your own .md files in ./user_context/")

    # No longer seed data/mentors/ from examples — the bundled-DB
    # fallback in app.ingest.db means reads against an empty user
    # data dir transparently use the bundled archives.  Just create
    # the directory so first-ingest doesn't have to mkdir.
    data_mentors = Path("./data/mentors")
    data_mentors.mkdir(parents=True, exist_ok=True)
    seeded = 0
    if seeded:
        _info(
            f"  ✔ seeded {seeded} example mentor archives into {data_mentors}/ "
            "(re-ingestion will overwrite them with fresh data)"
        )

    _info("\nNext:")
    _info("  1. Edit config/mentors.yaml to taste")
    _info("  2. Copy user_context/*.example.md → user_context/*.md and fill them in")
    _info("  3. The bundled example archives are queryable immediately —")
    _info("     just set OPENAI_API_KEY in .env and run:")
    _info("       council ask paulgraham 'what makes a startup hard?'")
    _info("       council convene 'should i raise venture capital?'")
    _info("  4. To pull fresh content for your own mentor lineup, set")
    _info("     TWITTER_API_KEY (if any mentor uses twitter) and run:")
    _info("       council ingest --all && council embed --all")


# ---------------------------------------------------------------------------
# list-mentors
# ---------------------------------------------------------------------------

@cli.command("list-mentors", help="Show every mentor in the YAML config.")
def list_mentors_cmd() -> None:
    mentors = _safe_load_mentors()
    if not mentors:
        _info("(no mentors configured — run `council init`)")
        return
    for cfg in mentors.values():
        _info(f"- {cfg.slug}  {cfg.display_name}")
        if cfg.domain_focus:
            _info(f"    focus: {cfg.domain_focus}")
        for s in cfg.sources:
            kind_detail = []
            if s.handle:
                kind_detail.append(f"handle=@{s.handle}")
            if s.domain:
                kind_detail.append(f"domain={s.domain}")
            if s.rss_url:
                kind_detail.append(f"rss={s.rss_url}")
            _info(f"    source: {s.type}  {' '.join(kind_detail)}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command(help="Per-mentor corpus snapshot (chunks, embedded, date range, source).")
def status() -> None:
    mentors = _safe_load_mentors()
    if not mentors:
        _info("(no mentors configured — run `council init`)")
        return
    from app.ingest.db import open_mentor_db, resolve_mentor_db_path

    rows: list[tuple[str, int, int, str, str, str]] = []
    for slug, _cfg in mentors.items():
        path, source = resolve_mentor_db_path(slug, fallback_to_bundled=True)
        if source == "missing":
            rows.append((slug, 0, 0, "—", "—", "missing"))
            continue
        db = open_mentor_db(slug, fallback_to_bundled=True)
        try:
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            try:
                embedded = db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
            except Exception:
                embedded = 0
            r = db.execute(
                "SELECT MIN(date), MAX(date) FROM chunks WHERE date IS NOT NULL"
            ).fetchone()
            mn, mx = r[0] or "—", r[1] or "—"
        finally:
            db.close()
        rows.append(
            (slug, int(chunks), int(embedded), str(mn)[:10], str(mx)[:10], source)
        )

    _info(
        f"{'mentor':18s} {'chunks':>7s} {'embedded':>9s}  "
        f"{'date range':23s} {'source':>10s}"
    )
    _info("-" * 74)
    for slug, ch, em, mn, mx, source in rows:
        date_range = f"{mn} → {mx}"
        _info(
            f"{slug:18s} {ch:>7d} {em:>9d}  {date_range:23s} {f'({source})':>10s}"
        )


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command(help="Backfill one mentor's archive (or all with --all).")
@click.argument("slug", required=False)
@click.option("--all", "all_", is_flag=True, help="Ingest every configured mentor.")
@click.option(
    "--source",
    type=click.Choice(["all", "twitter", "blog"]),
    default="all",
    show_default=True,
    help="Which source type(s) to pull.",
)
@click.option("--max-tweets", type=int, default=20000, show_default=True)
@click.option("--max-cost-usd", type=float, default=None, help="Optional cost cap per Twitter run.")
@click.option("--max-posts", type=int, default=500, show_default=True)
@click.option("--since", default="2024-01-01", show_default=True, help="ISO date floor for Twitter.")
@click.option("--restart", is_flag=True, help="Ignore stored cursor; start fresh.")
@click.option("--dry-run", is_flag=True, help="Fetch + chunk but do not write to the mentor DB.")
def ingest(
    slug: str | None,
    all_: bool,
    source: str,
    max_tweets: int,
    max_cost_usd: float | None,
    max_posts: int,
    since: str,
    restart: bool,
    dry_run: bool,
) -> None:
    if all_ == bool(slug):
        _err("Pass either a mentor slug or --all (not both, not neither).")
        sys.exit(2)

    mentors = _safe_load_mentors()
    targets = list(mentors.values()) if all_ else [mentors[slug]] if slug in mentors else []
    if not targets:
        _err(f"Unknown mentor: {slug!r}. Configured: {sorted(mentors.keys())}")
        sys.exit(2)

    from scripts.ingest import main as ingest_main

    final_rc = 0
    for cfg in targets:
        sources_to_run = _resolve_sources(cfg, source)
        if not sources_to_run:
            _info(f"[{cfg.slug}] no applicable sources for --source {source}; skipping")
            continue
        for src in sources_to_run:
            argv = [cfg.slug, "--source", src, "--since", since]
            if dry_run:
                argv.append("--dry-run")
            if restart:
                argv.append("--restart")
            if src == "twitter":
                argv += ["--max-tweets", str(max_tweets)]
                if max_cost_usd is not None:
                    argv += ["--max-cost-usd", str(max_cost_usd)]
            else:
                argv += ["--max-posts", str(max_posts)]
            _info(f"\n=== ingest [{cfg.slug}/{src}] ===")
            rc = ingest_main(argv)
            if rc and not final_rc:
                final_rc = rc
    sys.exit(final_rc)


def _resolve_sources(cfg: MentorConfig, requested: str) -> list[str]:
    has_blog = bool(cfg.blog_domains)
    has_twitter = cfg.twitter_handle is not None
    if requested == "twitter":
        return ["twitter"] if has_twitter else []
    if requested == "blog":
        return ["blog"] if has_blog else []
    out = []
    if has_blog:
        out.append("blog")
    if has_twitter:
        out.append("twitter")
    return out


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

@cli.command(help="Embed pending chunks (idempotent — only new chunks are sent to OpenAI).")
@click.argument("slug", required=False)
@click.option("--all", "all_", is_flag=True, help="Embed every configured mentor.")
@click.option("--limit", type=int, default=None, help="Stop after N chunks this run.")
@click.option("--dry-run", is_flag=True, help="Estimate cost; do not call OpenAI.")
def embed(slug: str | None, all_: bool, limit: int | None, dry_run: bool) -> None:
    if all_ == bool(slug):
        _err("Pass either a mentor slug or --all (not both, not neither).")
        sys.exit(2)
    mentors = _safe_load_mentors()
    targets = list(mentors.values()) if all_ else [mentors[slug]] if slug in mentors else []
    if not targets:
        _err(f"Unknown mentor: {slug!r}. Configured: {sorted(mentors.keys())}")
        sys.exit(2)

    from scripts.embed import main as embed_main

    final_rc = 0
    for cfg in targets:
        argv = [cfg.slug]
        if limit is not None:
            argv += ["--limit", str(limit)]
        if dry_run:
            argv.append("--dry-run")
        _info(f"\n=== embed [{cfg.slug}] ===")
        rc = embed_main(argv)
        if rc and not final_rc:
            final_rc = rc
    sys.exit(final_rc)


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

@cli.command(help="Query one mentor's archive and print the retrieved snippets.")
@click.argument("slug")
@click.argument("question", nargs=-1, required=True)
@click.option("-k", "--k", type=int, default=8, show_default=True)
@click.option("--recency-bias", is_flag=True, help="Prefer more recent snippets.")
def ask(slug: str, question: tuple[str, ...], k: int, recency_bias: bool) -> None:
    mentors = _safe_load_mentors()
    if slug not in mentors:
        _err(f"Unknown mentor: {slug!r}. Configured: {sorted(mentors.keys())}")
        sys.exit(2)
    q = " ".join(question).strip()
    if not q:
        _err("Provide a question.")
        sys.exit(2)
    from app.tools.search import search

    out = asyncio.run(search(mentor_slug=slug, question=q, k=k, recency_bias=recency_bias))
    click.echo(out)


# ---------------------------------------------------------------------------
# convene
# ---------------------------------------------------------------------------

@cli.command(help="Fan a question out to every configured mentor in parallel.")
@click.argument("question", nargs=-1, required=True)
@click.option("-k", "--k-per-mentor", type=int, default=8, show_default=True)
@click.option("--recency-bias", is_flag=True, help="Prefer more recent snippets.")
def convene(question: tuple[str, ...], k_per_mentor: int, recency_bias: bool) -> None:
    q = " ".join(question).strip()
    if not q:
        _err("Provide a question.")
        sys.exit(2)
    from app.tools.council_retrieve import council_retrieve

    out = asyncio.run(
        council_retrieve(question=q, k_per_mentor=k_per_mentor, recency_bias=recency_bias)
    )
    click.echo(out)


# ---------------------------------------------------------------------------
# Fallback yaml when the package's own template isn't found
# ---------------------------------------------------------------------------

_FALLBACK_YAML = textwrap.dedent("""\
    # council — mentor lineup. See CONFIGURING_MENTORS.md for the full spec.

    mentors:
      - slug: example
        display_name: Example Mentor
        domain_focus: "what this mentor's archive is best at"
        sources:
          - type: blog
            domain: example.com
""")


def main() -> None:
    cli(prog_name="council")


if __name__ == "__main__":
    main()
