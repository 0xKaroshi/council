# User context

The `user_context/` directory is where *your situation* lives.
Mentor archives carry generic operator wisdom; user-context files
carry the constraints, audience, brand, and open questions that
let an LLM turn generic advice into actionable advice for **you
specifically**.

## How it composes

```
council convene "how do I price my new product?"
        │
        ▼
   ┌────────────────────┐    ┌─────────────────────────┐
   │ council_retrieve   │    │ get_user_context        │
   │ (mentor snippets)  │    │ (your situation files)  │
   └────────────────────┘    └─────────────────────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
            calling LLM synthesizes
                       │
                       ▼
        "Given your audience size [audience.md] and
         your solo-operator constraint [constraints.md],
         the council's pricing advice resolves like this..."
```

The calling LLM is expected to read both pools — context first,
then mentor snippets — and produce a synthesis that **inline-cites
which slot informed which point**. That's how you tell whether
the answer is generic mentor parroting vs. actually grounded in
your situation.

## What ships in the repo

Empty templates ending in `.example.md`. They're committed to
git so the schema is visible without you having to fill them in
to clone the repo. Filled-in `.md` copies are gitignored — your
private context never leaves your machine.

```
user_context/
├── README.md                    # explains this layer (committed)
├── brand.example.md             # template (committed)
├── business.example.md          # template (committed)
├── audience.example.md          # template (committed)
├── constraints.example.md       # template (committed)
├── questions.example.md         # template (committed)
├── brand.md                     # YOUR copy (gitignored)
├── business.md                  # YOUR copy (gitignored)
└── ...
```

The set of files is **not fixed** — anything matching `*.md`
(and not `*.example.md`) is loaded. Add `metrics.md`,
`competitors.md`, `product_roadmap.md`, whatever you need.

## How the LLM tool sees it

`get_user_context()` returns:
- For `file="all"` (default): every real file concatenated, each
  preceded by `# <filename>\n\n` and separated by `---`.
- For `file="<stem>"`: just that one file's body.

Empty / missing slots surface as explicit markers:

```
# audience.md

(empty: audience.md — context slot not yet populated)
```

So the LLM can say "I'd answer this better if your `audience.md`
were filled in" instead of silently working without context.

## What to put in each template

The bundled `*.example.md` files are starting points — read each
one for the suggested shape. Quick summary:

| Template                  | Goes in this file                                                |
|---------------------------|------------------------------------------------------------------|
| `brand.example.md`        | One-line positioning, what you stand for, what you avoid, voice |
| `business.example.md`     | What you do, who pays, stage, what's working / stuck             |
| `audience.example.md`     | Composition, persona, what pulls them in, what they pay for      |
| `constraints.example.md`  | Time, team, money, energy, life situation, hard nos              |
| `questions.example.md`    | Open strategic questions you're actively pressure-testing        |

## What NOT to put here

- **Anything you don't want a future LLM to read.** Treat
  these files as inputs to an inference layer, not personal
  notes. Specifically: no API keys, no PII, no client names you
  haven't cleared, no half-baked legal positions.
- **Long-form essays.** Save those as actual blog posts and
  ingest them as a mentor source if you want the LLM to query
  them. The user-context layer is for the *current snapshot* of
  your situation, not your archive.
- **Things that change every week.** A "metrics" file you update
  daily is fine; a "what I did yesterday" file makes the
  context stale. Keep these documents at a half-life of a
  month or longer.

## Iteration loop

These files drift fast. Audience grows, positioning sharpens,
constraints change. Edit them freely; changes land on the next
tool call (no restart, no re-embed — `get_user_context` reads
from disk every call).

A reasonable rhythm:
- **Weekly**: `questions.md` — what's open, what just resolved.
- **Monthly**: `business.md` and `audience.md` — what's
  working, who's actually buying.
- **Quarterly**: `brand.md` and `constraints.md` — slower-
  changing positioning and life situation.

The ones you'd want to put off editing are the ones it pays
most to keep current.

## Why this layer exists

Mentor brains are powerful but generic. A `convene` answer that
ignores your real audience size, real time budget, and real open
questions is a worse answer than one that incorporates them
even partially. The user-context layer is the seam where "your
situation" enters the loop, and it's small on purpose: ~5
markdown files, hand-edited, no DB, no embedding, no churn.

If a piece of your context starts feeling like it belongs in a
DB instead of a markdown file (e.g., a long table of customer
segments), that's a signal that piece is more like a mentor
source than user context. Move it: ingest it as its own
"mentor" archive, query it from `convene`.
