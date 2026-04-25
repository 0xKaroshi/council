# user_context/

This directory holds your **own situation** — the markdown files
that ground every mentor query in your specific context rather than
producing generic advice.

When the LLM calls `get_user_context`, it reads every `.md` file in
this directory (skipping `.example.md` templates) and returns them
concatenated. The model uses these to answer questions like *"what
should I do?"* with your actual constraints, audience, brand
positioning, and open questions in front of it.

## Setup

```bash
# Inside this directory, copy each template you want to use and fill it in:
cp brand.example.md       brand.md
cp business.example.md    business.md
cp audience.example.md    audience.md
cp constraints.example.md constraints.md
cp questions.example.md   questions.md
```

The `.example.md` templates are **committed to the repo** — they're
the public, generic shape of each context type. The filled-in `.md`
copies are **gitignored** so your private context never leaves
your machine.

## What each file should contain

The names are conventions, not constraints — anything ending in
`.md` (and not `.example.md`) is loaded. Use the templates as a
starting point and add/rename files as your situation evolves.

| File              | What goes in it                                                    |
|-------------------|--------------------------------------------------------------------|
| `brand.md`        | One-line positioning, voice, what you stand for, what you avoid    |
| `business.md`     | What you do, who pays you, what stage you're at, what's working    |
| `audience.md`     | Who follows you, who buys from you, how they describe themselves   |
| `constraints.md`  | Solo / team size, time budget, energy budget, life situation       |
| `questions.md`    | Open strategic questions you're actively pressure-testing          |

Add others as needed (e.g. `metrics.md`, `competitors.md`,
`product_roadmap.md`). The LLM treats every file equally — name it
something the LLM can read.

## How citations work

When the LLM uses your context to answer, it inline-cites each
file like `[brand.md]` or `[constraints.md]` so you can trace
which slot informed which part of the answer. If a paragraph isn't
cited, the model wasn't using your context for that point.

## Iteration loop

These files are meant to be edited often. They drift fast — your
audience grows, your positioning sharpens, your constraints
change. Edit freely; changes land on the next tool call (no
restart, no re-embed).

## Related docs

- `USER_CONTEXT.md` (in repo root) — design notes on why this
  layer exists, how it composes with retrieval, and what NOT to
  put here.
