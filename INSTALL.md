# Install

Two install paths. Pick one — local CLI is what most people want.

## Local CLI (recommended)

Requires Python 3.10+ and an OpenAI API key.

```bash
git clone https://github.com/<your-username>/council.git
cd council

python3 -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -e .

cp .env.example .env
$EDITOR .env                       # set OPENAI_API_KEY at minimum

council init                       # scaffolds config + user_context
                                   # AND seeds the bundled example
                                   # archives into data/mentors/

# Verify the install path works against the bundled examples:
council ask paulgraham "what makes a startup hard?"
council convene "should I take VC funding?"
```

If `council convene` returns three labeled sections (Paul Graham,
Naval Ravikant, Patrick O'Shaughnessy) with snippets and
mentor-prefixed citations like `[paulgraham_1]`, install is
working.

### Verification steps (clean checkout)

These are the same commands the maintainer runs after building
the repo to confirm a stranger can clone and use it:

```bash
# 1. Clean directory.
rm -rf /tmp/council-clean-test
mkdir /tmp/council-clean-test
cd /tmp/council-clean-test

# 2. Clone (or copy) the repo.
git clone https://github.com/<your-username>/council.git .
# OR for local copies:  cp -r /path/to/council/. .

# 3. Install + scaffold.
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
echo "OPENAI_API_KEY=sk-..." > .env
council init

# 4. Smoke-test single-mentor + council mode.
council ask paulgraham "what makes a startup hard?"
council convene "should I take VC funding?"

# 5. Run the test suite.
pip install -e ".[dev]"
pytest tests/
```

If all four steps complete without errors, the repo is shippable.

### Adding your own mentors

```bash
# Edit the lineup
$EDITOR config/mentors.yaml          # see CONFIGURING_MENTORS.md

# Optional: set Twitter API key if any mentor uses Twitter
echo "TWITTER_API_KEY=..." >> .env

# Pull + embed
council ingest --all
council embed --all

# Or one mentor at a time
council ingest some_mentor --source blog --max-posts 50
council embed some_mentor
```

## Docker

A Dockerfile + docker-compose.yml are included for users who want
to run council inside a container (typically because they're also
exposing the MCP server through their existing reverse proxy
setup). The CLI works inside the container too:

```bash
docker compose up -d --build
docker compose exec council council list-mentors
docker compose exec council council ask paulgraham "..."
```

## MCP server mode (advanced)

This mode exposes council over the MCP Streamable HTTP transport
so Claude Desktop, Claude.ai, or any MCP client can call the
tools directly. Out of scope for v1: OAuth.

```bash
# Start the server (foreground for testing; use docker compose up -d for prod)
uvicorn app.main:app --host 0.0.0.0 --port 8440

# Verify
curl http://localhost:8440/health
# {"status":"ok","service":"council","env":"development","version":"0.1.0","auth_mode":"none"}
```

Six concrete deployment notes:

1. **Auth.** `MCP_AUTH_MODE=none` is the only supported mode in
   v1. **Don't expose this directly to the public internet.** Put
   it behind a reverse proxy (Traefik / nginx / Caddy) that
   handles auth — basic auth, mTLS, or your own gateway.
   OAuth is documented as future work in `CONTRIBUTING.md`.

2. **TLS.** The FastAPI app speaks plain HTTP. Your reverse proxy
   handles TLS termination + cert management.

3. **Public URL.** Set `PUBLIC_URL` to whatever the MCP client
   reaches the server at (must include scheme + host, no
   trailing slash). The MCP handshake advertises this URL.

4. **Allowed hosts.** Add your public host to `ALLOWED_HOSTS` (CSV).
   FastMCP's DNS-rebinding check rejects requests with mismatched
   `Host` headers.

5. **Storage.** `data/` is bind-mounted into the container so
   per-mentor SQLite files persist across restarts. Same for
   `config/` (mentor lineup, read-only) and `user_context/`
   (your situation files, read-only).

6. **Resource sizing.** The container is small (~200MB image, low
   memory). Embed runs are I/O-bound (OpenAI). Retrieval is fast
   (single-mentor < 50ms after embed). A single $5 VPS handles
   this comfortably.

### Deploying behind Traefik (sketch)

Your existing Traefik setup already handles certs and routing.
Add this label block to `docker-compose.yml`:

```yaml
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.council.rule=Host(`council.your-domain.com`)"
      - "traefik.http.routers.council.entrypoints=websecure"
      - "traefik.http.routers.council.tls=true"
      - "traefik.http.routers.council.tls.certresolver=letsencrypt"
      - "traefik.http.services.council.loadbalancer.server.port=8440"
```

Then put a basic-auth middleware in front, or accept that the
endpoint is open and rely on URL secrecy + monitoring.

### Connecting from Claude.ai

Claude.ai's "Connect to MCP server" form takes a URL. With
council deployed at `https://council.your-domain.com`, point it
at `https://council.your-domain.com/mcp/`. Without OAuth, this
works only if you add basic auth at the proxy layer (Claude.ai
will prompt for credentials on first connect). For a real
production setup with OAuth, see `CONTRIBUTING.md`.
