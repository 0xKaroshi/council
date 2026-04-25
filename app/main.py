"""FastAPI entrypoint for the optional MCP server mode.

The CLI (`council ask`, `council convene`) is the primary surface
in council v1 and needs none of this. This module is here for
users who want to expose their council over MCP — typically deployed
behind their own reverse proxy at their own domain.

OAuth is intentionally NOT shipped in v1. `MCP_AUTH_MODE=none` is
the only supported value; `oauth` is a documented future extension
(see CONTRIBUTING.md). For production deploys today, put the
server behind a reverse proxy that handles auth (mTLS, basic auth,
or your own gateway).

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8440
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.mcp_server import mcp


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="council MCP server",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "council",
        "env": settings.env,
        "version": "0.1.0",
        "auth_mode": settings.mcp_auth_mode,
    }


@app.get("/")
async def landing() -> dict:
    return {
        "service": "council",
        "description": (
            "Multi-mentor knowledge base. CLI is the primary surface; "
            "this MCP endpoint is the optional server deployment."
        ),
        "mcp_endpoint": f"{settings.public_url.rstrip('/')}/mcp/",
        "auth_mode": settings.mcp_auth_mode,
    }


app.mount("/mcp", mcp.streamable_http_app())
