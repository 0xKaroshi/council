"""FastMCP instance construction.

Registers the four council tools (ping, search, council_retrieve,
get_user_context) with descriptions generated from the live mentor
config so adding a mentor to YAML automatically updates the LLM-
facing tool descriptions on next server restart.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.config import settings
from app.tools.council_retrieve import (
    TOOL_NAME as COUNCIL_RETRIEVE_NAME,
)
from app.tools.council_retrieve import (
    build_tool_description as council_description,
)
from app.tools.council_retrieve import (
    council_retrieve,
)
from app.tools.get_user_context import (
    TOOL_DESCRIPTION as GET_USER_CONTEXT_DESCRIPTION,
)
from app.tools.get_user_context import (
    TOOL_NAME as GET_USER_CONTEXT_NAME,
)
from app.tools.get_user_context import (
    get_user_context,
)
from app.tools.ping import (
    TOOL_DESCRIPTION as PING_DESCRIPTION,
)
from app.tools.ping import (
    TOOL_NAME as PING_NAME,
)
from app.tools.ping import (
    ping,
)
from app.tools.search import (
    TOOL_NAME as SEARCH_NAME,
)
from app.tools.search import (
    build_tool_description as search_description,
)
from app.tools.search import (
    search,
)


def _transport_security() -> TransportSecuritySettings:
    """Expand each allowed host into bare + ":*" wildcard-port forms so
    FastMCP's DNS-rebinding check accepts both "host" and "host:443"."""
    hosts: list[str] = []
    origins: list[str] = []
    for h in settings.allowed_hosts:
        hosts.extend([h, f"{h}:*"])
        origins.extend([
            f"https://{h}",
            f"http://{h}",
            f"https://{h}:*",
            f"http://{h}:*",
        ])
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


mcp = FastMCP(
    name="council",
    instructions=(
        "council — multi-mentor knowledge base. Exposes four tools: "
        "`ping` for connectivity verification; `search` for hybrid "
        "retrieval over a single named mentor's archive; "
        "`council_retrieve` to fan a question out to every configured "
        "mentor in parallel and get back labeled snippet sets for "
        "multi-lens synthesis; and `get_user_context` for loading the "
        "user's own situation files (brand, business context, "
        "constraints) so answers are grounded rather than generic."
    ),
    website_url=settings.public_url,
    stateless_http=True,
    json_response=False,
    streamable_http_path="/",
    transport_security=_transport_security(),
)
mcp._mcp_server.version = "0.1.0"

mcp.tool(name=PING_NAME, description=PING_DESCRIPTION)(ping)
mcp.tool(name=SEARCH_NAME, description=search_description())(search)
mcp.tool(name=COUNCIL_RETRIEVE_NAME, description=council_description())(council_retrieve)
mcp.tool(
    name=GET_USER_CONTEXT_NAME, description=GET_USER_CONTEXT_DESCRIPTION
)(get_user_context)
