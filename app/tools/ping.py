"""Single placeholder MCP tool. Verifies that the server is reachable,
auth (when enabled) passed, and tool dispatch is wired up correctly
end-to-end."""

TOOL_NAME = "ping"
TOOL_DESCRIPTION = (
    "Health check. Returns confirmation that the council MCP server "
    "is reachable and tool dispatch is working."
)


async def ping() -> str:
    return "pong from council"
