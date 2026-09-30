"""
MCP Server Middleware — protect Model Context Protocol tool handlers.

Wraps MCP server tool handlers to add Belfry safety checks before
tool execution.
"""

import logging
import functools
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MCPToolBlockedError(Exception):
    """Raised when an MCP tool call is blocked by Belfry."""
    def __init__(self, tool_name: str, message: str):
        super().__init__(f"MCP tool '{tool_name}' blocked: {message}")
        self.tool_name = tool_name


class MCPServerMiddleware:
    """
    Middleware for MCP (Model Context Protocol) servers.

    Wraps tool handler registration to add Belfry safety checks.

    Usage with the official MCP Python SDK:
        from mcp import McpServer
        from belfry_labs.middleware.mcp import MCPServerMiddleware
        from belfry_labs import AsyncBelfryLabsClient

        belfry = AsyncBelfryLabsClient(api_key="...")
        mcp_middleware = MCPServerMiddleware(belfry, project_id="proj_123")

        server = McpServer("my-server")

        # Wrap a tool handler
        @mcp_middleware.tool("search_emails")
        async def search_emails(query: str) -> list:
            return email_db.search(query)
    """

    def __init__(
        self,
        belfry_client,
        project_id: Optional[str] = None,
        fail_open: bool = False,
    ):
        self.client = belfry_client
        self.project_id = project_id
        self.fail_open = fail_open

    def tool(self, tool_name: str) -> Callable:
        """Decorator factory for protecting MCP tool handlers."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                content = f"mcp_tool:{tool_name} args:{str(kwargs)[:500]}"
                try:
                    decision = await self.client.runtime(
                        content,
                        project_id=self.project_id,
                        context={"tool_name": tool_name, "check_type": "mcp_tool"},
                    )
                    action = (
                        decision.get("action", "allow")
                        if isinstance(decision, dict)
                        else getattr(decision, "action", "allow")
                    )
                    action = action.value if hasattr(action, "value") else str(action)

                    if action == "block":
                        raise MCPToolBlockedError(
                            tool_name,
                            "Blocked by Belfry safety engine",
                        )
                    elif action == "warn":
                        logger.warning(f"Belfry WARN on MCP tool '{tool_name}'")

                except MCPToolBlockedError:
                    raise
                except Exception as e:
                    logger.warning(f"Belfry MCP check failed for '{tool_name}': {e}")
                    if not self.fail_open:
                        raise MCPToolBlockedError(tool_name, f"Safety check unavailable: {e}")

                return await func(*args, **kwargs)

            return wrapper
        return decorator

    def protect_server(self, server: Any) -> Any:
        """
        Wrap an MCP server instance to protect all registered tools.

        Works with servers that have a `_tools` attribute mapping tool names to handlers.
        Returns the same server with wrapped tool handlers.
        """
        if hasattr(server, "_tools"):
            for tool_name, handler in server._tools.items():
                server._tools[tool_name] = self.tool(tool_name)(handler)
        return server
