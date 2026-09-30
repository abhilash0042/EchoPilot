"""
Tool Call Guard — intercept LLM tool/function calls before execution.

Decorates tool functions to check their inputs against the Belfry
runtime safety engine before executing.
"""

import asyncio
import logging
import functools
import json
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolCallBlockedError(Exception):
    """Raised when a tool call is blocked by Belfry safety engine."""
    def __init__(self, tool_name: str, message: str, decision: Dict = None):
        super().__init__(f"Tool '{tool_name}' blocked: {message}")
        self.tool_name = tool_name
        self.decision = decision or {}


class ToolCallGuard:
    """
    Guards tool/function calls from LLM agents.

    Can be used as a decorator or to wrap existing tool functions.

    Usage:
        from belfry_labs.middleware.tools import ToolCallGuard
        from belfry_labs import AsyncBelfryLabsClient

        client = AsyncBelfryLabsClient(api_key="...")
        guard = ToolCallGuard(client, project_id="proj_123")

        # Use as decorator
        @guard.protect
        async def send_email(to: str, subject: str, body: str) -> str:
            # ... send email
            return "sent"

        # Or wrap OpenAI-style tool call dispatch
        result = await guard.execute_tool_call(tool_call, tool_registry)
    """

    # Tools that should always be blocked regardless of content
    ALWAYS_BLOCKED_TOOLS = frozenset([
        "exec", "eval", "shell", "execute_command", "run_command",
        "delete_all", "drop_database", "format_disk",
    ])

    # Tools with elevated risk that require extra scrutiny
    HIGH_RISK_TOOLS = frozenset([
        "write_file", "delete_file", "send_email", "send_message",
        "post_webhook", "http_request", "sql_query", "database_query",
    ])

    def __init__(
        self,
        belfry_client,
        project_id: Optional[str] = None,
        agent_id: str = "sdk-agent",
        blocked_tools: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        max_calls_per_session: int = 50,
        fail_open: bool = False,
    ):
        self.client = belfry_client
        self.project_id = project_id
        self.agent_id = agent_id
        self.blocked_tools = set(blocked_tools or []) | self.ALWAYS_BLOCKED_TOOLS
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.max_calls_per_session = max_calls_per_session
        self.fail_open = fail_open
        self._call_count = 0
        self._call_history: List[Dict] = []

    def protect(self, func: Callable) -> Callable:
        """Decorator that adds Belfry safety checks to a tool function."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tool_name = func.__name__
            call_args = {"args": list(args), "kwargs": kwargs}
            await self._check_tool_call(tool_name, call_args)
            return await func(*args, **kwargs)
        return wrapper

    async def execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        tool_registry: Dict[str, Callable],
    ) -> Any:
        """
        Execute an OpenAI-style tool call with safety checks.

        Args:
            tool_call: Dict with 'function' key containing 'name' and 'arguments'
            tool_registry: Dict mapping tool names to callables

        Returns:
            Tool execution result
        """
        name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {"raw": arguments_str}

        await self._check_tool_call(name, arguments)

        if name not in tool_registry:
            raise ValueError(f"Tool '{name}' not found in registry")

        func = tool_registry[name]
        if asyncio.iscoroutinefunction(func):
            return await func(**arguments)
        return func(**arguments)

    async def _check_tool_call(self, tool_name: str, arguments: Dict) -> None:
        """Internal: Check a tool call against safety rules."""
        self._call_count += 1
        if self._call_count > self.max_calls_per_session:
            raise ToolCallBlockedError(
                tool_name,
                f"Session tool call limit ({self.max_calls_per_session}) exceeded",
            )

        if tool_name in self.blocked_tools:
            raise ToolCallBlockedError(
                tool_name,
                f"Tool '{tool_name}' is in the blocked tools list",
            )

        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            raise ToolCallBlockedError(
                tool_name,
                f"Tool '{tool_name}' is not in the allowed tools list",
            )

        content_to_check = f"tool_call:{tool_name} args:{json.dumps(arguments)[:500]}"
        try:
            if hasattr(self.client, "check_agent"):
                decision = await self.client.check_agent(
                    content_to_check,
                    tool_name=tool_name,
                    agent_id=self.agent_id,
                    project_id=self.project_id,
                    tool_calls_count=self._call_count,
                )
            else:
                decision = await self.client.runtime(
                    content_to_check,
                    project_id=self.project_id,
                    context={
                        "tool_name": tool_name,
                        "tool_calls_count": self._call_count,
                        "check_type": "agent_tool",
                    },
                )
            action = (
                decision.get("action", "allow")
                if isinstance(decision, dict)
                else getattr(decision, "action", "allow")
            )
            action = action.value if hasattr(action, "value") else str(action)

            if action == "block":
                raise ToolCallBlockedError(
                    tool_name,
                    "Blocked by Belfry safety engine",
                    decision=decision if isinstance(decision, dict) else {},
                )

        except ToolCallBlockedError:
            raise
        except Exception as e:
            logger.warning(f"Belfry tool check failed for '{tool_name}': {e}")
            if not self.fail_open:
                raise ToolCallBlockedError(
                    tool_name,
                    f"Safety check unavailable: {e}",
                )

        self._call_history.append({"tool": tool_name, "call_count": self._call_count})

    def reset_session(self) -> None:
        """Reset session counters (call between conversations)."""
        self._call_count = 0
        self._call_history.clear()

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_history(self) -> List[Dict]:
        return list(self._call_history)
