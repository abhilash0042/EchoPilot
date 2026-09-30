"""
FastAPI Middleware for Belfry Runtime Safety

Drop-in middleware for FastAPI applications that intercepts LLM-related
requests and enforces Belfry safety decisions.
"""

import json
import logging
from typing import Callable, List, Optional, Set

logger = logging.getLogger(__name__)


class BelfryFastAPIMiddleware:
    """
    FastAPI middleware that intercepts AI requests and enforces Belfry safety.

    Usage:
        from fastapi import FastAPI
        from belfry_labs.middleware.frameworks.fastapi import BelfryFastAPIMiddleware
        from belfry_labs import AsyncBelfryLabsClient

        app = FastAPI()
        belfry = AsyncBelfryLabsClient(api_key="...")

        app.add_middleware(
            BelfryFastAPIMiddleware,
            belfry_client=belfry,
            project_id="proj_123",
            protected_paths=["/chat", "/api/llm"],
        )
    """

    def __init__(
        self,
        app,
        belfry_client,
        project_id: Optional[str] = None,
        protected_paths: Optional[List[str]] = None,
        input_fields: Optional[List[str]] = None,
        fail_open: bool = True,
        block_status_code: int = 400,
    ):
        self.app = app
        self.client = belfry_client
        self.project_id = project_id
        self.protected_paths: Set[str] = set(protected_paths or [])
        self.input_fields = input_fields or ["message", "prompt", "content", "query", "input", "text"]
        self.fail_open = fail_open
        self.block_status_code = block_status_code

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        if self.protected_paths and not any(path.startswith(p) for p in self.protected_paths):
            await self.app(scope, receive, send)
            return

        # Buffer the request body so it can be re-read by the downstream app
        body_chunks = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        body = b"".join(body_chunks)

        content_to_check = None
        try:
            if body:
                payload = json.loads(body.decode("utf-8"))
                for field in self.input_fields:
                    if field in payload:
                        val = payload[field]
                        if isinstance(val, str):
                            content_to_check = val
                            break
                        elif isinstance(val, list):
                            user_messages = [
                                m["content"]
                                for m in val
                                if isinstance(m, dict) and m.get("role") == "user"
                            ]
                            if user_messages:
                                content_to_check = user_messages[-1]
                            break
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        if content_to_check:
            try:
                decision = await self.client.protect(content_to_check, project_id=self.project_id)
                action = (
                    decision.get("action", "allow")
                    if isinstance(decision, dict)
                    else getattr(decision, "action", "allow")
                )
                action = action.value if hasattr(action, "value") else str(action)

                if action == "block":
                    await self._send_blocked_response(send, "Request blocked by Belfry safety policy")
                    return

            except Exception as e:
                logger.warning(f"Belfry FastAPI middleware check failed: {e}")
                if not self.fail_open:
                    await self._send_blocked_response(send, "Safety check unavailable")
                    return

        # Reconstruct receive callable with the already-buffered body
        body_sent = False

        async def receive_with_body():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Park until the connection closes (should not normally be reached)
            return {"type": "http.disconnect"}

        await self.app(scope, receive_with_body, send)

    async def _send_blocked_response(self, send: Callable, message: str) -> None:
        """Send a blocked HTTP response."""
        body = json.dumps({"error": message, "blocked_by": "belfry"}).encode()
        await send({
            "type": "http.response.start",
            "status": self.block_status_code,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})
