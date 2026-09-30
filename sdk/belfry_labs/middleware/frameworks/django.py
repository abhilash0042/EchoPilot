"""
Django Middleware for Belfry Runtime Safety
"""

import json
import logging
import asyncio
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class BelfryDjangoMiddleware:
    """
    Django middleware for Belfry runtime safety.

    Add to MIDDLEWARE in settings.py:
        MIDDLEWARE = [
            ...
            'belfry_labs.middleware.frameworks.django.BelfryDjangoMiddleware',
        ]

    Configure in settings.py:
        BELFRY_API_KEY = "your-api-key"
        BELFRY_PROJECT_ID = "proj_123"
        BELFRY_PROTECTED_PATHS = ["/api/chat/", "/api/llm/"]
        BELFRY_FAIL_OPEN = True
    """

    def __init__(self, get_response: Callable):
        self.get_response = get_response
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        from django.conf import settings  # type: ignore[import-untyped]

        self.project_id: Optional[str] = getattr(settings, "BELFRY_PROJECT_ID", None)
        self.protected_paths: List[str] = getattr(settings, "BELFRY_PROTECTED_PATHS", [])
        self.fail_open: bool = getattr(settings, "BELFRY_FAIL_OPEN", True)
        self.input_fields: List[str] = getattr(
            settings, "BELFRY_INPUT_FIELDS", ["message", "prompt", "content", "query"]
        )
        api_key: Optional[str] = getattr(settings, "BELFRY_API_KEY", None)

        if api_key:
            try:
                from belfry_labs import AsyncBelfryLabsClient
                self._client = AsyncBelfryLabsClient(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to init Belfry client: {e}")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def __call__(self, request):
        if self._client and self._should_check(request):
            content = self._extract_content(request)
            if content:
                try:
                    loop = self._get_loop()
                    decision = loop.run_until_complete(
                        self._client.protect(content, project_id=self.project_id)
                    )
                    action = (
                        decision.get("action", "allow")
                        if isinstance(decision, dict)
                        else getattr(decision, "action", "allow")
                    )
                    action = action.value if hasattr(action, "value") else str(action)

                    if action == "block":
                        from django.http import JsonResponse  # type: ignore[import-untyped]
                        return JsonResponse(
                            {"error": "Request blocked by Belfry safety policy", "blocked_by": "belfry"},
                            status=400,
                        )
                except Exception as e:
                    logger.warning(f"Belfry Django check failed: {e}")
                    if not self.fail_open:
                        from django.http import JsonResponse  # type: ignore[import-untyped]
                        return JsonResponse({"error": "Safety check unavailable"}, status=503)

        return self.get_response(request)

    def _should_check(self, request) -> bool:
        if not self.protected_paths:
            return bool(request.content_type and "json" in request.content_type)
        return any(request.path.startswith(p) for p in self.protected_paths)

    def _extract_content(self, request) -> Optional[str]:
        try:
            payload = json.loads(request.body)
            for field in self.input_fields:
                if field in payload:
                    val = payload[field]
                    if isinstance(val, str):
                        return val
                    elif isinstance(val, list):
                        user_msgs = [
                            m["content"]
                            for m in val
                            if isinstance(m, dict) and m.get("role") == "user"
                        ]
                        if user_msgs:
                            return user_msgs[-1]
        except Exception:
            pass
        return None
