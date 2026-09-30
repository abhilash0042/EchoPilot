"""
Flask Middleware for Belfry Runtime Safety

Provides before/after request hooks for Flask applications.
"""

import json
import logging
import asyncio
from typing import List, Optional

logger = logging.getLogger(__name__)


class BelfryFlaskExtension:
    """
    Flask extension for Belfry runtime safety.

    Usage:
        from flask import Flask
        from belfry_labs import AsyncBelfryLabsClient
        from belfry_labs.middleware.frameworks.flask import BelfryFlaskExtension

        app = Flask(__name__)
        belfry = AsyncBelfryLabsClient(api_key="...")
        belfry_ext = BelfryFlaskExtension(belfry, project_id="proj_123")
        belfry_ext.init_app(app)
    """

    def __init__(
        self,
        belfry_client,
        project_id: Optional[str] = None,
        protected_endpoints: Optional[List[str]] = None,
        input_fields: Optional[List[str]] = None,
        fail_open: bool = True,
    ):
        self.client = belfry_client
        self.project_id = project_id
        self.protected_endpoints = set(protected_endpoints or [])
        self.input_fields = input_fields or ["message", "prompt", "content", "query", "input"]
        self.fail_open = fail_open
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def init_app(self, app) -> None:
        """Initialize the extension with a Flask app."""
        app.before_request(self._before_request)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return a running event loop, creating one if necessary."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _before_request(self):
        """Flask before_request hook."""
        try:
            from flask import request
        except ImportError:
            return None

        if self.protected_endpoints and request.endpoint not in self.protected_endpoints:
            return None

        content_to_check: Optional[str] = None
        try:
            if request.is_json:
                payload = request.get_json(force=True, silent=True) or {}
                for field in self.input_fields:
                    if field in payload:
                        val = payload[field]
                        if isinstance(val, str):
                            content_to_check = val
                            break
                        elif isinstance(val, list):
                            user_msgs = [
                                m["content"]
                                for m in val
                                if isinstance(m, dict) and m.get("role") == "user"
                            ]
                            if user_msgs:
                                content_to_check = user_msgs[-1]
                            break
        except Exception:
            return None

        if not content_to_check:
            return None

        try:
            loop = self._get_loop()
            decision = loop.run_until_complete(
                self.client.protect(content_to_check, project_id=self.project_id)
            )
            action = (
                decision.get("action", "allow")
                if isinstance(decision, dict)
                else getattr(decision, "action", "allow")
            )
            action = action.value if hasattr(action, "value") else str(action)

            if action == "block":
                from flask import jsonify
                response = jsonify({"error": "Request blocked by Belfry safety policy", "blocked_by": "belfry"})
                response.status_code = 400
                return response

        except Exception as e:
            logger.warning(f"Belfry Flask check failed: {e}")
            if not self.fail_open:
                from flask import jsonify
                response = jsonify({"error": "Safety check unavailable"})
                response.status_code = 503
                return response

        return None
