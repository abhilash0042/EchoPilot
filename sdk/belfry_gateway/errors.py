"""Errors raised by the gateway client."""

from __future__ import annotations

from typing import Any, Optional


class BelfryGatewayError(Exception):
    """Request reached the gateway and was rejected, or the response could not be read."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class AuthenticationError(BelfryGatewayError):
    """The access secret or token was rejected."""


class RateLimitError(BelfryGatewayError):
    """The gateway returned HTTP 429."""


class GatewayTimeout(BelfryGatewayError):
    """The call exceeded the client timeout."""


class ConfigurationError(BelfryGatewayError):
    """The client was constructed or called with missing settings."""
