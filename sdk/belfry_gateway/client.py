"""Blocking and streaming client for POST /api/v1/gateway/v1/chat/completions."""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Iterator, List, Optional
from urllib import error, request

from belfry_gateway.errors import (
    AuthenticationError,
    BelfryGatewayError,
    ConfigurationError,
    GatewayTimeout,
    RateLimitError,
)
from belfry_gateway.types import ChatChunk, ChatCompletion

DEFAULT_BASE_URL = "http://127.0.0.1:8000/api/v1"
DEFAULT_TIMEOUT_SECONDS = 330
CHAT_PATH = "/gateway/v1/chat/completions"
RESUME_PATH = "/gateway/v1/chat/completions/resume"
_USER_AGENT = "belfry-gateway-sdk/0.1.0"


class Completions:
    def __init__(self, gateway: "BelfryGateway") -> None:
        self._gateway = gateway

    def create(
        self,
        *,
        project_id: str,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        """Wait for one JSON completion. This is the default production call."""
        payload = self._gateway.build_chat_body(
            project_id=project_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
            stream=False,
        )
        data = self._gateway.post_json(CHAT_PATH, payload)
        return ChatCompletion.from_dict(data)

    def stream(
        self,
        *,
        project_id: str,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[ChatChunk]:
        """Read SSE chunks from the same route with stream set to true.

        The gateway finishes the provider call before the first content chunk,
        then emits the answer in pieces. Stop when the server sends ``[DONE]``.
        """
        payload = self._gateway.build_chat_body(
            project_id=project_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
            stream=True,
        )
        response = self._gateway.open(CHAT_PATH, payload, accept="text/event-stream")
        try:
            yield from _read_sse(response)
        finally:
            response.close()

    def resume(self, *, approval_id: str, request_hash: str) -> ChatCompletion:
        """Continue a require_approval completion after a human approves it."""
        if not approval_id or not request_hash:
            raise ConfigurationError("approval_id and request_hash are required")
        data = self._gateway.post_json(
            RESUME_PATH,
            {"approval_id": approval_id, "request_hash": request_hash},
        )
        return ChatCompletion.from_dict(data)


class Chat:
    def __init__(self, gateway: "BelfryGateway") -> None:
        self.completions = Completions(gateway)


class BelfryGateway:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise ConfigurationError("api_key is required")
        if not base_url or not str(base_url).strip():
            raise ConfigurationError("base_url is required")
        self.api_key = str(api_key).strip()
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.chat = Chat(self)

    def close(self) -> None:
        return None

    def __enter__(self) -> "BelfryGateway":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def build_chat_body(
        self,
        *,
        project_id: str,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        metadata: Optional[Dict[str, Any]],
        stream: bool,
    ) -> Dict[str, Any]:
        if not project_id or not str(project_id).strip():
            raise ConfigurationError("project_id is required")
        if not model or not str(model).strip():
            raise ConfigurationError("model is required")
        if not messages:
            raise ConfigurationError("messages is required")
        body: Dict[str, Any] = {
            "project_id": project_id,
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if metadata:
            body["metadata"] = metadata
        return body

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.open(path, payload, accept="application/json")
        try:
            raw = response.read()
        finally:
            response.close()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def open(self, path: str, payload: Dict[str, Any], *, accept: str):
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "User-Agent": _USER_AGENT,
        }
        if self.api_key.startswith("bas_"):
            headers["X-API-Key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            response = request.urlopen(req, timeout=self.timeout)
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise _http_error(exc.code, raw) from exc
        except socket.timeout as exc:
            raise GatewayTimeout(f"gateway timed out after {self.timeout}s") from exc
        except error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                raise GatewayTimeout(f"gateway timed out after {self.timeout}s") from exc
            raise BelfryGatewayError(f"gateway request failed: {reason}") from exc
        status = getattr(response, "status", None) or response.getcode()
        if status >= 400:
            raw = response.read().decode("utf-8", errors="replace")
            response.close()
            raise _http_error(status, raw)
        return response


def _read_sse(response) -> Iterator[ChatChunk]:
    """Read every SSE event. A dropped connection is an error, not a short answer."""
    saw_done = False
    while True:
        line = response.readline()
        if not line:
            break
        if isinstance(line, bytes):
            text = line.decode("utf-8", errors="replace")
        else:
            text = line
        text = text.strip()
        if not text or text.startswith(":"):
            continue
        if not text.startswith("data:"):
            continue
        data = text[5:].strip()
        if data == "[DONE]":
            saw_done = True
            break
        payload = json.loads(data)
        chunk = ChatChunk.from_dict(payload)
        # Keep prior lines on screen. A bare carriage return would erase them.
        chunk.delta = chunk.delta.replace("\r\n", "\n").replace("\r", "\n")
        yield chunk
    if not saw_done:
        raise BelfryGatewayError(
            "The gateway closed the stream before it finished. The text above is incomplete."
        )


def _http_error(status: int, raw: str) -> BelfryGatewayError:
    body: Any
    try:
        body = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        body = raw
    message = _error_message(status, body, raw)
    if status in {401, 403}:
        return AuthenticationError(message, status=status, body=body)
    if status == 429:
        return RateLimitError(message, status=status, body=body)
    return BelfryGatewayError(message, status=status, body=body)


def _error_message(status: int, body: Any, raw: str) -> str:
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            code = detail.get("code")
            text = detail.get("message") or detail.get("detail")
            if code and text:
                return f"{code}: {text}"
            if code:
                return str(code)
        if detail:
            return str(detail)
    if raw:
        return raw
    return f"gateway request failed with HTTP {status}"
