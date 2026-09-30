"""Belfry Atlas MCP stdio server.

Forwards JSON-RPC to POST https://app.belfrylabs.ai/api/v1/mcp (or BELFRY_API_URL).

Usage::

    BELFRY_API_KEY=bak_... python -m belfry_labs.mcp_server
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict

DEFAULT_URL = "https://app.belfrylabs.ai/api/v1/mcp"

TOOL_NAMES = (
    "list_projects",
    "quick_scan",
    "list_findings",
    "build_executive_report",
    "list_mcp_servers",
)


def _endpoint() -> str:
    base = (os.getenv("BELFRY_API_URL") or DEFAULT_URL).rstrip("/")
    if base.endswith("/mcp"):
        return base
    if base.endswith("/api/v1"):
        return base + "/mcp"
    return base + "/api/v1/mcp"


def _headers() -> Dict[str, str]:
    key = os.getenv("BELFRY_API_KEY") or os.getenv("BELFRY_LABS_API_KEY") or ""
    token = os.getenv("BELFRY_TOKEN") or ""
    headers = {"Content-Type": "application/json", "User-Agent": "belfry-mcp-stdio/1.0"}
    if key:
        headers["X-API-Key"] = key
        headers["Authorization"] = f"Bearer {key}"
    elif token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def forward(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(_endpoint(), data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {"code": exc.code, "message": body[:500]},
        }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        out = forward(msg)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
