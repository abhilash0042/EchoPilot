"""EchoPilot → Belfry Atlas runtime checks via the official Python SDK."""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

# Ensure .env in backend directory is always loaded regardless of current working directory
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)
load_dotenv()  # also load root .env if present

# Vendored SDK (CI/Render) first, then a local Atlas checkout. Override with BELFRY_SDK_PATH.
_SDK_CANDIDATES = [
    Path(os.getenv("BELFRY_SDK_PATH", "")),
    Path(__file__).resolve().parents[1] / "sdk",
    Path(r"C:\Users\Abhilash\belfry-labs\python-sdk"),
]
for _sdk_root in _SDK_CANDIDATES:
    if _sdk_root and (_sdk_root / "belfry_labs" / "__init__.py").is_file():
        _sdk_str = str(_sdk_root)
        if _sdk_str not in sys.path:
            sys.path.insert(0, _sdk_str)
        break

BELFRY_BASE_URL = os.getenv("BELFRY_BASE_URL", "http://localhost:8001/api/v1").rstrip("/")
BELFRY_TENANT_ID = os.getenv("BELFRY_TENANT_ID", "belfry-local")
BELFRY_PROJECT_ID = os.getenv("BELFRY_PROJECT_ID", "proj_5e0b58be7d5a")
BELFRY_API_KEY = (os.getenv("BELFRY_API_KEY") or os.getenv("BELFRY_LABS_API_KEY", "")).strip()

_sdk_client = None
_sdk_import_error = None
if BELFRY_API_KEY:
    try:
        from belfry_labs import AsyncBelfryLabsClient

        _sdk_client = AsyncBelfryLabsClient(
            api_key=BELFRY_API_KEY,
            base_url=BELFRY_BASE_URL,
            tenant_id=BELFRY_TENANT_ID,
            timeout=8,
        )
        print(
            f"[Belfry SDK] Connected for EchoPilot "
            f"project={BELFRY_PROJECT_ID} tenant={BELFRY_TENANT_ID}"
        )
    except ImportError as exc:
        _sdk_import_error = exc
        print(f"[Belfry SDK] Import failed ({exc}); falling back to HTTP checks.")
else:
    print("[Belfry SDK] Notice: BELFRY_API_KEY is not configured; SDK calls will bypass gracefully.")


def sdk_connected() -> bool:
    """True when EchoPilot is using AsyncBelfryLabsClient (not the HTTP fallback)."""
    return _sdk_client is not None


def _headers() -> dict:
    return {
        "X-API-Key": BELFRY_API_KEY,
        "X-Tenant-ID": BELFRY_TENANT_ID,
        "Content-Type": "application/json",
    }


# Runtime "block" stops the call. Runtime "allow" lets it through.
# "warn" is not a block, so the text still passes. A missing decision does not.
_BLOCK_ACTIONS = {"block", "blocked", "terminate", "require_approval"}
_ALLOW_ACTIONS = {"allow", "allowed", "warn"}


def _normalize(result: dict) -> dict:
    action = str(result.get("action") or "error").lower()
    if action == "blocked":
        action = "block"
    elif action == "allowed":
        action = "allow"
    result["action"] = action
    return result


def may_call_llm(result: dict | None) -> bool:
    """Follow the runtime decision: allow proceeds, block stops."""
    if not result:
        return False
    action = str(result.get("action") or "").lower()
    if action in _BLOCK_ACTIONS:
        return False
    return action in _ALLOW_ACTIONS


async def _check_http(kind: str, text: str) -> dict:
    url = f"{BELFRY_BASE_URL}/runtime-safety/check/{kind}"
    payload = {
        "content": text[:1800],
        "tenant_id": BELFRY_TENANT_ID,
        "project_id": BELFRY_PROJECT_ID,
        "persist": True,
    }
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(url, headers=_headers(), json=payload)
        response.raise_for_status()
        return _normalize(response.json())


def _atlas_is_loopback() -> bool:
    host = (urlparse(BELFRY_BASE_URL).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


async def _check(kind: str, text: str) -> dict:
    """POST a runtime check that persists into Belfry Runtime Events."""
    if not text:
        return {"action": "allow"}
    text = text[:1800]
    # Atlas on this machine is not reachable from a Vercel function. Skipping
    # only applies to that loopback URL; a public Atlas URL is still enforced.
    if os.getenv("VERCEL") and _atlas_is_loopback():
        print("[Belfry SDK] Atlas is on localhost; skipping runtime checks on Vercel.")
        return {"action": "allow", "skipped": "belfry_loopback"}
    if not BELFRY_API_KEY:
        print("[Belfry SDK] No API key; model call stopped (fail closed).")
        return {"action": "error", "error": "missing_api_key"}
    try:
        if _sdk_client is not None:
            if kind == "output":
                result = await _sdk_client.check_output(text, project_id=BELFRY_PROJECT_ID)
            else:
                result = await _sdk_client.check_input(text, project_id=BELFRY_PROJECT_ID)
            return _normalize(result)
        return await _check_http(kind, text)
    except Exception as e:
        print(f"[Belfry SDK] {kind} check failed; model call stopped (fail closed): {e}")
        return {"action": "error", "error": str(e)}


async def belfry_check_input(text: str) -> dict:
    """Check user input BEFORE sending to LLM using Belfry."""
    return await _check("input", text)


async def belfry_check_output(text: str) -> dict:
    """Check LLM output BEFORE returning to user using Belfry."""
    return await _check("output", text)


_PASS_ACTIONS = _ALLOW_ACTIONS | {"redact"}
REFUSAL = (
    "I can't help with that request. "
    "I can book a clinic appointment if you'd like."
)


def _logged(kind: str, result: dict) -> dict:
    action = str(result.get("action") or "error")
    print(
        f"[Belfry] {kind} action={action} "
        f"project={BELFRY_PROJECT_ID} tenant={BELFRY_TENANT_ID}"
    )
    return result


def belfry_requested(value) -> bool:
    """True unless the showcase toggle explicitly turned the SDK off."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "off", "no"}


async def screen_inbound(text: str, *, enabled: bool = True) -> tuple[bool, str, str]:
    """Send the patient message to Belfry before any model call.

    Returns (allowed, text, action). With the showcase toggle off, the
    message is passed through and the action is ``off``.
    """
    if not enabled:
        return True, text, "off"
    result = _logged("input", await belfry_check_input(text))
    action = str(result.get("action") or "error").lower()
    # A failed check is not a policy block. Only block/terminate stop the turn.
    if action == "error":
        return True, text, "unavailable"
    if action not in _PASS_ACTIONS:
        return False, REFUSAL, action
    if action == "redact" and result.get("processed_content"):
        return True, str(result["processed_content"]), action
    return True, text, action


async def screen_outbound(text: str, *, enabled: bool = True) -> tuple[str, str]:
    """Send the assistant reply to Belfry before it is shown or spoken."""
    if not enabled:
        return text, "off"
    result = _logged("output", await belfry_check_output(text))
    action = str(result.get("action") or "error").lower()
    if action == "error":
        return text, "unavailable"
    if action not in _PASS_ACTIONS:
        return REFUSAL, action
    if action == "redact" and result.get("processed_content"):
        return str(result["processed_content"]), action
    return text, action
