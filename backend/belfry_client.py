"""EchoPilot → Belfry Atlas runtime checks via the official Python SDK."""

import os
import sys
from pathlib import Path

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


def _normalize(result: dict) -> dict:
    action = str(result.get("action") or "allow").lower()
    result["action"] = action
    return result


async def _check_http(kind: str, text: str) -> dict:
    url = f"{BELFRY_BASE_URL}/runtime-safety/check/{kind}"
    payload = {
        "content": text,
        "tenant_id": BELFRY_TENANT_ID,
        "project_id": BELFRY_PROJECT_ID,
        "persist": True,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, headers=_headers(), json=payload)
        response.raise_for_status()
        return _normalize(response.json())


async def _check(kind: str, text: str) -> dict:
    """POST a runtime check that persists into Belfry Runtime Events."""
    if not text or not BELFRY_API_KEY:
        return {"action": "allow"}
    try:
        if _sdk_client is not None:
            if kind == "output":
                result = await _sdk_client.check_output(text, project_id=BELFRY_PROJECT_ID)
            else:
                result = await _sdk_client.check_input(text, project_id=BELFRY_PROJECT_ID)
            return _normalize(result)
        return await _check_http(kind, text)
    except Exception as e:
        print(f"[Belfry SDK] {kind} check warning: {e}")
        return {"action": "allow"}


async def belfry_check_input(text: str) -> dict:
    """Check user input BEFORE sending to LLM using Belfry."""
    return await _check("input", text)


async def belfry_check_output(text: str) -> dict:
    """Check LLM output BEFORE returning to user using Belfry."""
    return await _check("output", text)
