import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Ensure .env in backend directory is always loaded regardless of current working directory
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)
load_dotenv()  # also load root .env if present

BELFRY_BASE_URL = os.getenv("BELFRY_BASE_URL", "http://localhost:8001/api/v1").rstrip("/")
BELFRY_TENANT_ID = os.getenv("BELFRY_TENANT_ID", "belfry-local")
BELFRY_PROJECT_ID = os.getenv("BELFRY_PROJECT_ID", "proj_5e0b58be7d5a")
BELFRY_API_KEY = (os.getenv("BELFRY_API_KEY") or os.getenv("BELFRY_LABS_API_KEY", "")).strip()

_sdk_client = None
if BELFRY_API_KEY:
    try:
        from belfry_labs import AsyncBelfryLabsClient
        _sdk_client = AsyncBelfryLabsClient(
            api_key=BELFRY_API_KEY,
            base_url=BELFRY_BASE_URL,
            tenant_id=BELFRY_TENANT_ID,
        )
        # Docker Belfry authenticates access keys via X-API-Key, not Bearer JWT.
        _sdk_client._client.headers["X-API-Key"] = BELFRY_API_KEY
    except ImportError:
        _sdk_client = None
else:
    print("[Belfry SDK] Notice: BELFRY_API_KEY is not configured; SDK calls will bypass gracefully.")


def _headers() -> dict:
    return {
        "X-API-Key": BELFRY_API_KEY,
        "X-Tenant-ID": BELFRY_TENANT_ID,
        "Content-Type": "application/json",
    }


async def _check(kind: str, text: str) -> dict:
    """POST a runtime check that persists into Belfry Runtime Events."""
    if not text or not BELFRY_API_KEY:
        return {"action": "allow"}
    url = f"{BELFRY_BASE_URL}/runtime-safety/check/{kind}"
    payload = {
        "content": text,
        "tenant_id": BELFRY_TENANT_ID,
        "project_id": BELFRY_PROJECT_ID,
        "persist": True,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=_headers(), json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"[Belfry SDK] {kind} check warning: {e}")
        return {"action": "allow"}


async def belfry_check_input(text: str) -> dict:
    """Check user input BEFORE sending to LLM using Belfry."""
    return await _check("input", text)


async def belfry_check_output(text: str) -> dict:
    """Check LLM output BEFORE returning to user using Belfry."""
    return await _check("output", text)
