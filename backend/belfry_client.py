import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env in backend directory is always loaded regardless of current working directory
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)
load_dotenv()  # also load root .env if present

BELFRY_BASE_URL = os.getenv("BELFRY_BASE_URL", "http://localhost:8001/api/v1")
BELFRY_TENANT_ID = os.getenv("BELFRY_TENANT_ID", "belfry-local")
BELFRY_PROJECT_ID = os.getenv("BELFRY_PROJECT_ID", "proj_5e0b58be7d5a")
BELFRY_API_KEY = (os.getenv("BELFRY_API_KEY") or os.getenv("BELFRY_LABS_API_KEY", "")).strip()

client = None
if BELFRY_API_KEY:
    try:
        from belfry_labs import AsyncBelfryLabsClient
        client = AsyncBelfryLabsClient(
            api_key=BELFRY_API_KEY,
            base_url=BELFRY_BASE_URL,
            tenant_id=BELFRY_TENANT_ID
        )
    except ImportError:
        client = None
else:
    print("[Belfry SDK] Notice: BELFRY_API_KEY is not configured; SDK calls will bypass gracefully.")

async def belfry_check_input(text: str) -> dict:
    """Check user input BEFORE sending to LLM using Belfry SDK."""
    if not text or client is None:
        return {"action": "allow"}
    try:
        return await client.check_input(text=text, project_id=BELFRY_PROJECT_ID)
    except Exception as e:
        print(f"[Belfry SDK] Input check warning: {e}")
        return {"action": "allow"}

async def belfry_check_output(text: str) -> dict:
    """Check LLM output BEFORE returning to user using Belfry SDK."""
    if not text or client is None:
        return {"action": "allow"}
    try:
        return await client.check_output(text=text, project_id=BELFRY_PROJECT_ID)
    except Exception as e:
        print(f"[Belfry SDK] Output check warning: {e}")
        return {"action": "allow"}
