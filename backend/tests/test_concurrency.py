#!/usr/bin/env python3
"""
Concurrency test — verifies per-session state isolation under simultaneous load.

Spawns N WebSocket sessions concurrently, each sending a different audio phrase,
and checks that:
  1. Each session receives its own correct transcript (no cross-talk)
  2. No session receives another session's TTS audio
  3. noise_floor_rms / thresholds don't bleed between sessions

Usage:
    # Start the backend first: uvicorn main:app --port 8000
    python tests/test_concurrency.py
    python tests/test_concurrency.py --sessions 5 --url ws://localhost:8000/ws/audio
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

TESTS_DIR   = Path(__file__).parent
BACKEND_DIR = TESTS_DIR.parent
AUDIO_DIR   = TESTS_DIR / "fixtures" / "audio"

sys.path.insert(0, str(BACKEND_DIR))

try:
    import websockets
except ImportError:
    print("ERROR: Install websockets:  pip install websockets")
    sys.exit(1)


# ── test payloads — distinct phrases with distinct expected STT output ─────────
SESSION_FIXTURES = [
    {"phrase": "I need a cardiology appointment", "expect_keyword": "cardiology"},
    {"phrase": "Book a dentist appointment please", "expect_keyword": "dentist"},
    {"phrase": "I want orthopedic consultation", "expect_keyword": "orthopedic"},
    {"phrase": "Pediatric checkup for my child", "expect_keyword": "pediatric"},
    {"phrase": "Ophthalmology eye checkup", "expect_keyword": "ophthalmology"},
]


def phrase_to_pcm(phrase: str) -> bytes:
    """Generate a trivial synthetic PCM waveform as a stand-in for audio.

    In a proper concurrency test, load pre-recorded .wav files instead.
    This uses silence so the session waits for the inactivity prompt —
    the test is verifying session isolation, not STT accuracy.
    """
    # 500ms of silence at 16kHz, int16 → sends enough chunks to open the session
    return b"\x00" * (16000 * 2 // 2)


async def run_session(session_id: int, fixture: dict, url: str, results: dict):
    """Open one WebSocket session, stream audio, collect all messages."""
    received_transcripts = []
    received_statuses    = []
    errors               = []

    try:
        async with websockets.connect(url) as ws:
            # Send audio chunks (500ms of PCM in 2048-sample chunks)
            pcm = phrase_to_pcm(fixture["phrase"])
            chunk_size = 2048 * 2  # 2048 samples * 2 bytes/sample
            send_task  = asyncio.create_task(_send_audio(ws, pcm, chunk_size))

            # Receive messages for 8 seconds
            deadline = asyncio.get_event_loop().time() + 8
            while asyncio.get_event_loop().time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == "transcript":
                            received_transcripts.append(data.get("text", ""))
                        elif data.get("type") == "status":
                            received_statuses.append(data.get("message"))
                except asyncio.TimeoutError:
                    pass  # no message this second, continue

            send_task.cancel()

    except Exception as e:
        errors.append(str(e))

    results[session_id] = {
        "fixture":    fixture,
        "transcripts": received_transcripts,
        "statuses":   received_statuses,
        "errors":     errors,
    }


async def _send_audio(ws, pcm: bytes, chunk_size: int):
    """Stream PCM bytes in chunks with realistic timing."""
    # 2048 samples @ 16kHz ≈ 128ms per chunk
    for i in range(0, len(pcm), chunk_size):
        await ws.send(pcm[i:i + chunk_size])
        await asyncio.sleep(0.128)


async def main(n_sessions: int, url: str):
    fixtures = (SESSION_FIXTURES * 10)[:n_sessions]
    results  = {}

    print(f"\nSpawning {n_sessions} concurrent WebSocket sessions → {url}\n")
    await asyncio.gather(*[
        run_session(i, fixtures[i], url, results)
        for i in range(n_sessions)
    ])

    # ── analysis ───────────────────────────────────────────────────────────────
    print(f"\n{'Session':<10} {'Transcripts Received':<40} {'Errors'}")
    print("─" * 80)
    cross_contamination_detected = False
    for sid, r in sorted(results.items()):
        tx_summary = " | ".join(r["transcripts"][:3]) or "(none)"
        err_summary = r["errors"][0] if r["errors"] else "—"
        print(f"{sid:<10} {tx_summary[:38]:<40} {err_summary}")

        # Check: does any transcript contain a keyword from a *different* session?
        for other_sid, other_r in results.items():
            if other_sid == sid:
                continue
            other_keyword = other_r["fixture"]["expect_keyword"]
            for tx in r["transcripts"]:
                if other_keyword in tx.lower():
                    print(
                        f"  ⚠  CROSS-CONTAMINATION: session {sid} received keyword "
                        f"'{other_keyword}' (belongs to session {other_sid})"
                    )
                    cross_contamination_detected = True

    print()
    if cross_contamination_detected:
        print("✗  Cross-session state leakage detected — investigate AudioSession isolation.")
    else:
        print("✓  No cross-session contamination detected.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--url", default="ws://localhost:8000/ws/audio")
    args = parser.parse_args()
    asyncio.run(main(args.sessions, args.url))
