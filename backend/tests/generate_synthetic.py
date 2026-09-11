#!/usr/bin/env python3
"""
Synthetic fixture generator — bootstraps the test corpus with TTS-generated audio.

Because it uses the bot's own Edge TTS to generate audio, synthetic fixtures
test the STT decoder in isolation from human variability (accent, pace, noise).
They are NOT a substitute for real recordings but let you run the full
regression suite immediately before you've recorded anything.

Usage:
    python tests/generate_synthetic.py

Output:
    tests/fixtures/audio/<id>.wav   (16kHz mono WAV, same format the pipeline expects)

Requirements (beyond main backend deps):
    pip install miniaudio   # pure-Python, no ffmpeg needed
"""

import asyncio
import json
import os
import site
import sys
import wave
from pathlib import Path

TESTS_DIR   = Path(__file__).parent
BACKEND_DIR = TESTS_DIR.parent
AUDIO_DIR   = TESTS_DIR / "fixtures" / "audio"
CORPUS_PATH = TESTS_DIR / "corpus.json"

sys.path.insert(0, str(BACKEND_DIR))

# ── Windows CUDA DLL fix (mirrors main.py startup) ────────────────────────────
# Without this, importing faster-whisper in the test script raises
# "cublas64_12.dll is not found or cannot be loaded" on Windows.
if os.name == "nt":
    for site_dir in site.getsitepackages():
        for lib in ["cublas", "cudnn"]:
            bin_path = os.path.join(site_dir, "nvidia", lib, "bin")
            if os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(bin_path)

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

from tts import synthesize  # Edge TTS — returns MP3 bytes

# ── miniaudio for MP3 → PCM conversion (pure-Python, no ffmpeg needed) ────────
try:
    import miniaudio
except ImportError:
    print(
        "ERROR: miniaudio is required to decode Edge TTS MP3 output.\n"
        "  pip install miniaudio"
    )
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────
SYNTHETIC_SKIP_CATEGORIES = {"noisy", "device"}
SILENCE_GAP_SECONDS = 1.5   # pause injected at '...' markers for paused fixtures


# ── audio helpers ─────────────────────────────────────────────────────────────
def mp3_bytes_to_pcm16_mono_16k(mp3_bytes: bytes) -> bytes:
    """Decode MP3 bytes (including ID3 watermark) to raw int16 mono PCM at 16kHz.
    Uses miniaudio — pure-Python binary wheel, no ffmpeg required.
    """
    decoded = miniaudio.decode(
        mp3_bytes,
        nchannels=1,
        sample_rate=16000,
        output_format=miniaudio.SampleFormat.SIGNED16,
    )
    return bytes(decoded.samples)


def silence_pcm(seconds: float) -> bytes:
    """Return silence as int16 mono 16kHz raw PCM bytes."""
    return b"\x00" * int(16000 * 2 * seconds)


def write_wav(path: Path, pcm: bytes, rate: int = 16000):
    """Write raw int16 mono PCM as a 16kHz WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


# ── per-fixture generation ────────────────────────────────────────────────────
async def generate_fixture(fixture: dict) -> bool:
    """Generate synthetic audio for one fixture. Returns True if newly generated."""
    if fixture.get("category") in SYNTHETIC_SKIP_CATEGORIES:
        return False

    out_path = AUDIO_DIR / fixture["file"]
    if out_path.exists():
        print(f"  [skip] {fixture['id']}  (file already exists)")
        return False

    transcript = fixture.get("expected_transcript", "").strip()

    # Silence fixtures: write raw silence directly — no TTS needed
    if not transcript:
        duration = 7.0 if "long" in fixture.get("subcategory", "") else 3.0
        write_wav(out_path, silence_pcm(duration))
        print(f"  [gen]  {fixture['id']}  -> {duration}s silence")
        return True

    # Paused fixtures: split on '...' and insert silence gaps between segments
    if "..." in transcript:
        parts = [p.strip().rstrip(".").strip() for p in transcript.split("...") if p.strip()]
        pcm_parts = []
        for part in parts:
            mp3 = await synthesize(part)
            if not mp3:
                print(f"  [ERR]  {fixture['id']}: TTS returned empty for '{part}'")
                return False
            pcm_parts.append(mp3_bytes_to_pcm16_mono_16k(mp3))
            pcm_parts.append(silence_pcm(SILENCE_GAP_SECONDS))
        write_wav(out_path, b"".join(pcm_parts))
        print(f"  [gen]  {fixture['id']}  -> {len(parts)} segments + {SILENCE_GAP_SECONDS}s pauses")
        return True

    # Normal fixture: synthesize the whole transcript as one phrase
    mp3 = await synthesize(transcript)
    if not mp3:
        print(f"  [ERR]  {fixture['id']}: TTS returned empty bytes")
        return False

    pcm = mp3_bytes_to_pcm16_mono_16k(mp3)
    write_wav(out_path, pcm)
    print(f"  [gen]  {fixture['id']}  -> '{transcript[:60]}'")
    return True


# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating synthetic fixtures -> {AUDIO_DIR}\n")
    print("NOTE: TTS-generated audio (Edge TTS MP3 -> 16kHz WAV), not real human speech.")
    print("      Replace with real recordings for meaningful STT accuracy numbers.\n")

    generated = skipped_existing = skipped_category = errors = 0

    for fixture in corpus:
        cat = fixture.get("category", "")
        if cat in SYNTHETIC_SKIP_CATEGORIES:
            print(f"  [skip] {fixture['id']}  (category={cat} -- needs real recording)")
            skipped_category += 1
            continue
        try:
            did_gen = await generate_fixture(fixture)
            if did_gen:
                generated += 1
            else:
                skipped_existing += 1
        except Exception as e:
            print(f"  [ERR]  {fixture['id']}: {e}")
            errors += 1

    print(
        f"\nDone.  generated={generated}  skipped={skipped_existing}  "
        f"needs_recording={skipped_category}  errors={errors}"
    )
    if errors:
        print("  Check that ffmpeg is on PATH (required by pydub to decode MP3).")
        print("  Install: winget install Gyan.FFmpeg  (then restart terminal)")
    print("\nRun the test suite:\n  python tests/test_pipeline.py\n")


if __name__ == "__main__":
    asyncio.run(main())
