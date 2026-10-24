"""Pluggable STT engines: Groq Whisper Large → NVIDIA Nemotron 3.5 → local Faster-Whisper.

Whisper Large stays the quality pass for complete utterances. Nemotron is the
streaming-capable fallback (NIM / OpenAI-compatible HTTP). Local Faster-Whisper
is the offline last resort (prefer distil-large-v3 / large-v3 over tiny/small).
"""
from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from stt import (
    STT_MODEL,
    STT_PROMPT,
    groq_segment_confidence,
    is_low_confidence,
    pcm_to_wav_bytes,
)

GROQ_FAILURE_THRESHOLD = 3
GROQ_CIRCUIT_OPEN_SECONDS = 30
GROQ_TIMEOUT_SECONDS = 4.0
NEMO_TIMEOUT_SECONDS = float(os.getenv("NEMO_TIMEOUT_SECONDS", "6.0"))
STT_LOCAL_MODEL = os.getenv("STT_LOCAL_MODEL", "distil-large-v3")

_groq_failure_count = 0
_groq_circuit_open_until = 0.0
_local_whisper_model = None
_local_whisper_tried = False


@dataclass
class TranscriptResult:
    text: str
    backend: str


def groq_circuit_open() -> bool:
    return time.time() < _groq_circuit_open_until


def list_stt_engines() -> dict:
    return {
        "groq": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "nemotron": bool(
            os.getenv("NVIDIA_API_KEY", "").strip()
            and os.getenv("NVIDIA_ASR_URL", "").strip()
        ),
        "local": True,
        "local_model": STT_LOCAL_MODEL,
        "groq_circuit_open": groq_circuit_open(),
    }


def _note_groq_failure(err: Exception) -> None:
    global _groq_failure_count, _groq_circuit_open_until
    _groq_failure_count += 1
    remaining = GROQ_FAILURE_THRESHOLD - _groq_failure_count
    if _groq_failure_count >= GROQ_FAILURE_THRESHOLD:
        _groq_circuit_open_until = time.time() + GROQ_CIRCUIT_OPEN_SECONDS
        print(
            f"[STT] Groq circuit breaker OPEN after {_groq_failure_count} failures "
            f"— Nemotron/local fallback for {GROQ_CIRCUIT_OPEN_SECONDS}s: {err}"
        )
    else:
        print(
            f"[STT] Groq error ({_groq_failure_count}/{GROQ_FAILURE_THRESHOLD}) "
            f"— {remaining} until circuit opens: {err}"
        )


def _note_groq_success() -> None:
    global _groq_failure_count
    _groq_failure_count = 0


def _close_circuit_if_due() -> None:
    global _groq_failure_count
    if _groq_failure_count >= GROQ_FAILURE_THRESHOLD and time.time() >= _groq_circuit_open_until:
        _groq_failure_count = 0
        print("[STT] Groq circuit breaker CLOSED — will retry Groq next turn")


def get_local_whisper_model():
    """Lazy-load Faster-Whisper. Prefer distil-large-v3, then fall back to small."""
    global _local_whisper_model, _local_whisper_tried
    if _local_whisper_model is not None or _local_whisper_tried:
        return _local_whisper_model
    _local_whisper_tried = True
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[STT] faster-whisper not installed — local fallback unavailable")
        return None

    candidates = [STT_LOCAL_MODEL]
    if STT_LOCAL_MODEL not in ("small", "base"):
        candidates.append("small")

    for name in candidates:
        for device, compute in (("cuda", "float16"), ("cpu", "int8")):
            try:
                _local_whisper_model = WhisperModel(name, device=device, compute_type=compute)
                print(f"[STT] Local Faster-Whisper loaded: {name} {device}/{compute}")
                return _local_whisper_model
            except Exception as exc:
                print(f"[STT] Local Whisper {name} {device} failed: {exc}")
    return None


def _transcribe_groq_sync(pcm: np.ndarray, language: Optional[str], groq_client, prompt: str) -> str:
    wav_io = io.BytesIO(pcm_to_wav_bytes(pcm))
    wav_io.name = "audio.wav"
    stt_prompt = prompt or STT_PROMPT

    def _call(response_format: str):
        wav_io.seek(0)
        kwargs = {
            "file": wav_io,
            "model": STT_MODEL,
            "temperature": 0,
            "prompt": stt_prompt,
            "timeout": GROQ_TIMEOUT_SECONDS,
            "response_format": response_format,
        }
        if language:
            kwargs["language"] = language
        return groq_client.audio.transcriptions.create(**kwargs)

    try:
        transcription = _call("verbose_json")
    except Exception:
        wav_io.seek(0)
        transcription = _call("json")

    no_speech_prob, avg_logprob = groq_segment_confidence(transcription)
    if is_low_confidence(no_speech_prob, avg_logprob):
        print(f"[STT] Low confidence drop (no_speech={no_speech_prob}, avg_logprob={avg_logprob})")
        return ""
    return (getattr(transcription, "text", None) or "").strip()


def _transcribe_nemotron_sync(pcm: np.ndarray, language: Optional[str]) -> str:
    url = os.getenv("NVIDIA_ASR_URL", "").strip().rstrip("/")
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not url or not key:
        return ""
    try:
        from openai import OpenAI
    except ImportError:
        return ""

    model = os.getenv("NVIDIA_ASR_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
    wav_io = io.BytesIO(pcm_to_wav_bytes(pcm))
    wav_io.name = "audio.wav"
    client = OpenAI(base_url=url, api_key=key)
    kwargs = {
        "file": wav_io,
        "model": model,
        "temperature": 0,
        "timeout": NEMO_TIMEOUT_SECONDS,
    }
    # Telugu is not in Nemotron 3.5 locales — skip forcing te, use auto.
    if language and language != "te":
        kwargs["language"] = language
    try:
        result = client.audio.transcriptions.create(**kwargs)
    except Exception as exc:
        print(f"[STT:nemotron] {exc}")
        return ""
    return (getattr(result, "text", None) or "").strip()


def _transcribe_local_sync(pcm: np.ndarray, language: Optional[str], prompt: str = "") -> str:
    model = get_local_whisper_model()
    if not model:
        return ""
    pcm_float = np.asarray(pcm, dtype=np.float32) / 32768.0
    local_kwargs = {
        "temperature": 0,
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 250,
        },
        "beam_size": 5,
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4,
        "no_speech_threshold": 0.6,
        "initial_prompt": prompt or STT_PROMPT,
    }
    if language:
        local_kwargs["language"] = language
    segments, _info = model.transcribe(pcm_float, **local_kwargs)
    segment_list = list(segments)
    text = " ".join(seg.text.strip() for seg in segment_list).strip()
    if segment_list:
        no_speech_prob = getattr(segment_list[0], "no_speech_prob", None)
        avg_logprob = getattr(segment_list[0], "avg_logprob", None)
        if is_low_confidence(no_speech_prob, avg_logprob):
            return ""
    return text


_MEDIA_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "webm": "video/webm",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
}


def transcribe_media_bytes(data: bytes, filename: str = "speech.wav") -> str:
    """Transcribe an uploaded audio or video file.

    Same order Belfry uses: Groq whisper-large-v3 on the file bytes, then
    local Faster-Whisper. Video containers are read for their soundtrack.
    An empty Groq transcript is silence, not a failure.
    """
    if not data:
        return ""
    name = filename or "speech.wav"
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        try:
            return _transcribe_groq_file(data, name, key)
        except Exception as exc:
            print(f"[STT] Groq file transcription failed for {name}: {exc}")
    if os.getenv("VERCEL"):
        return ""
    return _transcribe_local_file(data, name)


def _wav_is_silence(data: bytes) -> bool:
    """Skip a quiet WAV before Whisper, which invents phrases on silence."""
    if not (data.startswith(b"RIFF") and len(data) > 44 and data[8:12] == b"WAVE"):
        return False
    offset = 12
    pcm = b""
    while offset + 8 <= len(data):
        chunk = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        start = offset + 8
        if chunk == b"data":
            pcm = data[start : start + size]
            break
        offset = start + size + (size % 2)
    if len(pcm) < 2:
        return True
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype=np.int16)
    if samples.size < 1600:
        return True
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
    return rms < 220.0


def _transcribe_groq_file(data: bytes, filename: str, key: str) -> str:
    from openai import OpenAI

    if _wav_is_silence(data):
        print(f"[STT] Skipped silent upload {filename}")
        return ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
    mime = _MEDIA_MIME.get(ext, "application/octet-stream")
    audio = io.BytesIO(data)
    audio.name = filename
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key, timeout=12.0)
    result = client.audio.transcriptions.create(
        file=(filename, audio, mime),
        model=STT_MODEL,
        temperature=0,
        language="en",
        prompt=STT_PROMPT,
        response_format="json",
    )
    return (getattr(result, "text", None) or "").strip()


def _transcribe_local_file(data: bytes, filename: str) -> str:
    model = get_local_whisper_model()
    if model is None:
        return ""
    suffix = Path(filename).suffix or ".wav"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = handle.name
    try:
        handle.write(data)
        handle.close()
        segments, _info = model.transcribe(
            path,
            temperature=0,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text).strip()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def transcribe_utterance(
    pcm: np.ndarray,
    *,
    language: Optional[str],
    groq_client=None,
    prompt: Optional[str] = None,
) -> TranscriptResult:
    """Quality-first chain: Groq Whisper Large → Nemotron 3.5 → local Whisper."""
    _close_circuit_if_due()
    stt_prompt = prompt or STT_PROMPT

    if groq_client and not groq_circuit_open():
        try:
            text = await asyncio.to_thread(
                _transcribe_groq_sync, pcm, language, groq_client, stt_prompt
            )
            _note_groq_success()
            # Empty transcript is a successful Groq call — do not wait on Nemotron/local.
            return TranscriptResult(text=text or "", backend="groq")
        except Exception as exc:
            _note_groq_failure(exc)
    elif groq_client and groq_circuit_open():
        secs = int(_groq_circuit_open_until - time.time())
        print(f"[STT] Groq circuit OPEN — Nemotron/local ({secs}s remaining)")

    if os.getenv("NVIDIA_API_KEY", "").strip() and os.getenv("NVIDIA_ASR_URL", "").strip():
        try:
            text = await asyncio.to_thread(_transcribe_nemotron_sync, pcm, language)
            if text:
                return TranscriptResult(text=text, backend="nemotron")
        except Exception as exc:
            print(f"[STT] Nemotron fallback failed: {exc}")

    try:
        text = await asyncio.to_thread(_transcribe_local_sync, pcm, language, stt_prompt)
        if text:
            return TranscriptResult(text=text, backend="local")
    except Exception as exc:
        print(f"[STT] Local Whisper failed: {exc}")

    return TranscriptResult(text="", backend="none")
