"""STT audio prep and transcript cleanup for the voice path."""
import os
import re
from typing import Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")
NO_SPEECH_PROB_THRESHOLD = 0.80
AVG_LOGPROB_THRESHOLD = -1.0

STT_PROMPT = (
    "Meridian Health clinic phone call. The caller is booking an appointment. "
    "Words: cardiology, dermatology, orthopedic, pediatric, ophthalmology, dentist, "
    "checkup, tomorrow, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, "
    "morning, afternoon, evening, ten thirty, eleven AM, two PM, yes, no, confirm, cancel."
)

_DOMAIN_FIXES = (
    (r"\bopthalmology\b", "ophthalmology"),
    (r"\bopthalmologist\b", "ophthalmology"),
    (r"\bdermatologist appointment\b", "dermatology appointment"),
    (r"\bortho\s*peedic\b", "orthopedic"),
    (r"\bortho\s*paedic\b", "orthopedic"),
    (r"\bpaediatric\b", "pediatric"),
    (r"\bto\s*morrow\b", "tomorrow"),
    (r"\bday after to\s*morrow\b", "day after tomorrow"),
    (r"\bcheck[\s-]*up\b", "checkup"),
    (r"\ba\.?\s*m\.?\b", "AM"),
    (r"\bp\.?\s*m\.?\b", "PM"),
)


def trim_silence(pcm: np.ndarray, noise_floor: float, pad_ms: int = 80) -> np.ndarray:
    """Drop leading/trailing noise frames, keep a short pad so word edges survive."""
    if pcm is None or len(pcm) == 0:
        return np.array([], dtype=np.int16)

    samples = np.asarray(pcm, dtype=np.int16)
    frame = int(SAMPLE_RATE * 0.03)
    if len(samples) < frame:
        return samples

    threshold = max(float(noise_floor) * 1.6, 180.0)
    frames = len(samples) // frame
    rms = np.sqrt(
        np.mean(
            samples[: frames * frame].reshape(frames, frame).astype(np.float32) ** 2,
            axis=1,
        )
    )
    voiced = np.flatnonzero(rms >= threshold)
    if voiced.size == 0:
        return samples

    pad = int(SAMPLE_RATE * (pad_ms / 1000.0))
    start = max(0, int(voiced[0]) * frame - pad)
    end = min(len(samples), int(voiced[-1] + 1) * frame + pad)
    return samples[start:end]


def normalize_pcm(pcm: np.ndarray, target_rms: float = 2500.0, max_gain: float = 6.0) -> np.ndarray:
    """Lift quiet mics toward a Whisper-friendly level without clipping."""
    samples = np.asarray(pcm, dtype=np.float32)
    if samples.size == 0:
        return np.array([], dtype=np.int16)

    samples = samples - float(np.mean(samples))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 80.0:
        return np.clip(samples, -32768, 32767).astype(np.int16)
    if rms < target_rms:
        samples = samples * min(target_rms / rms, max_gain)
    return np.clip(samples, -32768, 32767).astype(np.int16)


def prepare_utterance(pcm: np.ndarray, noise_floor: float) -> np.ndarray:
    return normalize_pcm(trim_silence(pcm, noise_floor), target_rms=2500.0)


def cleanup_transcript(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip(" \"'")
    for pattern, replacement in _DOMAIN_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def is_low_confidence(no_speech_prob: Optional[float], avg_logprob: Optional[float]) -> bool:
    noisy = no_speech_prob is not None and no_speech_prob >= NO_SPEECH_PROB_THRESHOLD
    weak = avg_logprob is not None and avg_logprob <= AVG_LOGPROB_THRESHOLD
    return bool(noisy and weak)


def pick_stt_language(recent_text: str) -> Optional[str]:
    """Pin English unless the call is already in Telugu — auto-detect is worse on Indian English."""
    if any("\u0c00" <= ch <= "\u0c7f" for ch in recent_text or ""):
        return None
    return "en"


def groq_segment_confidence(transcription) -> Tuple[Optional[float], Optional[float]]:
    segments = getattr(transcription, "segments", None) or []
    if not segments:
        return None, None
    first = segments[0]
    if isinstance(first, dict):
        return first.get("no_speech_prob"), first.get("avg_logprob")
    return getattr(first, "no_speech_prob", None), getattr(first, "avg_logprob", None)
