"""STT audio prep and transcript cleanup for the voice path."""
import io
import os
import re
import struct
from difflib import SequenceMatcher, get_close_matches
from typing import Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")
NO_SPEECH_PROB_THRESHOLD = 0.80
AVG_LOGPROB_THRESHOLD = -1.0

# Whisper prompt is a lexicon + prior transcript, not instructions.
# VoiceStudio does the same with Faster-Whisper initial_prompt + Silero VAD.
STT_PROMPT = (
    "Clinic appointment call. The caller name is often Abhilash. "
    "Words: cardiology, dermatology, dentist, checkup, tomorrow, Monday."
)

_FIELD_LEXICON = {
    "service": (
        "cardiology dermatology orthopedic pediatric ophthalmology dentist "
        "surgery consultation checkup hospital doctor appointment"
    ),
    "date": (
        "tomorrow today Monday Tuesday Wednesday Thursday Friday Saturday Sunday "
        "January February March April May June July August September October November December "
        "fifteenth sixteenth next week"
    ),
    "time": "ten eleven twelve one two three four thirty morning afternoon evening AM PM o'clock",
    "name": "Abhilash Asha Priya Rahul my name is",
    "phone": "nine eight seven six five four three two one zero phone number",
}

CLINIC_VOCAB = (
    "cardiology", "dermatology", "orthopedic", "pediatric", "ophthalmology",
    "dentist", "surgery", "consultation", "checkup", "appointment", "hospital",
    "doctor", "tomorrow", "today", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "morning", "afternoon", "evening",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "confirm", "cancel",
    "abhilash", "asha",
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
    (r"\bi\s*am\s+a\s+blush\b", "Abhilash"),
    (r"\bi['’]?m\s+a\s+blush\b", "Abhilash"),
    (r"\bi\s*am\s+a\s+blast\b", "Abhilash"),
    (r"\bi['’]?m\s+a\s+blast\b", "Abhilash"),
    (r"\ba\s+blush\b", "Abhilash"),
    (r"\bablush\b", "Abhilash"),
    (r"\bblush\b", "Abhilash"),
    (r"\babilash\b", "Abhilash"),
    (r"\bbilash\b", "Abhilash"),
    (r"\bi['’]?m a blast\b", "Abhilash"),
    (r"\ba blast\b", "Abhilash"),
    (r"\bi did\b(?=.*\b(appointment|surgery|doctor|checkup|hospital|book)\b)", "I need"),
    (r"\bend of the surgery\b", "I need the surgery"),
    (r"\bi need a (blast|blush)\b", "Abhilash"),
)


def trim_silence(pcm: np.ndarray, noise_floor: float, pad_ms: int = 300) -> np.ndarray:
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


def build_stt_prompt(recent_text: str = "", expected_field: Optional[str] = None) -> str:
    """Lexicon plus *user* turns only — assistant copy in the prompt causes echo hallucinations."""
    chunks = [STT_PROMPT]
    if expected_field and expected_field in _FIELD_LEXICON:
        chunks.append(_FIELD_LEXICON[expected_field])
    user_bits = []
    for line in (recent_text or "").split("User:"):
        line = line.split("Assistant:")[0].strip()
        if line:
            user_bits.append(line)
    if user_bits:
        last_user = user_bits[-1][-200:]
        if not is_hallucination(last_user):
            chunks.append(last_user)
    return " ".join(chunks)[:700]


def correct_clinic_words(text: str) -> str:
    """Snap long tokens to the clinic lexicon when Whisper is one typo off."""
    if not text:
        return ""
    parts = re.findall(r"[A-Za-z']+|\d+|[^\w\s]|\s+", text)
    out = []
    for part in parts:
        if not re.fullmatch(r"[A-Za-z']{5,}", part):
            out.append(part)
            continue
        lower = part.lower()
        if lower in CLINIC_VOCAB:
            out.append(part)
            continue
        match = get_close_matches(lower, CLINIC_VOCAB, n=1, cutoff=0.84)
        if match:
            fixed = match[0]
            if part[0].isupper():
                fixed = fixed.title()
            out.append(fixed)
        else:
            out.append(part)
    return "".join(out)


def rewrite_known_names(text: str) -> str:
    """Map Whisper's Indian-English mishears of Abhilash before NLU sees them."""
    if not text:
        return ""
    stripped = text.strip().rstrip(".!?,")
    if re.fullmatch(
        r"(?:i\s*(?:am|'m)\s+)?(?:a\s+)?(?:blush|blast|ablush|abilash|abhilash|bilash)",
        stripped,
        flags=re.IGNORECASE,
    ):
        return "Abhilash"
    return text


def cleanup_transcript(text: str, expected_field: Optional[str] = None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip(" \"'")
    for pattern, replacement in _DOMAIN_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = correct_clinic_words(cleaned)
    cleaned = rewrite_known_names(cleaned)
    if expected_field == "name":
        cleaned = rewrite_known_names(cleaned)
    return cleaned.strip()


def is_low_confidence(no_speech_prob: Optional[float], avg_logprob: Optional[float]) -> bool:
    noisy = no_speech_prob is not None and no_speech_prob >= NO_SPEECH_PROB_THRESHOLD
    weak = avg_logprob is not None and avg_logprob <= AVG_LOGPROB_THRESHOLD
    return bool(noisy and weak)


def pick_stt_language(recent_text: str) -> Optional[str]:
    """Pin English unless the call is already in Telugu — auto-detect is worse on Indian English."""
    if any("\u0c00" <= ch <= "\u0c7f" for ch in recent_text or ""):
        return None
    return "en"


IGNORE_UTTERANCES = {
    "no problem", "no problems", "no problem at all", "okay", "ok", "cool",
    "cooldown", "cool down", "take your time", "i'm still here",
    "whenever you're ready", "happy to help", "got it", "sure", "alright",
    "all right", "mm", "mm-hmm", "mhm",
}

HALLUCINATION_SUBSTRINGS = (
    "thank you for watching",
    "subscribe to our",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "thank you and thank others",
    "субтитри",
    "субтитры",
)
HALLUCINATION_EXACT = {
    "hmm", "hmm.", "uh.", "uh", "um.", "um",
    "you", "you.", "дякую", "дякую.",
    "thank you", "thanks", "thanks.", "thank you.",
    "thank you so much", "thank you very much", "thanks a lot",
    "thx", "okay thank you", "ok thank you",
}

# Whisper often dumps a lone "Thank you" onto silence or speaker bleed.
_THANK_YOU_ONLY = re.compile(
    r"^(?:ok(?:ay)?[,.\s]+)?(?:thanks|thank you)(?:\s+(?:so much|very much|a lot))?[.!?]*$",
    re.IGNORECASE,
)


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", (text or "").lower()))


def is_echo_transcript(user_text: str, assistant_text: str) -> bool:
    """True when Whisper transcribed Elena's speakers (or her last line) as the user."""
    user = (user_text or "").strip()
    assistant = (assistant_text or "").strip()
    if not user or not assistant:
        return False
    if SequenceMatcher(None, user.lower(), assistant.lower()).ratio() >= 0.42:
        return True
    uw, aw = _word_set(user), _word_set(assistant)
    if not uw:
        return True
    overlap = uw & aw
    return len(overlap) >= 3 and (len(overlap) / len(uw)) >= 0.45


def is_ignore_utterance(text: str) -> bool:
    t = re.sub(r"[.!?]+$", "", (text or "").strip().lower())
    t = re.sub(r"\s+", " ", t)
    if t in IGNORE_UTTERANCES:
        return True
    if t in HALLUCINATION_EXACT or len(t) < 2:
        return True
    if _THANK_YOU_ONLY.match(t):
        return True
    return any(p in t for p in HALLUCINATION_SUBSTRINGS)


def is_repeat_transcript(user_text: str, last_user_text: str) -> bool:
    """Drop the same STT phrase fired twice in a row (common Whisper stuck-phrase)."""
    a = re.sub(r"[.!?]+$", "", (user_text or "").strip().lower())
    b = re.sub(r"[.!?]+$", "", (last_user_text or "").strip().lower())
    return bool(a) and a == b


def is_hallucination(text: str) -> bool:
    if not text:
        return True
    t_lower = text.lower().strip()
    has_cyrillic = any("\u0400" <= char <= "\u04ff" for char in text)
    return (
        is_ignore_utterance(text)
        or has_cyrillic
        or t_lower in HALLUCINATION_EXACT
        or bool(_THANK_YOU_ONLY.match(t_lower))
        or any(p in t_lower for p in HALLUCINATION_SUBSTRINGS)
    )


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Build a mono 16-bit WAV in memory (no wave.Wave_read/write typing issues)."""
    samples = np.asarray(pcm, dtype=np.int16)
    raw = samples.tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(raw)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(raw)))
    buf.write(raw)
    return buf.getvalue()


def webrtc_speech_in_tail(pcm: np.ndarray, window_ms: int = 300) -> Optional[bool]:
    """Return True/False if webrtcvad is installed, else None so RMS VAD can run."""
    try:
        import webrtcvad  # type: ignore
    except ImportError:
        return None
    samples = np.asarray(pcm, dtype=np.int16)
    frame = int(SAMPLE_RATE * 0.03)
    if samples.size < frame:
        return None
    window = max(frame, int(SAMPLE_RATE * (window_ms / 1000.0)))
    tail = samples[-window:]
    vad = webrtcvad.Vad(2)
    voiced = 0
    total = 0
    for i in range(0, len(tail) - frame + 1, frame):
        chunk = tail[i : i + frame].tobytes()
        if len(chunk) != frame * 2:
            continue
        total += 1
        try:
            if vad.is_speech(chunk, SAMPLE_RATE):
                voiced += 1
        except Exception:
            return None
    if total == 0:
        return None
    return voiced / total >= 0.4


def groq_segment_confidence(transcription) -> Tuple[Optional[float], Optional[float]]:
    segments = getattr(transcription, "segments", None) or []
    if not segments:
        return None, None
    first = segments[0]
    if isinstance(first, dict):
        return first.get("no_speech_prob"), first.get("avg_logprob")
    return getattr(first, "no_speech_prob", None), getattr(first, "avg_logprob", None)
