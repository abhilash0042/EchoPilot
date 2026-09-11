import edge_tts
import tempfile
import os
import asyncio
from typing import Optional

# Indian English & Telugu Expressive Neural Voices
VOICE_EN = "en-IN-NeerjaNeural"  # Professional, natural human Indian English voice
VOICE_TE = "te-IN-ShrutiNeural"  # Fluent, natural Telugu voice

# ---------------------------------------------------------------------------
# TTS Phrase Cache
# ---------------------------------------------------------------------------
# Key: (normalized_text, voice) — avoids API round-trips for repeated phrases.
# Common case: all fixed dialogue prompts (greetings, slot-collection lines,
# inactivity check-ins) hit cache after the first call or after warm_cache().
# Value: MP3 bytes (already watermarked).
_phrase_cache: dict[tuple[str, str], bytes] = {}

def _cache_key(text: str, voice: str) -> tuple[str, str]:
    """Normalize cache key — strips surrounding whitespace, lowercases for lookup."""
    return (text.strip().lower(), voice)

async def warm_cache(phrases: list[str]) -> None:
    """Pre-synthesize a list of known static phrases at startup.

    Populates _phrase_cache so the first real call to synthesize() for any
    of these texts is a cache hit with zero API latency. Should be called
    once from the FastAPI startup handler.
    """
    tasks = [synthesize(phrase) for phrase in phrases if phrase and phrase.strip()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    hits = sum(1 for r in results if isinstance(r, bytes) and r)
    print(f"[TTS] Cache warm: {hits}/{len(phrases)} phrases pre-synthesized")


def has_telugu(text: str) -> bool:
    """Check if text contains Telugu script characters (\u0c00 - \u0c7f)."""
    return any('\u0c00' <= char <= '\u0c7f' for char in text)

def inject_ai_watermark(audio_bytes: bytes) -> bytes:
    """Injects an ID3v2 metadata header into the MP3 stream to declare it as machine-generated AI speech."""
    if not audio_bytes or audio_bytes.startswith(b"ID3"):
        return audio_bytes
        
    artist = b"EchoPilot Elena AI"
    title = b"AI Generated Spoken Audio"
    
    def make_frame(frame_id: bytes, content: bytes) -> bytes:
        payload = b"\x00" + content
        size = len(payload).to_bytes(4, byteorder='big')
        return frame_id + size + b"\x00\x00" + payload
        
    f_artist = make_frame(b"TPE1", artist)
    f_title = make_frame(b"TIT2", title)
    frames = f_artist + f_title
    
    tag_size = len(frames)
    s1 = (tag_size >> 21) & 0x7F
    s2 = (tag_size >> 14) & 0x7F
    s3 = (tag_size >> 7) & 0x7F
    s4 = tag_size & 0x7F
    
    header = b"ID3\x03\x00\x00" + bytes([s1, s2, s3, s4])
    return header + frames + audio_bytes


# ---------------------------------------------------------------------------
# TTS synthesis — 3-tier priority chain
# ---------------------------------------------------------------------------
# Tier 1: Azure Cognitive Services (if AZURE_SPEECH_KEY env var set)
#         Official, licensed, same neural voices, SLA-backed.
# Tier 2: edge-tts (always available as a middle tier)
#         Free but no SLA — Microsoft can change internal auth without notice.
# Tier 3: Piper local TTS (if piper-tts installed)
#         Fully offline, no network dependency.
#
# The phrase cache sits in front of all three tiers.
# ---------------------------------------------------------------------------

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus").strip()

# Lazy-init Azure SDK (optional dep — only import if key is set)
_azure_synthesizer_cache: dict[str, object] = {}

def _get_azure_synthesizer(voice: str) -> Optional[object]:
    """Returns a cached Azure SpeechSynthesizer for the given voice, or None if unavailable."""
    if not AZURE_SPEECH_KEY:
        return None
    try:
        import azure.cognitiveservices.speech as speechsdk
        if voice not in _azure_synthesizer_cache:
            speech_config = speechsdk.SpeechConfig(
                subscription=AZURE_SPEECH_KEY,
                region=AZURE_SPEECH_REGION
            )
            speech_config.speech_synthesis_voice_name = voice
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
            )
            # Use PullAudioOutputStream to capture bytes in-memory (no file I/O)
            _azure_synthesizer_cache[voice] = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=None  # null config = in-memory output
            )
        return _azure_synthesizer_cache[voice]
    except ImportError:
        return None
    except Exception as e:
        print(f"[TTS:azure] Synthesizer init error: {e}")
        return None


async def _synthesize_azure(text: str, voice: str) -> bytes:
    """Synthesize using Azure Cognitive Services Speech SDK (Tier 1)."""
    synthesizer = _get_azure_synthesizer(voice)
    if synthesizer is None:
        return b""
    try:
        import azure.cognitiveservices.speech as speechsdk
        # Azure SDK is synchronous — run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: synthesizer.speak_text_async(text).get()
        )
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return bytes(result.audio_data)
        else:
            cancellation = result.cancellation_details
            print(f"[TTS:azure] Synthesis failed: {cancellation.reason} — {cancellation.error_details}")
            return b""
    except Exception as e:
        print(f"[TTS:azure] Error: {e}")
        return b""


async def _synthesize_edge(text: str, voice: str) -> bytes:
    """Synthesize using edge-tts (Tier 2 — no SLA, free)."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        communicate = edge_tts.Communicate(text, voice, rate="+0%")
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[TTS:edge] Error with voice {voice}: {e}")
        return b""
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _synthesize_piper(text: str) -> bytes:
    """Synthesize using Piper local TTS (Tier 3 — fully offline fallback)."""
    try:
        from piper.voice import PiperVoice  # type: ignore[import]
        import wave, io as _io
        # Lazy-load Piper model from env (defaults to a lightweight en-US model)
        model_path = os.getenv("PIPER_MODEL_PATH", "")
        if not model_path or not os.path.exists(model_path):
            print("[TTS:piper] PIPER_MODEL_PATH not set or file missing — Piper skipped")
            return b""
        voice = PiperVoice.load(model_path)
        wav_buf = _io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            voice.synthesize(text, wf)
        # Piper outputs WAV — convert to raw bytes (browser can decode WAV via decodeAudioData)
        return wav_buf.getvalue()
    except ImportError:
        return b""
    except Exception as e:
        print(f"[TTS:piper] Error: {e}")
        return b""


async def synthesize(text: str) -> bytes:
    """Returns TTS audio bytes (MP3 or WAV) for the given text.

    Priority chain: phrase cache → Azure → edge-tts → Piper local.
    Returns empty bytes b"" only if all three tiers fail.
    """
    if not text or not text.strip():
        return b""

    voice = VOICE_TE if has_telugu(text) else VOICE_EN
    key = _cache_key(text, voice)

    # --- Cache check (all tiers) ---
    if key in _phrase_cache:
        print(f"[TTS:cache] hit — '{text[:60]}'")
        return _phrase_cache[key]

    audio_bytes = b""
    tier_used = "none"

    # --- Tier 1: Azure ---
    if AZURE_SPEECH_KEY:
        audio_bytes = await _synthesize_azure(text, voice)
        if audio_bytes:
            tier_used = "azure"

    # --- Tier 2: edge-tts ---
    if not audio_bytes:
        audio_bytes = await _synthesize_edge(text, voice)
        if not audio_bytes and voice != VOICE_EN:
            # Fallback to English neural voice if Telugu voice fails
            audio_bytes = await _synthesize_edge(text, VOICE_EN)
        if audio_bytes:
            tier_used = "edge"

    # --- Tier 3: Piper local ---
    if not audio_bytes:
        audio_bytes = await _synthesize_piper(text)
        if audio_bytes:
            tier_used = "piper"

    if not audio_bytes:
        print(f"[TTS] All tiers failed for text: '{text[:80]}'")
        return b""

    # Watermark MP3 output (skip for WAV from Piper)
    if tier_used in ("azure", "edge"):
        audio_bytes = inject_ai_watermark(audio_bytes)

    print(f"[TTS:{tier_used}] synthesized {len(audio_bytes)} bytes — '{text[:60]}'")

    # Populate cache for future calls
    _phrase_cache[key] = audio_bytes
    return audio_bytes
