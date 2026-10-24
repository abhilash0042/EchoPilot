import time
import asyncio
import numpy as np
import os
import site
import sys
import json
from typing import Optional, List, Dict
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

# Vercel loads this module from the repo root. Local imports expect backend/ on the path.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)
load_dotenv(override=True)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI


from dialogue_manager import handle_turn, handle_confirmation, PROMPTS
from attachments import AttachmentTooLarge, build_user_message, extract_upload
from belfry_client import (
    BELFRY_PROJECT_ID,
    BELFRY_TENANT_ID,
    belfry_requested,
    screen_inbound,
    screen_outbound,
    sdk_connected,
)
from booking import BookingSession, BookingState
from tts import synthesize, warm_cache, split_spoken_sentences
from extraction import add_to_history, clean_spoken_text
from stt import (
    build_stt_prompt,
    cleanup_transcript,
    is_echo_transcript,
    is_hallucination,
    is_ignore_utterance,
    is_repeat_transcript,
    pick_stt_language,
    prepare_utterance,
    spoken_booking_text,
    webrtc_speech_in_tail,
)
from speech_engines import list_stt_engines, transcribe_utterance
import database
from seed_data import seed_database

# --- Windows GPU DLL Fix ---
# Dynamically add pip-installed NVIDIA libraries to the Windows DLL search path
# so ctranslate2 can find cublas64_12.dll and cudnn64_8.dll
if os.name == 'nt':
    for site_dir in site.getsitepackages():
        for lib in ["cublas", "cudnn"]:
            bin_path = os.path.join(site_dir, "nvidia", lib, "bin")
            if os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(bin_path)
# ---------------------------

app = FastAPI(
    title="Meridian Health AI Voice Agent API",
    description="Healthcare voice assistant with SQLite database integration",
    version="2.0.0"
)

@app.get("/health")
async def health_check():
    from urllib.parse import urlparse
    import belfry_client
    host = urlparse(belfry_client.BELFRY_BASE_URL).hostname or ""
    return {
        "status": "healthy",
        "stt_engines": list_stt_engines(),
        "belfry": {
            "sdk_connected": sdk_connected(),
            "host": host,
            "project_id": BELFRY_PROJECT_ID,
            "tenant_id": BELFRY_TENANT_ID,
        },
    }

@app.get("/api/telemetry")
async def telemetry_check():
    return {"status": "ok"}

def _warm_image_reader() -> None:
    try:
        from image_ocr import warm
        warm()
    except Exception as exc:
        print(f"[Server] Image reader warmup skipped: {exc}")


# Auto-initialize database and sample data on startup
@app.on_event("startup")
async def on_startup():
    database.init_db()
    seed_database(force_refresh=False)
    print("[Server] Database initialized and verified ready.")

    # Pre-synthesize all fixed dialogue prompts so first call to each is a
    # cache hit with zero TTS API latency. Only dynamic content (names, dates,
    # booking confirmations) will hit the TTS API on first utterance.
    #
    # IMPORTANT: dialogue prompts are sourced directly from dialogue_manager.PROMPTS
    # (the single source of truth) so any copy edit to those strings is automatically
    # reflected here on next restart — no duplicate to update in main.py.
    static_tts_phrases = list(PROMPTS.values()) + [
        # Inactivity check-ins (defined here in main.py, not in PROMPTS)
        "Still there? Take your time, I'm listening!",
        "Hello! Can you hear me okay? Just let me know whenever you're ready!",
        "Take your time, I'm still here whenever you're ready.",
        # Common safety/fallback replies
        "I didn't quite catch that — could you say that again?",
        "I apologize, but I cannot process that request. How else may I assist you?",
    ]
    # Pre-warming talks to the TTS service. On Vercel that must not block the
    # first request, or the function startup hits the platform time limit.
    if os.getenv("VERCEL"):
        asyncio.create_task(warm_cache(static_tts_phrases))
        asyncio.create_task(asyncio.to_thread(_warm_image_reader))
    else:
        await warm_cache(static_tts_phrases)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://echopilot-voice-agent.onrender.com"
]

_allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if _allowed_origins_env:
    ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://[\w.-]+\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

groq_client: Optional[OpenAI] = None
if GROQ_API_KEY:
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
print("[STT] Connected to Groq Cloud API (whisper-large-v3)")

SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad requires 10/20/30ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per VAD frame

SILENCE_MS_TO_FINALIZE = int(os.getenv("SILENCE_MS_TO_FINALIZE", "800"))
MAX_UTTERANCE_BYTES = int(SAMPLE_RATE * 2 * int(os.getenv("MAX_UTTERANCE_SECONDS", "15")))

BARGE_IN_CONFIRM_MS = int(os.getenv("BARGE_IN_CONFIRM_MS", "550"))

class AudioSession:
    def __init__(self):
        self.pcm_buffer = bytearray()
        self.silence_ms = 0
        self.last_frame_had_speech = False

        # Barge-in state
        self.assistant_speaking = False
        self.barge_in_speech_ms = 0
        self.interrupted = False
        # Cooldown timestamp: discard incoming audio chunks until this time
        # to prevent TTS tail-echo slipping into the new buffer after an interrupt
        self.interrupt_cooldown_until: float = 0.0

        # Inactivity state
        self.user_speaking_start_time = 0
        self.last_speech_or_prompt_time = time.time()
        self.inactivity_check_count = 0

        # Adaptive noise-floor calibration
        # Collects the first ~500ms of audio per session to measure ambient RMS,
        # then sets a session-specific speech threshold instead of a hardcoded one.
        self.noise_floor_rms: float = 150.0   # conservative default before calibration
        self.threshold_calibrated: bool = False
        self._calibration_buffer: bytearray = bytearray()
        self.last_recalibration_time: float = 0.0  # for rolling mid-call recalibration
        self.allow_calibration: bool = True
        self.force_finalize: bool = False
        self.turn_in_flight: bool = False
        self._buffer_cap_warned: bool = False
        self.echo_guard_until: float = 0.0

    def add_chunk(self, chunk: bytes):
        self.pcm_buffer.extend(chunk)
        if len(self.pcm_buffer) > MAX_UTTERANCE_BYTES:
            if not self._buffer_cap_warned:
                print("[AudioSession] Utterance hit max length — force-finalizing instead of dropping the start.")
                self._buffer_cap_warned = True
            self.force_finalize = True

        if not self.allow_calibration or self.threshold_calibrated:
            return
        self._calibration_buffer.extend(chunk)
        if len(self._calibration_buffer) >= 16000:
            cal_pcm = np.frombuffer(bytes(self._calibration_buffer[:16000]), dtype=np.int16)
            rms = float(np.sqrt(np.mean(np.square(cal_pcm.astype(np.float32)))))
            self.noise_floor_rms = rms
            self.threshold_calibrated = True
            print(f"[AudioSession] Noise floor calibrated: RMS={rms:.1f}, "
                  f"speech threshold={self.speech_threshold:.1f}")

    @property
    def speech_threshold(self) -> float:
        """Dynamic RMS speech detection threshold: 3x noise floor, clamped 150–600."""
        return max(150.0, min(600.0, self.noise_floor_rms * 3.0))

    def get_full_pcm(self) -> np.ndarray:
        if not self.pcm_buffer:
            return np.array([], dtype=np.int16)
        return np.frombuffer(bytes(self.pcm_buffer), dtype=np.int16)

    def reset_after_transcript(self):
        self.pcm_buffer = bytearray()
        self._buffer_cap_warned = False
        self.force_finalize = False

    def reset_for_interrupt(self, confirmed_speech_ms: int):
        """On barge-in interrupt, preserve the user's confirmed speech tail.

        When the barge-in confirmation fires, pcm_buffer already contains
        `confirmed_speech_ms` of the user's real opening words (the samples that
        were used to confirm the interrupt). Blanking the entire buffer with
        reset_after_transcript() would discard those real words, causing the
        first word or two of every interruption to be clipped.

        Instead, keep exactly the last `confirmed_speech_ms` of buffered audio
        so those confirmed frames remain available for transcription once the
        user finishes speaking.
        """
        keep_bytes = int(SAMPLE_RATE * (confirmed_speech_ms / 1000.0)) * 2  # int16 = 2 bytes/sample
        if len(self.pcm_buffer) > keep_bytes:
            self.pcm_buffer = self.pcm_buffer[-keep_bytes:]
        # else: buffer is short enough — keep it all


def has_speech(pcm: np.ndarray, window_ms: int = 300, threshold: float = 500) -> bool:
    """Require RMS energy. If webrtcvad is present, reject frames it marks as unvoiced."""
    if len(pcm) < FRAME_SIZE:
        return False
    window_frames = int(window_ms / FRAME_MS)
    tail = pcm[-FRAME_SIZE * window_frames:]
    if len(tail) == 0:
        return False

    rms = np.sqrt(np.mean(np.square(tail.astype(np.float32))))
    if rms <= threshold:
        return False
    vad = webrtc_speech_in_tail(pcm, window_ms=window_ms)
    if vad is False:
        return False
    return True


def get_peak_rms(pcm: np.ndarray, frame_ms: int = 30) -> float:
    """Computes peak RMS across frame_ms windows of PCM audio.
    
    Using peak frame RMS instead of whole-buffer average prevents silently
    dropping utterances that contain genuine speech followed by trailing silence.
    """
    if len(pcm) == 0:
        return 0.0
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    if len(pcm) < frame_samples:
        return float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))
    num_frames = len(pcm) // frame_samples
    frames = pcm[:num_frames * frame_samples].reshape(num_frames, frame_samples).astype(np.float32)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.max(frame_rms))


async def speak(websocket: WebSocket, session: AudioSession, text: str):
    """Sends TTS audio, ensuring blank/empty messages are never spoken or rendered."""
    if not text or not str(text).strip():
        print("[speak] Warning: Blank response rejected; returning to listening state.")
        await websocket.send_json({"type": "status", "message": "listening"})
        return

    clean_text = clean_spoken_text(text)
    session.assistant_speaking = True
    session.barge_in_speech_ms = 0
    session.interrupted = False
    session.last_speech_or_prompt_time = time.time()

    await websocket.send_json({"type": "transcript", "text": clean_text, "final": True, "speaker": "assistant"})
    await websocket.send_json({"type": "status", "message": "speaking"})
    await websocket.send_json({"type": "tts_stream", "phase": "start"})

    try:
        sent_any = False
        for sentence in split_spoken_sentences(clean_text):
            if session.interrupted:
                break
            audio_bytes = await synthesize(sentence)
            if not audio_bytes:
                continue
            if session.interrupted:
                break
            await websocket.send_bytes(audio_bytes)
            sent_any = True
        if not session.interrupted:
            await websocket.send_json({"type": "tts_stream", "phase": "end"})
        if not sent_any and not session.interrupted:
            session.assistant_speaking = False
            await websocket.send_json({"type": "status", "message": "listening"})
            return
        # Note: We do NOT set assistant_speaking = False here anymore.
        # We wait for the 'playback_ended' signal from the frontend.
    except WebSocketDisconnect:
        session.assistant_speaking = False
    except Exception as e:
        print(f"TTS Error: {e}")
        session.assistant_speaking = False
        if not session.interrupted:
            try:
                await websocket.send_json({"type": "status", "message": "listening"})
            except Exception:
                pass


@app.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket):
    await websocket.accept()
    session = AudioSession()
    session.allow_calibration = False
    session.belfry_enabled = belfry_requested(websocket.query_params.get("belfry"))
    booking_session = BookingSession()
    
    session_id = f"sess_voice_{int(time.time()*1000)}"
    call_start_time = time.time()
    transcript_history = []
    
    await websocket.send_json({"type": "status", "message": "connected"})
    
    greeting = PROMPTS[BookingState.GREETING]
    add_to_history("assistant", greeting, history=booking_session.conversation_history)
    transcript_history.append(f"Assistant: {greeting}")

    last_check = time.time()

    async def process_user_text(text: str, *, turn_t0: float | None = None):
        owned_lock = False
        if not session.turn_in_flight:
            session.turn_in_flight = True
            owned_lock = True
        try:
            text = cleanup_transcript(text)
            if not text:
                return
            transcript_history.append(f"User: {text}")
            await websocket.send_json({"type": "transcript", "text": text, "final": True, "speaker": "user"})
            await websocket.send_json({"type": "status", "message": "thinking"})

            allowed, screened, belfry_action = await screen_inbound(
                text, enabled=session.belfry_enabled
            )
            await websocket.send_json({
                "type": "belfry",
                "enabled": session.belfry_enabled,
                "action": belfry_action,
            })
            if not allowed:
                transcript_history.append(f"Assistant: {screened}")
                await speak(websocket, session, screened)
                return

            t0 = time.time()
            if booking_session.state == BookingState.CONFIRM:
                reply_text = handle_confirmation(booking_session, screened)
            else:
                reply_text = handle_turn(booking_session, screened)
            reply_text, _outbound_action = await screen_outbound(
                reply_text, enabled=session.belfry_enabled
            )

            if not reply_text or not str(reply_text).strip():
                current_st = booking_session.state
                reply_text = PROMPTS.get(current_st, "Could you please repeat that? What can I help you with today?")

            llm_ms = int((time.time() - t0) * 1000)
            print(f"[LLM Result ({llm_ms}ms)] '{reply_text}'")

            transcript_history.append(f"Assistant: {reply_text}")
            tts_t0 = time.time()
            await speak(websocket, session, reply_text)
            if turn_t0 is not None:
                print(
                    f"[TURN] nlu={llm_ms}ms tts_send={int((time.time() - tts_t0) * 1000)}ms "
                    f"since_stt={int((time.time() - turn_t0) * 1000)}ms"
                )
        finally:
            if owned_lock:
                session.turn_in_flight = False

    async def transcribe_and_reply(final_pcm: np.ndarray):
        session.turn_in_flight = True
        try:
            await websocket.send_json({"type": "status", "message": "transcribing"})
            if len(final_pcm) < 4000:
                await websocket.send_json({"type": "status", "message": "listening"})
                return

            peak_rms = get_peak_rms(final_pcm, frame_ms=30)
            silence_threshold = max(session.noise_floor_rms * 2.0, float(os.getenv("MIN_SILENCE_THRESHOLD", "220.0")))
            if peak_rms < silence_threshold:
                duration_s = len(final_pcm) / SAMPLE_RATE
                print(
                    f"[STT] Filtered ambient silence/noise (peak_rms={peak_rms:.1f} < threshold={silence_threshold:.1f}, "
                    f"noise_floor={session.noise_floor_rms:.1f}, duration={duration_s:.2f}s) — Whisper skipped"
                )
                await websocket.send_json({"type": "status", "message": "listening"})
                return

            prepared = prepare_utterance(final_pcm, session.noise_floor_rms)
            if len(prepared) < 3200:
                await websocket.send_json({"type": "status", "message": "listening"})
                return

            recent_text = " ".join(transcript_history[-4:])
            stt_language = pick_stt_language(recent_text)
            expected_field = None
            try:
                expected_field = booking_session.next_prompt_field()
            except Exception:
                expected_field = None
            stt_t0 = time.time()
            result = await transcribe_utterance(
                prepared,
                language=stt_language,
                groq_client=groq_client,
                prompt=build_stt_prompt(recent_text, expected_field),
            )
            stt_ms = int((time.time() - stt_t0) * 1000)
            text = cleanup_transcript(result.text, expected_field=expected_field)
            print(
                f"[STT:{result.backend} {stt_ms}ms] raw={result.text!r} "
                f"cleaned={text!r} field={expected_field}"
            )
            last_assistant = ""
            last_user = ""
            for line in reversed(transcript_history):
                if not last_assistant and line.startswith("Assistant:"):
                    last_assistant = line.split(":", 1)[-1].strip()
                if not last_user and line.startswith("User:"):
                    last_user = line.split(":", 1)[-1].strip()
                if last_assistant and last_user:
                    break
            if (
                is_hallucination(text)
                or is_ignore_utterance(text)
                or is_echo_transcript(text, last_assistant)
                or is_repeat_transcript(text, last_user)
            ):
                print(f"[STT] Dropped echo/filler/repeat: '{text[:80]}'")
                text = ""
            if not text:
                await websocket.send_json({"type": "status", "message": "listening"})
                return

            session.last_speech_or_prompt_time = time.time()
            session.inactivity_check_count = 0
            await process_user_text(text, turn_t0=stt_t0)
        except WebSocketDisconnect:
            raise
        except Exception as exc:
            print(f"[STT] turn failed: {exc}")
            try:
                await websocket.send_json({"type": "status", "message": "listening"})
            except Exception:
                pass
        finally:
            session.turn_in_flight = False

    def confirm_barge_in():
        session.interrupted = True
        session.assistant_speaking = False
        session.interrupt_cooldown_until = time.time() + 0.20
        session.reset_for_interrupt(max(session.barge_in_speech_ms, BARGE_IN_CONFIRM_MS))
        session.silence_ms = 0
        session.last_frame_had_speech = True
        session.user_speaking_start_time = 0

    asyncio.create_task(speak(websocket, session, greeting))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if message.get("bytes") is not None:
                if time.time() >= session.interrupt_cooldown_until:
                    session.add_chunk(message["bytes"])
                    if session.assistant_speaking:
                        max_bytes = int(SAMPLE_RATE * 0.4) * 2
                        if len(session.pcm_buffer) > max_bytes:
                            session.pcm_buffer = session.pcm_buffer[-max_bytes:]
            elif message.get("text"):
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "playback_ended":
                        session.assistant_speaking = False
                        session.last_speech_or_prompt_time = time.time()
                        session.inactivity_check_count = 0
                        session.allow_calibration = True
                        session.echo_guard_until = time.time() + 0.30
                        session.reset_after_transcript()
                        session.last_frame_had_speech = False
                        session.silence_ms = 0
                        session.interrupted = False
                        await websocket.send_json({"type": "status", "message": "listening"})
                    elif data.get("type") in ("barge_in", "interrupt"):
                        # Ignore client-side energy barge-in — speaker echo was cutting Elena
                        # and then transcribing her own voice as the patient.
                        pass
                    elif data.get("type") == "belfry":
                        session.belfry_enabled = belfry_requested(data.get("enabled"))
                        await websocket.send_json({
                            "type": "belfry",
                            "enabled": session.belfry_enabled,
                            "action": "off" if not session.belfry_enabled else "ready",
                        })
                    elif data.get("type") == "user_text":
                        typed = str(data.get("text") or "").strip()
                        if typed and not session.turn_in_flight:
                            asyncio.create_task(process_user_text(typed))
                except Exception as e:
                    print(f"WS Text Error: {e}")

            now = time.time()
            elapsed_ms = min(int((now - last_check) * 1000), 200)
            if elapsed_ms < 20:
                continue
            last_check = now

            if now < session.echo_guard_until:
                continue

            pcm = session.get_full_pcm()
            if len(pcm) == 0:
                continue

            if session.assistant_speaking:
                barge_threshold = max(session.speech_threshold * 3.5, 900.0)
                speaking_detected = has_speech(pcm, window_ms=150, threshold=barge_threshold)
                if speaking_detected:
                    session.barge_in_speech_ms += elapsed_ms
                    if session.barge_in_speech_ms >= BARGE_IN_CONFIRM_MS and not session.interrupted:
                        confirm_barge_in()
                        await websocket.send_json({"type": "interrupt"})
                        await websocket.send_json({"type": "status", "message": "listening"})
                else:
                    session.barge_in_speech_ms = 0
                continue

            speaking_now = has_speech(pcm, window_ms=300, threshold=session.speech_threshold)
            if speaking_now:
                session.silence_ms = 0
                session.last_frame_had_speech = True
                if session.user_speaking_start_time == 0:
                    session.user_speaking_start_time = now
            else:
                session.silence_ms += elapsed_ms
                if session.silence_ms >= 500:
                    session.user_speaking_start_time = 0
                if (session.silence_ms >= 1000
                        and len(pcm) >= FRAME_SIZE
                        and now - session.last_recalibration_time > 10.0):
                    tail = pcm[-FRAME_SIZE * int(300 / FRAME_MS):]
                    ambient_rms = float(np.sqrt(np.mean(np.square(tail.astype(np.float32)))))
                    session.noise_floor_rms = 0.7 * session.noise_floor_rms + 0.3 * ambient_rms
                    session.last_recalibration_time = now
                    print(f"[AudioSession] Rolling recalibration: ambient_rms={ambient_rms:.1f}, "
                          f"noise_floor={session.noise_floor_rms:.1f}, "
                          f"speech_threshold={session.speech_threshold:.1f}")

            if session.turn_in_flight:
                continue

            if not session.last_frame_had_speech and (now - session.last_speech_or_prompt_time > 25.0):
                if session.inactivity_check_count == 0:
                    session.inactivity_check_count = 1
                    session.last_speech_or_prompt_time = now
                    check_in = "Take your time, I'm still here whenever you're ready."
                    transcript_history.append(f"Assistant: {check_in}")
                    asyncio.create_task(speak(websocket, session, check_in))
                    continue

            should_finalize = (
                session.last_frame_had_speech
                and (session.silence_ms >= SILENCE_MS_TO_FINALIZE or session.force_finalize)
            )
            if should_finalize:
                final_pcm = session.get_full_pcm()
                session.reset_after_transcript()
                session.silence_ms = 0
                session.last_frame_had_speech = False
                session.user_speaking_start_time = 0
                asyncio.create_task(transcribe_and_reply(final_pcm))

    except (WebSocketDisconnect, RuntimeError):
        print("Client disconnected")
    finally:
        # Save complete call session to database
        try:
            duration = int(time.time() - call_start_time)
            status = "COMPLETED" if booking_session.state == BookingState.BOOKED else "INTERRUPTED"
            database.save_call_log(
                session_id=session_id,
                caller_phone=booking_session.slots.phone,
                caller_name=booking_session.slots.name,
                service_requested=booking_session.slots.service,
                call_duration_seconds=duration,
                transcript="\n".join(transcript_history),
                status=status
            )
            print(f"[Database] Call log saved for session {session_id} ({duration}s, status: {status})")
        except Exception as ex:
            print(f"[Database] Error logging call: {ex}")


# ==========================================
# REST API ENDPOINTS FOR DATABASE & RECORDS
# ==========================================

class QuickBookingRequest(BaseModel):
    service: str
    date: str
    time: str
    name: str
    phone: str
    notes: Optional[str] = None

@app.get("/api/database/overview")
async def api_db_overview():
    """Returns database summary, table counts, and connection health."""
    return database.get_db_summary()

@app.get("/api/database/hospitals")
async def api_db_hospitals():
    """Returns hospital administrative and licensing records."""
    return {"hospitals": database.get_all_hospitals()}

@app.get("/api/database/doctors")
async def api_db_doctors():
    """Returns all medical specialists, departments, and consultation fees."""
    return {"doctors": database.get_all_doctors()}

@app.get("/api/database/services")
async def api_db_services():
    """Returns clinical services catalogue."""
    return {"services": database.get_all_services()}

@app.get("/api/database/patients")
async def api_db_patients(limit: int = 50):
    """Returns patient profiles including sensitive medical history, MRN, and insurance."""
    return {"patients": database.get_all_patients(limit=limit)}

@app.get("/api/database/appointments")
async def api_db_appointments(limit: int = 50):
    """Returns all booked appointments with doctor and patient details."""
    return {"appointments": database.get_all_appointments(limit=limit)}

@app.get("/api/database/call-logs")
async def api_db_call_logs(limit: int = 50):
    """Returns voice agent call logs with duration and transcripts."""
    return {"call_logs": database.get_all_call_logs(limit=limit)}

@app.post("/api/database/seed")
async def api_db_seed(force: bool = True):
    """Re-populates database with clean sample healthcare records."""
    seed_database(force_refresh=force)
    return {"success": True, "message": "Database seeded successfully with sample data", "summary": database.get_db_summary()}

@app.post("/api/database/book")
async def api_db_quick_book(req: QuickBookingRequest):
    """Direct appointment creation endpoint for testing and UI."""
    is_free = database.check_slot_available(req.date, req.time)
    if not is_free:
        return {"success": False, "message": f"Time slot {req.time} on {req.date} is already booked."}
        
    appointment = database.create_appointment(
        service_name=req.service,
        date=req.date,
        time=req.time,
        patient_name=req.name,
        patient_phone=req.phone,
        booked_via="WEB_PORTAL",
        notes=req.notes
    )
    return {"success": True, "appointment": appointment}


# ==========================================
# TEXT CHATBOT ENDPOINTS
# ==========================================

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str
    belfry_enabled: bool = True

class ChatResetRequest(BaseModel):
    session_id: str

# In-memory session store for text chatbot
chat_sessions: Dict[str, BookingSession] = {}

def get_state_quick_replies(state: BookingState) -> List[str]:
    """Provides smart suggestion chips for the chat interface based on conversation state."""
    suggestions = {
        BookingState.GREETING: ["Schedule an appointment", "I need a checkup", "See a specialist"],
        BookingState.COLLECT_SERVICE: ["General Checkup", "Cardiology Consultation", "Dermatology Skin Exam", "Pediatric Screening", "Orthopedic Evaluation"],
        BookingState.COLLECT_DATE: ["Tomorrow", "Day after tomorrow", "This Friday", "Next Monday"],
        BookingState.COLLECT_TIME: ["10:00 AM", "11:30 AM", "02:00 PM", "04:30 PM"],
        BookingState.COLLECT_NAME: [],
        BookingState.COLLECT_PHONE: [],
        BookingState.CONFIRM: ["Yes, sounds great!", "No, need to change"],
        BookingState.BOOKED: ["Schedule another appointment", "View my booking in DB", "Services & pricing"]
    }
    return suggestions.get(state, [])

async def _chat_turn_response(session_id: str, user_text: str, *, belfry_enabled: bool = True) -> dict:
    """Run one concierge turn and return the chat payload."""
    session_id = session_id.strip() if session_id else "default_chat_session"
    if session_id not in chat_sessions:
        chat_sessions[session_id] = BookingSession()

    session = chat_sessions[session_id]

    if not user_text:
        return {
            "reply": PROMPTS[BookingState.GREETING],
            "state": session.state.value,
            "slots": {
                "service": session.slots.service,
                "date": session.slots.date,
                "time": session.slots.time,
                "name": session.slots.name,
                "phone": session.slots.phone,
            },
            "quick_replies": get_state_quick_replies(session.state),
            "is_booked": False,
            "belfry_enabled": belfry_enabled,
            "belfry_action": "off" if not belfry_enabled else "allow",
        }

    # Every patient message is checked by Belfry before the model sees it,
    # unless the showcase toggle is off.
    allowed, screened, belfry_action = await screen_inbound(user_text, enabled=belfry_enabled)
    if not allowed:
        return {
            "reply": screened,
            "state": session.state.value,
            "slots": {
                "service": session.slots.service,
                "date": session.slots.date,
                "time": session.slots.time,
                "name": session.slots.name,
                "phone": session.slots.phone,
            },
            "quick_replies": get_state_quick_replies(session.state),
            "is_booked": False,
            "belfry_enabled": belfry_enabled,
            "belfry_action": belfry_action,
        }

    if session.state == BookingState.CONFIRM:
        reply_text = handle_confirmation(session, screened)
    else:
        reply_text = handle_turn(session, screened)
    reply_text, outbound_action = await screen_outbound(reply_text, enabled=belfry_enabled)
    if belfry_action in {"allow", "warn"} and outbound_action not in {"allow", "warn", "off"}:
        belfry_action = outbound_action

    is_booked = (session.state == BookingState.BOOKED)

    slots_dict = {
        "service": session.slots.service,
        "date": session.slots.date,
        "time": session.slots.time,
        "name": session.slots.name,
        "phone": session.slots.phone,
    }

    return {
        "reply": reply_text,
        "state": session.state.value,
        "slots": slots_dict,
        "quick_replies": get_state_quick_replies(session.state),
        "is_booked": is_booked,
        "belfry_enabled": belfry_enabled,
        "belfry_action": belfry_action,
    }


@app.post("/api/chat/message")
async def api_chat_message(req: ChatMessageRequest):
    """Handles text message turns for the interactive Chatbot mode."""
    session_id = req.session_id.strip() if req.session_id else "default_chat_session"
    return await _chat_turn_response(
        session_id,
        req.message.strip(),
        belfry_enabled=belfry_requested(req.belfry_enabled),
    )


async def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    data = await file.read(15 * 1024 * 1024 + 1)
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File is larger than 15 MB.")
    return file.filename or "upload", data


def _attachment_message(filename: str, data: bytes, caption: str, kind: str) -> tuple[str, dict]:
    extracted = _read_extracted(filename, data, kind)
    return build_user_message(caption, extracted), _attachment_meta(extracted)


def _read_extracted(filename: str, data: bytes, kind: str):
    try:
        return extract_upload(
            filename,
            data,
            force_image=(kind == "image"),
            force_audio=(kind == "audio"),
        )
    except AttachmentTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


def _attachment_meta(extracted) -> dict:
    return {
        "filename": extracted.filename,
        "kind": extracted.kind,
        "warning": extracted.warning,
    }


_REVIEW_SYSTEM = (
    "You review clinic files and images. Explain what the attachment actually says. "
    "Mention names, dates, times, phone numbers, departments, and requests only when "
    "they appear in the extracted text. If a question is included, answer it from that text. "
    "Do not book an appointment, do not ask for a name or phone number, and do not invent details. "
    "Use at most 130 words."
)


def _review_with_model(filename: str, kind: str, text: str, question: str) -> str:
    from extraction import MODEL_NAME, client as llm_client

    question = (question or "").strip()
    user = f"Attachment type: {kind}\nFilename: {filename}\n"
    if question:
        user += f"Question: {question}\n"
    user += f"\nExtracted text:\n{text[:6000]}"
    result = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _REVIEW_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=320,
        temperature=0.2,
    )
    return (result.choices[0].message.content or "").strip()


def _plain_review(filename: str, text: str) -> str:
    clipped = text.strip()[:900]
    return f"Here's what I could read from {filename}:\n{clipped}"


async def _review_payload(
    filename: str,
    data: bytes,
    question: str,
    kind: str,
    *,
    belfry_enabled: bool,
) -> dict:
    """Read a file or image and explain it. This does not touch a booking."""
    extracted = await asyncio.to_thread(_read_extracted, filename, data, kind)
    attachment = _attachment_meta(extracted)
    source = extracted.text.strip()
    if not source:
        return {
            "reply": extracted.warning or f"I couldn't read any text from {extracted.filename}.",
            "purpose": "review",
            "attachment": attachment,
            "belfry_enabled": belfry_enabled,
            "belfry_action": "off" if not belfry_enabled else "allow",
        }

    screened_input = f"{question.strip()}\n{source}" if question.strip() else source
    allowed, screened, belfry_action = await screen_inbound(screened_input, enabled=belfry_enabled)
    if not allowed:
        return {
            "reply": screened,
            "purpose": "review",
            "attachment": attachment,
            "belfry_enabled": belfry_enabled,
            "belfry_action": belfry_action,
        }

    try:
        reply = await asyncio.to_thread(
            _review_with_model,
            extracted.filename,
            extracted.kind,
            source,
            question,
        )
    except Exception as exc:
        print(f"[Review] Model summary failed for {extracted.filename}: {exc}")
        reply = _plain_review(extracted.filename, source)
    if not reply:
        reply = _plain_review(extracted.filename, source)
    reply = reply.replace("**", "").replace("`", "")
    reply, outbound_action = await screen_outbound(reply, enabled=belfry_enabled)
    if belfry_action in {"allow", "warn"} and outbound_action not in {"allow", "warn", "off"}:
        belfry_action = outbound_action
    return {
        "reply": reply,
        "purpose": "review",
        "attachment": attachment,
        "belfry_enabled": belfry_enabled,
        "belfry_action": belfry_action,
    }


@app.post("/api/review/upload")
async def api_review_upload(
    message: str = Form(""),
    kind: str = Form("file"),
    belfry_enabled: str = Form("true"),
    file: UploadFile = File(...),
):
    """Explain a file or image. Booking stays on text and audio."""
    filename, data = await _read_upload(file)
    return await _review_payload(
        filename,
        data,
        message,
        kind,
        belfry_enabled=belfry_requested(belfry_enabled),
    )


def _unclear_audio_response(session_id: str, enabled: bool, attachment: dict) -> dict:
    """A silent or invented transcript must not open or change a booking."""
    session = chat_sessions.get(session_id)
    state = session.state if session else BookingState.GREETING
    slots = session.slots if session else None
    return {
        "reply": (
            "I didn't catch a clear appointment request in that recording. "
            "Please say the service, day, and time again."
        ),
        "purpose": "booking",
        "state": state.value,
        "slots": {
            "service": getattr(slots, "service", None),
            "date": getattr(slots, "date", None),
            "time": getattr(slots, "time", None),
            "name": getattr(slots, "name", None),
            "phone": getattr(slots, "phone", None),
        },
        "quick_replies": get_state_quick_replies(state),
        "is_booked": False,
        "belfry_enabled": enabled,
        "belfry_action": "off" if not enabled else "allow",
        "attachment": attachment,
    }


@app.post("/api/chat/upload")
async def api_chat_upload(
    session_id: str = Form("default_chat_session"),
    message: str = Form(""),
    kind: str = Form("file"),
    belfry_enabled: str = Form("true"),
    file: UploadFile = File(...),
):
    """Audio continues a booking. A file or image is reviewed instead."""
    filename, data = await _read_upload(file)
    enabled = belfry_requested(belfry_enabled)
    if kind != "audio":
        return await _review_payload(filename, data, message, kind, belfry_enabled=enabled)
    user_text, attachment = await asyncio.to_thread(_attachment_message, filename, data, message, kind)
    warning = (attachment.get("warning") or "").lower()
    if "no speech could be transcribed" in warning:
        typed = spoken_booking_text(message)
        if not typed:
            return _unclear_audio_response(session_id, enabled, attachment)
        user_text = typed
    payload = await _chat_turn_response(session_id, user_text, belfry_enabled=enabled)
    payload["attachment"] = attachment
    payload["purpose"] = "booking"
    return payload


@app.post("/api/attachments/extract")
async def api_extract_attachment(
    kind: str = Form("file"),
    file: UploadFile = File(...),
):
    """Read a file or image so a live voice call can send it as a patient turn."""
    filename, data = await _read_upload(file)
    user_text, attachment = await asyncio.to_thread(_attachment_message, filename, data, "", kind)
    return {"message": user_text, "attachment": attachment}


@app.post("/api/chat/reset")
async def api_chat_reset(req: ChatResetRequest):
    """Resets the chatbot session to greeting state."""
    session_id = req.session_id.strip() if req.session_id else "default_chat_session"
    chat_sessions[session_id] = BookingSession()
    greeting = PROMPTS[BookingState.GREETING]
    return {
        "success": True,
        "message": "Chat session reset",
        "reply": greeting,
        "state": BookingState.GREETING.value,
        "quick_replies": get_state_quick_replies(BookingState.GREETING)
    }

# Mount the frontend directory to serve the static UI at the root path '/'
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


