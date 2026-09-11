import io
import time
import numpy as np
import os
import site
import json
import wave
from typing import Optional, List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
try:
    # pyrefly: ignore [missing-import]
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
from openai import OpenAI


from dialogue_manager import handle_turn, handle_confirmation, PROMPTS
from booking import BookingSession, BookingState
from tts import synthesize, warm_cache
from extraction import add_to_history, reset_history
import database
from seed_data import seed_database
from belfry_client import belfry_check_input, belfry_check_output

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
        # Common safety/fallback replies
        "I didn't quite catch that — could you say that again?",
        "I apologize, but I cannot process that request. How else may I assist you?",
    ]
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

# Circuit-breaker for Groq STT — prevents hammering a down/slow Groq endpoint
# turn-by-turn and waiting out long timeouts on every single turn.
#
# State machine:
#   CLOSED (normal): Groq calls allowed. On failure, increment _groq_failure_count.
#   OPEN:  After GROQ_FAILURE_THRESHOLD consecutive failures, circuit opens.
#          All turns skip Groq and go directly to local fallback for
#          GROQ_CIRCUIT_OPEN_SECONDS, then auto-close and retry.
_groq_failure_count: int = 0
_groq_circuit_open_until: float = 0.0
GROQ_FAILURE_THRESHOLD = 3        # consecutive failures before opening circuit
GROQ_CIRCUIT_OPEN_SECONDS = 30    # seconds to stay on local fallback before retrying Groq
GROQ_TIMEOUT_SECONDS = 4.0        # per-call hard timeout — fast-fails on network hangs

# Fallback local whisper model (lazy-loaded if Groq cloud STT is unavailable)
_local_whisper_model = None

def get_local_whisper_model():
    global _local_whisper_model
    if _local_whisper_model is None and WhisperModel:
        try:
            _local_whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
            print("[STT] Local CUDA Faster-Whisper model loaded")
        except Exception:
            try:
                _local_whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
                print("[STT] Local CPU Faster-Whisper model loaded")
            except Exception as e2:
                print(f"[STT] Local Whisper model skipped: {e2}")
    return _local_whisper_model



SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad requires 10/20/30ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per VAD frame

SILENCE_MS_TO_FINALIZE = 900  # 900ms silence threshold optimized for natural pauses


BARGE_IN_CONFIRM_MS = 250  # Reduced from 350 for faster interrupts

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

    def add_chunk(self, chunk: bytes):
        self.pcm_buffer.extend(chunk)
        # Cap buffer at ~15 seconds (16000 rate * 2 bytes/sample * 15s = 480000 bytes)
        if len(self.pcm_buffer) > 480000:
            print("[AudioSession] WARNING: 15s buffer cap hit — dropping head of utterance. "
                  "User may be speaking very long sentences.")
            self.pcm_buffer = self.pcm_buffer[-480000:]

        # Calibration: accumulate the first ~500ms of audio (16000 Hz * 0.5s * 2 bytes = 16000 bytes)
        # to measure the ambient noise floor and derive a session-specific RMS threshold.
        if not self.threshold_calibrated:
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
    """Runs a simple RMS energy check over the last window_ms of audio."""
    if len(pcm) < FRAME_SIZE:
        return False
    window_frames = int(window_ms / FRAME_MS)
    tail = pcm[-FRAME_SIZE * window_frames:]
    if len(tail) == 0:
        return False
        
    rms = np.sqrt(np.mean(np.square(tail.astype(np.float32))))
    return rms > threshold


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
    """Sends TTS audio, but checks after synthesis whether an interrupt
    already happened before playing."""
    session.assistant_speaking = True
    session.barge_in_speech_ms = 0
    session.interrupted = False
    session.last_speech_or_prompt_time = time.time()

    await websocket.send_json({"type": "transcript", "text": text, "final": True, "speaker": "assistant"})
    await websocket.send_json({"type": "status", "message": "speaking"})

    try:
        audio_bytes = await synthesize(text)
        
        if session.interrupted:
            # User already started talking before we finished synthesizing — skip playback
            session.assistant_speaking = False
            return

        await websocket.send_bytes(audio_bytes)
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
    booking_session = BookingSession()
    reset_history()  # Clear conversation history for fresh call
    
    session_id = f"sess_voice_{int(time.time()*1000)}"
    call_start_time = time.time()
    transcript_history = []
    
    await websocket.send_json({"type": "status", "message": "connected"})
    
    greeting = PROMPTS[BookingState.GREETING]
    add_to_history("assistant", greeting)
    transcript_history.append(f"Assistant: {greeting}")
    await speak(websocket, session, greeting)

    last_check = time.time()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                # Drop audio chunks during interrupt cooldown window to prevent
                # TTS tail-echo from entering the fresh transcription buffer
                if time.time() >= session.interrupt_cooldown_until:
                    session.add_chunk(message["bytes"])
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "playback_ended":
                        session.assistant_speaking = False
                        session.last_speech_or_prompt_time = time.time()
                        session.inactivity_check_count = 0
                        if not session.interrupted:
                            await websocket.send_json({"type": "status", "message": "listening"})
                except Exception as e:
                    print(f"WS Text Error: {e}")

            now = time.time()
            if now - last_check < 0.15:  # check more frequently for responsive barge-in
                continue
            last_check = now

            pcm = session.get_full_pcm()
            if len(pcm) == 0:
                continue

            # --- Barge-in check: only relevant while the assistant is speaking ---
            if session.assistant_speaking:
                barge_threshold = max(session.speech_threshold, 200.0)
                speaking_detected = has_speech(pcm, window_ms=150, threshold=barge_threshold)
                if speaking_detected:
                    session.barge_in_speech_ms += 150
                    if session.barge_in_speech_ms >= BARGE_IN_CONFIRM_MS and not session.interrupted:
                        session.interrupted = True
                        session.assistant_speaking = False
                        # Set a 150ms cooldown so TTS tail audio doesn't enter the fresh buffer
                        session.interrupt_cooldown_until = time.time() + 0.15
                        await websocket.send_json({"type": "interrupt"})
                        await websocket.send_json({"type": "status", "message": "listening"})

                        # Preserve the confirmed user speech already captured in the buffer.
                        # reset_after_transcript() would discard the 250ms of real opening
                        # words used to confirm the interrupt, clipping the user's first
                        # word on every interruption.
                        # reset_for_interrupt() keeps only those confirmed speech frames;
                        # everything before them (TTS audio that leaked before VAD fired)
                        # is dropped, and the subsequent 150ms cooldown guards the tail-echo.
                        session.reset_for_interrupt(session.barge_in_speech_ms)
                        session.silence_ms = 0
                        session.last_frame_had_speech = True
                        session.user_speaking_start_time = 0
                else:
                    session.barge_in_speech_ms = 0
                continue

            # Use adaptive threshold derived from per-session noise floor calibration
            speaking_now = has_speech(pcm, window_ms=300, threshold=session.speech_threshold)
            if speaking_now:
                session.silence_ms = 0
                session.last_frame_had_speech = True

                if session.user_speaking_start_time == 0:
                    session.user_speaking_start_time = now
            else:
                session.silence_ms += 150
                if session.silence_ms >= 500:
                    session.user_speaking_start_time = 0

                # Rolling noise-floor recalibration during confirmed silence.
                # Every 10s of accumulated silence, blend the current ambient RMS into
                # the threshold via EMA (0.7 old + 0.3 new) so mid-call environment
                # changes (AC unit, door opening, background talker) don't cause
                # permanent false-positive or false-negative VAD decisions.
                if (session.silence_ms >= 1000
                        and len(pcm) >= FRAME_SIZE
                        and now - session.last_recalibration_time > 10.0):
                    tail = pcm[-FRAME_SIZE * int(300 / FRAME_MS):]  # last 300ms
                    ambient_rms = float(np.sqrt(np.mean(np.square(tail.astype(np.float32)))))
                    old_floor = session.noise_floor_rms
                    session.noise_floor_rms = 0.7 * old_floor + 0.3 * ambient_rms
                    session.last_recalibration_time = now
                    print(f"[AudioSession] Rolling recalibration: ambient_rms={ambient_rms:.1f}, "
                          f"noise_floor={session.noise_floor_rms:.1f}, "
                          f"speech_threshold={session.speech_threshold:.1f}")

            # --- 7-Second Inactivity / Silence Check-In ---
            if not session.last_frame_had_speech and (now - session.last_speech_or_prompt_time > 7.0):
                if session.inactivity_check_count == 0:
                    session.inactivity_check_count = 1
                    session.last_speech_or_prompt_time = now
                    check_in = "Are you there? Take your time, I'm right here!"
                    transcript_history.append(f"Assistant: {check_in}")
                    await speak(websocket, session, check_in)
                    continue
                elif session.inactivity_check_count == 1 and (now - session.last_speech_or_prompt_time > 9.0):
                    session.inactivity_check_count = 2
                    session.last_speech_or_prompt_time = now
                    check_in = "Hello! Can you hear me okay? Just let me know whenever you're ready!"
                    transcript_history.append(f"Assistant: {check_in}")
                    await speak(websocket, session, check_in)
                    continue

            # User was talking, and has now gone quiet for long enough -> finalize
            if session.last_frame_had_speech and session.silence_ms >= SILENCE_MS_TO_FINALIZE:
                await websocket.send_json({"type": "status", "message": "transcribing"})

                # Capture the COMPLETE utterance BEFORE resetting the buffer.
                # The `pcm` variable above was snapshotted earlier in the loop and may
                # be missing the last 150ms chunk. Re-read directly from the buffer now.
                final_pcm = session.get_full_pcm()

                session.reset_after_transcript()
                session.silence_ms = 0
                session.last_frame_had_speech = False
                session.user_speaking_start_time = 0

                if len(final_pcm) < 1600:
                    # Less than 0.1s of audio — noise or tiny click, safely ignore
                    await websocket.send_json({"type": "status", "message": "listening"})
                    continue

                # Guard against Whisper hallucinating on pure silence or ambient room hum.
                # Whisper-family models commonly hallucinate stock phrases ("Thank you for watching")
                # when given silent audio buffers.
                #
                # IMPORTANT: We check PEAK frame RMS (over 30ms frames), NOT whole-buffer average.
                # A buffer with a real but quiet word (e.g. 200ms speech) followed by 900ms of
                # trailing silence has a low buffer-wide average RMS, which would falsely drop genuine
                # utterances. Peak frame RMS ensures that if ANY window had speech-level energy, we
                # transcribe it.
                peak_rms = get_peak_rms(final_pcm, frame_ms=30)
                silence_threshold = max(session.noise_floor_rms * 1.5, 120.0)
                if peak_rms < silence_threshold:
                    duration_s = len(final_pcm) / SAMPLE_RATE
                    print(
                        f"[STT] Skipping transcription — audio energy below speech threshold "
                        f"(peak_rms={peak_rms:.1f} < threshold={silence_threshold:.1f}, "
                        f"noise_floor={session.noise_floor_rms:.1f}, duration={duration_s:.2f}s)"
                    )
                    await websocket.send_json({"type": "status", "message": "listening"})
                    continue

                text = ""
                stt_backend = "none"

                # Thresholds for Whisper's model-side confidence signals (verbose_json).
                # Used as a SECOND gate after RMS: model-side confidence complements the
                # energy-side RMS check — catches loud ambient noise the RMS gate can't distinguish.
                # Both conditions must be true to reject (prevents false silencing of quiet speech).
                NO_SPEECH_PROB_THRESHOLD = 0.80   # segment-level P(no speech) above this = model says silence
                AVG_LOGPROB_THRESHOLD = -1.0       # log-probability below this = low confidence decode

                global _groq_failure_count, _groq_circuit_open_until

                groq_circuit_open = time.time() < _groq_circuit_open_until
                if groq_client and not groq_circuit_open:
                    try:
                        wav_io = io.BytesIO()
                        with wave.open(wav_io, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(final_pcm.tobytes())  # use final_pcm — complete utterance
                        wav_io.seek(0)
                        wav_io.name = "audio.wav"

                        # verbose_json gives segment-level no_speech_prob + avg_logprob.
                        # timeout=GROQ_TIMEOUT_SECONDS fast-fails on network hangs so a
                        # single slow turn doesn't stall the whole session for 60s.
                        transcription = groq_client.audio.transcriptions.create(
                            file=wav_io,
                            model="whisper-large-v3",
                            language="en",            # pin language — prevents wrong-language decode on short utterances
                            temperature=0,            # deterministic output — eliminates non-determinism as a failure source
                            response_format="verbose_json",
                            prompt=(
                                "Healthcare voice conversation in English and Telugu at Meridian Clinic. "
                                "Appointment with doctor, general physician, doctor consultation, checkup, head surgery, surgery, brain surgery, "
                                "cardiology, dermatology, orthopedic, pediatric, ophthalmology, dentist, blood test, X-ray, "
                                "tomorrow, Monday, Tuesday, Wednesday, morning, afternoon, evening, 10 AM, 11 AM, 2 PM, "
                                "haan, nahi, theek hai, kal, parso, subah, dopahar, sham, "
                                "Abhilash, confirm, cancel, change."
                            ),
                            timeout=GROQ_TIMEOUT_SECONDS,
                        )

                        # Check model-side confidence signals before accepting transcript.
                        # Reject only if ALL segments agree there's no speech AND confidence is low.
                        # A single confident segment is enough to pass (handles mixed audio).
                        segments_data = getattr(transcription, "segments", None) or []
                        if segments_data:
                            all_no_speech = all(
                                getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_PROB_THRESHOLD
                                and getattr(seg, "avg_logprob", 0.0) < AVG_LOGPROB_THRESHOLD
                                for seg in segments_data
                            )
                            # Log per-turn for threshold calibration visibility
                            avg_nsp = sum(getattr(s, "no_speech_prob", 0.0) for s in segments_data) / len(segments_data)
                            avg_lp = sum(getattr(s, "avg_logprob", 0.0) for s in segments_data) / len(segments_data)
                            print(f"[STT:groq] no_speech_prob={avg_nsp:.2f} avg_logprob={avg_lp:.2f}")
                            if all_no_speech:
                                print(
                                    f"[STT:groq] Confidence gate rejected transcript — "
                                    f"no_speech_prob={avg_nsp:.2f} > {NO_SPEECH_PROB_THRESHOLD}, "
                                    f"avg_logprob={avg_lp:.2f} < {AVG_LOGPROB_THRESHOLD}"
                                )
                                await websocket.send_json({"type": "status", "message": "listening"})
                                continue

                        text = transcription.text.strip()
                        stt_backend = "groq"

                        # Circuit closed: reset failure count on any success
                        _groq_failure_count = 0

                    except Exception as e:
                        _groq_failure_count += 1
                        remaining = GROQ_FAILURE_THRESHOLD - _groq_failure_count
                        if _groq_failure_count >= GROQ_FAILURE_THRESHOLD:
                            _groq_circuit_open_until = time.time() + GROQ_CIRCUIT_OPEN_SECONDS
                            print(
                                f"[STT] Groq circuit breaker OPEN after {_groq_failure_count} failures "
                                f"— switching to local fallback for {GROQ_CIRCUIT_OPEN_SECONDS}s: {e}"
                            )
                        else:
                            print(
                                f"[STT] Groq error ({_groq_failure_count}/{GROQ_FAILURE_THRESHOLD}) "
                                f"— falling back to local this turn ({remaining} until circuit opens): {e}"
                            )

                elif groq_client and groq_circuit_open:
                    secs_remaining = int(_groq_circuit_open_until - time.time())
                    print(f"[STT] Groq circuit breaker OPEN — using local fallback ({secs_remaining}s remaining)")

                if not text:
                    local_model = get_local_whisper_model()
                    if local_model:
                        print("[STT] Using local fallback model (Groq unavailable or circuit open)")
                        pcm_float = final_pcm.astype(np.float32) / 32768.0  # use final_pcm
                        segments, _ = local_model.transcribe(
                            pcm_float,
                            language="en",   # pin language on local model too
                            temperature=0,   # deterministic output — matches Groq config
                            vad_filter=False,  # our RMS VAD already gated this segment — Silero re-trimming would clip word edges
                            beam_size=5,
                            initial_prompt=(
                                "Healthcare voice conversation in English and Telugu at Meridian Clinic. "
                                "Appointment with doctor, general physician, doctor consultation, checkup, head surgery, surgery, brain surgery, "
                                "cardiology, dermatology, orthopedic, pediatric, ophthalmology, dentist, "
                                "tomorrow, Monday, Tuesday, Wednesday, morning, afternoon, evening, "
                                "haan, nahi, theek hai, kal, parso, subah, dopahar, sham, "
                                "Abhilash, confirm, cancel, change."
                            ),
                        )
                        text = " ".join(seg.text.strip() for seg in segments).strip()
                        stt_backend = "local"

                        # Auto-close circuit: if local succeeds and circuit period has elapsed, reset
                        if time.time() >= _groq_circuit_open_until and _groq_failure_count >= GROQ_FAILURE_THRESHOLD:
                            _groq_failure_count = 0
                            print("[STT] Groq circuit breaker CLOSED — will retry Groq next turn")

                
                # Filter Whisper hallucinations — only block clear YouTube-style noise,
                # NOT valid single-word responses like "haan", "okay", "nahi" that users
                # genuinely say in a healthcare booking conversation.
                HALLUCINATION_SUBSTRINGS = [
                    "thank you for watching", "subscribe to our", "thanks for watching",
                    "please subscribe", "like and subscribe",
                ]
                HALLUCINATION_EXACT = {"hmm", "hmm.", "uh.", "uh", "um.", "um"}
                t_lower = text.lower().strip()
                is_hallucination = (
                    len(t_lower) < 2
                    or t_lower in HALLUCINATION_EXACT
                    or any(p in t_lower for p in HALLUCINATION_SUBSTRINGS)
                )
                if is_hallucination:
                    text = ""

                if not text:
                    await websocket.send_json({"type": "status", "message": "listening"})
                    continue

                # Valid user speech detected — reset silence timer and check-in count
                session.last_speech_or_prompt_time = now
                session.inactivity_check_count = 0

                print(f"[STT:{stt_backend}] raw='{text}'")
                transcript_history.append(f"User: {text}")
                await websocket.send_json({"type": "transcript", "text": text, "final": True, "speaker": "user"})
                await websocket.send_json({"type": "status", "message": "thinking"})

                # Check user input BEFORE sending to LLM (via Belfry SDK)
                in_res = await belfry_check_input(text)
                if in_res.get("action") == "block":
                    refusal = "I apologize, but I cannot process that request. How else may I assist you?"
                    transcript_history.append(f"Assistant: {refusal}")
                    await speak(websocket, session, refusal)
                    continue

                t0 = time.time()
                if booking_session.state == BookingState.CONFIRM:
                    reply_text = handle_confirmation(booking_session, text)
                else:
                    reply_text = handle_turn(booking_session, text)
                
                llm_ms = int((time.time() - t0) * 1000)
                print(f"[LLM Result ({llm_ms}ms)] '{reply_text}'")

                # Check LLM output BEFORE returning to user (via Belfry SDK)
                out_res = await belfry_check_output(reply_text)
                if out_res.get("action") == "block":
                    reply_text = "I apologize, but I cannot share that information. Let's proceed with your appointment."

                transcript_history.append(f"Assistant: {reply_text}")

                await speak(websocket, session, reply_text)

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

@app.post("/api/chat/message")
async def api_chat_message(req: ChatMessageRequest):
    """Handles text message turns for the interactive Chatbot mode."""
    session_id = req.session_id.strip() if req.session_id else "default_chat_session"
    user_text = req.message.strip()

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
            "is_booked": False
        }

    # Check user input BEFORE sending to LLM (via Belfry SDK)
    in_res = await belfry_check_input(user_text)
    if in_res.get("action") == "block":
        return {
            "reply": "I apologize, but I cannot process that request. How else can I assist with your appointment?",
            "state": session.state.value,
            "slots": {
                "service": session.slots.service,
                "date": session.slots.date,
                "time": session.slots.time,
                "name": session.slots.name,
                "phone": session.slots.phone,
            },
            "quick_replies": get_state_quick_replies(session.state),
            "is_booked": False
        }

    # Execute dialogue turn
    if session.state == BookingState.CONFIRM:
        reply_text = handle_confirmation(session, user_text)
    else:
        reply_text = handle_turn(session, user_text)

    # Check LLM output BEFORE returning to user (via Belfry SDK)
    out_res = await belfry_check_output(reply_text)
    if out_res.get("action") == "block":
        reply_text = "I apologize, but I cannot share that information. How else can I help you?"

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
        "is_booked": is_booked
    }

@app.post("/api/chat/reset")
async def api_chat_reset(req: ChatResetRequest):
    """Resets the chatbot session to greeting state."""
    session_id = req.session_id.strip() if req.session_id else "default_chat_session"
    chat_sessions[session_id] = BookingSession()
    reset_history()
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


