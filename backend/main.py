import asyncio
import io
import time
import numpy as np
import os
import site
import json
from dotenv import load_dotenv

import wave
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
from openai import OpenAI


from dialogue_manager import handle_turn, handle_confirmation
from booking import BookingSession, BookingState
from tts import synthesize
from extraction import add_to_history, reset_history

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if GROQ_API_KEY:
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
    print("[STT] Connected to Groq Cloud API (whisper-large-v3)")
else:
    groq_client = None

# Fallback local whisper model
model = None
if WhisperModel:
    try:
        model = WhisperModel("small", device="cuda", compute_type="float16")
        print("[STT] Local CUDA Faster-Whisper model loaded")
    except Exception as e:
        try:
            model = WhisperModel("small", device="cpu", compute_type="int8")
            print("[STT] Local CPU Faster-Whisper model loaded")
        except Exception as e2:
            print(f"[STT] Local Whisper model skipped: {e2}")



SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad requires 10/20/30ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per VAD frame

SILENCE_MS_TO_FINALIZE = 800  # how long user must be silent to trigger transcription


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
        
        # Backchanneling state
        self.user_speaking_start_time = 0
        self.backchannel_played = False

    def add_chunk(self, chunk: bytes):
        self.pcm_buffer.extend(chunk)
        # Cap buffer at ~3 seconds to prevent memory bloat (16000 rate * 2 bytes/sample * 3s = 96000 bytes)
        if len(self.pcm_buffer) > 96000:
            self.pcm_buffer = self.pcm_buffer[-96000:]

    def get_full_pcm(self) -> np.ndarray:
        if not self.pcm_buffer:
            return np.array([], dtype=np.int16)
        return np.frombuffer(bytes(self.pcm_buffer), dtype=np.int16)

    def reset_after_transcript(self):
        self.pcm_buffer = bytearray()


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


async def speak(websocket: WebSocket, session: AudioSession, text: str):
    """Sends TTS audio, but checks after synthesis whether an interrupt
    already happened before playing."""
    session.assistant_speaking = True
    session.barge_in_speech_ms = 0
    session.interrupted = False

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
    await websocket.send_json({"type": "status", "message": "connected"})
    
    greeting = "Hey there! I'm Lumina from the health clinic. What can I help you with today?"
    booking_session.state = BookingState.COLLECT_SERVICE
    add_to_history("assistant", greeting)
    await speak(websocket, session, greeting)

    last_check = time.time()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                session.add_chunk(message["bytes"])
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "playback_ended":
                        session.assistant_speaking = False
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
                # Use a shorter window AND lower threshold for faster interrupt detection
                speaking_detected = has_speech(pcm, window_ms=150, threshold=350)
                if speaking_detected:
                    session.barge_in_speech_ms += 150
                    if session.barge_in_speech_ms >= BARGE_IN_CONFIRM_MS and not session.interrupted:
                        session.interrupted = True
                        session.assistant_speaking = False
                        await websocket.send_json({"type": "interrupt"})
                        await websocket.send_json({"type": "status", "message": "listening"})
                        
                        # Reset the buffer so we start capturing THIS utterance cleanly
                        session.reset_after_transcript()
                        session.silence_ms = 0
                        session.last_frame_had_speech = True
                        session.user_speaking_start_time = 0
                        session.backchannel_played = False
                else:
                    session.barge_in_speech_ms = 0
                continue

            # --- Normal listening flow (bot not speaking) ---
            speaking_now = has_speech(pcm, window_ms=300)
            if speaking_now:
                session.silence_ms = 0
                session.last_frame_had_speech = True
                
                if session.user_speaking_start_time == 0:
                    session.user_speaking_start_time = now
                    
                elapsed_speaking_ms = (now - session.user_speaking_start_time) * 1000
                
                # --- Backchanneling ("mhm", "ok") ---
                # If the user has been speaking for > 4 seconds since starting their turn
                if elapsed_speaking_ms > 4000 and not session.backchannel_played:
                    session.backchannel_played = True
                    # Fire-and-forget an async task without setting assistant_speaking to avoid race conditions
                    async def play_backchannel():
                        try:
                            audio = await synthesize("mhm.")
                            if not session.interrupted:
                                await websocket.send_bytes(audio)
                        except Exception as e:
                            pass
                    asyncio.create_task(play_backchannel())
            else:
                session.silence_ms += 150
                # Only reset user speaking duration on sustained silence, not just a brief 150ms pause
                if session.silence_ms >= 500:
                    session.user_speaking_start_time = 0

            # User was talking, and has now gone quiet for long enough -> finalize
            if session.last_frame_had_speech and session.silence_ms >= SILENCE_MS_TO_FINALIZE:
                await websocket.send_json({"type": "status", "message": "transcribing"})

                pcm_float = pcm.astype(np.float32) / 32768.0

                # IMMEDIATELY reset buffer so new audio doesn't pile up
                session.reset_after_transcript()
                session.silence_ms = 0
                session.last_frame_had_speech = False
                session.user_speaking_start_time = 0
                session.backchannel_played = False

                text = ""
                if groq_client:
                    try:
                        wav_io = io.BytesIO()
                        with wave.open(wav_io, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(pcm.tobytes())
                        wav_io.seek(0)
                        wav_io.name = "audio.wav"

                        transcription = groq_client.audio.transcriptions.create(
                            file=wav_io,
                            model="whisper-large-v3",
                            prompt="Abhilash, Lumina, health clinic, checkup, booking, appointment, doctor, నమస్కారం, అపాయింట్మెంట్, చెకప్, రేపు, ఎల్లుండి, సాయంత్రం, ఉదయం",
                        )
                        text = transcription.text.strip()
                    except Exception as e:
                        print(f"Groq Whisper STT Error, falling back to local model: {e}")

                if not text and model:
                    pcm_float = pcm.astype(np.float32) / 32768.0
                    segments, _ = model.transcribe(
                        pcm_float,
                        vad_filter=True,
                        beam_size=5,
                        initial_prompt="Abhilash, Lumina, health clinic, checkup, booking, appointment, doctor, నమస్కారం, అపాయింట్మెంట్, చెకప్",
                    )
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                
                # Filter Whisper hallucinations
                hallucinations = ["thank you for watching", "subscribe to", "thanks for watching", "thank you.", "you.", "you", "please subscribe", "subscribe."]
                if len(text) < 2 or text.lower() in hallucinations or "thank you for watching" in text.lower():
                    text = ""


                if not text:
                    await websocket.send_json({"type": "status", "message": "listening"})
                    continue

                await websocket.send_json({"type": "transcript", "text": text, "final": True, "speaker": "user"})
                await websocket.send_json({"type": "status", "message": "thinking"})

                if booking_session.state == BookingState.CONFIRM:
                    reply_text = handle_confirmation(booking_session, text)
                else:
                    reply_text = handle_turn(booking_session, text)

                await speak(websocket, session, reply_text)

    except (WebSocketDisconnect, RuntimeError):
        print("Client disconnected")


# Mount the frontend directory to serve the static UI at the root path '/'
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
