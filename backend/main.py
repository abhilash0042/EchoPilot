import asyncio
import io
import time
import numpy as np
import os
import site
import json
import wave
from typing import Optional, List, Dict, Any
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
    title="Lumina Health AI Voice Agent API",
    description="Healthcare voice assistant with SQLite database integration",
    version="2.0.0"
)

# Auto-initialize database and sample data on startup
@app.on_event("startup")
def on_startup():
    database.init_db()
    seed_database(force_refresh=False)
    print("[Server] Database initialized and verified ready.")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://echopilot-voice-agent.onrender.com"
]

if os.getenv("ALLOWED_ORIGINS"):
    ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
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
        
        # Backchanneling & Inactivity state
        self.user_speaking_start_time = 0
        self.backchannel_played = False
        self.last_speech_or_prompt_time = time.time()
        self.inactivity_check_count = 0

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
    
    greeting = "Hey there! I'm Lumina from the health clinic. What can I help you with today?"
    booking_session.state = BookingState.COLLECT_SERVICE
    add_to_history("assistant", greeting)
    transcript_history.append(f"Assistant: {greeting}")
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

            speaking_now = has_speech(pcm, window_ms=300, threshold=550)
            if speaking_now:
                session.silence_ms = 0
                session.last_frame_had_speech = True
                
                if session.user_speaking_start_time == 0:
                    session.user_speaking_start_time = now
                    
                elapsed_speaking_ms = (now - session.user_speaking_start_time) * 1000
                
                # --- Backchanneling ("mhm", "ok") ---
                if elapsed_speaking_ms > 4000 and not session.backchannel_played:
                    session.backchannel_played = True
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
                if session.silence_ms >= 500:
                    session.user_speaking_start_time = 0

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
                            prompt="Abhilash, Lumina, health clinic, checkup, booking, appointment, doctor, cardiology, dermatology",
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
                        initial_prompt="Abhilash, Lumina, health clinic, checkup, booking, appointment, doctor",
                    )
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                
                # Filter Whisper hallucinations
                hallucinations = [
                    "thank you for watching", "subscribe to", "thanks for watching", 
                    "thank you.", "you.", "you", "please subscribe", "subscribe.",
                    "hmm.", "haan.", "okay.", "ok.", "ha.", "theek hai.", "nahi.", "haan ji.", "ji haan."
                ]
                if len(text) < 2 or text.lower() in hallucinations or "thank you for watching" in text.lower():
                    text = ""

                if not text:
                    await websocket.send_json({"type": "status", "message": "listening"})
                    continue

                # Valid user speech detected — reset silence timer and check-in count
                session.last_speech_or_prompt_time = now
                session.inactivity_check_count = 0

                print(f"[STT Result] '{text}'")
                transcript_history.append(f"User: {text}")
                await websocket.send_json({"type": "transcript", "text": text, "final": True, "speaker": "user"})
                await websocket.send_json({"type": "status", "message": "thinking"})

                t0 = time.time()
                if booking_session.state == BookingState.CONFIRM:
                    reply_text = handle_confirmation(booking_session, text)
                else:
                    reply_text = handle_turn(booking_session, text)
                
                llm_ms = int((time.time() - t0) * 1000)
                print(f"[LLM Result ({llm_ms}ms)] '{reply_text}'")
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

from pydantic import BaseModel

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
            "reply": "Hello! I'm Lumina from the health clinic. How can I help you today?",
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
    greeting = "Hello! I'm Lumina from the health clinic. What can I help you with today?"
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


