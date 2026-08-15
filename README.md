<div align="center">
  <img src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/heartbeat.svg" width="80" height="80" alt="Lumina Health Logo">
  <h1>🎙️ EchoPilot</h1>
  <p><strong>A 100% Local, Low-Latency AI Voice Receptionist</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/Ollama-Llama_3.2-black?style=flat&logo=ollama" alt="Ollama">
    <img src="https://img.shields.io/badge/Whisper-STT-green.svg" alt="Whisper">
  </p>
</div>

---

**EchoPilot** is an open-source, fully offline conversational AI agent designed specifically for booking and scheduling workflows. By combining **Ollama** (for local LLM inference), **Faster-Whisper** (for near-instant speech-to-text), and **WebSockets**, EchoPilot delivers a fluid, human-like phone conversation experience right from your browser—with absolutely **zero API costs or rate limits**.

## ✨ Features

- **🔒 100% Local & Private:** Powered by Ollama (`llama3.2:3b`), keeping all patient/caller data on-device. No cloud LLM fees.
- **⚡ Ultra-Low Latency:** Optimized for sub-second response times using `faster-whisper` and Edge TTS.
- **🗣️ True Barge-In Support:** The AI stops speaking instantly if you interrupt it, just like a real human.
- **🧠 Contextual Memory & Persona:** It doesn't just read a menu. It tracks conversation history, engages in small talk, and gently steers users back to the booking flow.
- **💬 Premium UI:** A sleek, dark-mode frontend with glassmorphism, dynamic audio visualizers, and iMessage-style transcript bubbles.
- **🛠️ Robust State Machine:** Built-in slot filling mechanism to extract required data (Service, Date, Time, Name, Phone) and format it into structured JSON.

## 🏗️ Architecture

1. **Frontend (Browser):** Captures microphone audio (16kHz PCM) via WebRTC and streams it over WebSockets. Renders the sleek UI and dynamic visualizers.
2. **Backend (FastAPI):**
   - **VAD (Voice Activity Detection):** Detects when the user starts and stops speaking.
   - **STT (Faster-Whisper):** Transcribes the audio chunks locally on GPU/CPU.
   - **LLM (Ollama / Llama 3.2):** Processes the transcript, maintains persona, and figures out the next state in the booking flow.
   - **TTS (Edge-TTS):** Generates the AI's spoken response and streams the binary audio back to the frontend.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Ollama** installed on your system (get it from [ollama.com](https://ollama.com))

### 2. Download the Local LLM
Open your terminal and pull the lightweight Llama 3.2 model:
```bash
ollama run llama3.2
```
*(Leave this running in the background, or ensure the Ollama system tray icon is active).*

### 3. Backend Setup
Clone the repository and install the Python dependencies:
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # (On Windows)
pip install -r requirements.txt
```

Start the FastAPI WebSocket server:
```bash
uvicorn main:app --reload
```

### 4. Frontend Setup
In a new terminal window, serve the frontend files:
```bash
cd frontend
python -m http.server 3000
```
Open your browser and navigate to **[http://localhost:3000](http://localhost:3000)**. Click "Call AI Assistant" to begin!

## 🤝 Contributing
Contributions are welcome! Whether it's adding support for Twilio/SIP phone calls, improving the barge-in detection algorithms, or adding new language models.

---
<div align="center">
  <i>Built with ❤️ for the future of conversational Voice AI.</i>
</div>
