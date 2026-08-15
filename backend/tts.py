import edge_tts
import tempfile
import os

# Indian English female voice — natural-sounding Neural TTS
VOICE = "en-IN-NeerjaNeural"


async def synthesize(text: str) -> bytes:
    """Returns MP3 audio bytes using Microsoft Edge Neural TTS.
    
    Edge TTS is free, high-quality, and supports Indian English voices.
    The browser's AudioContext.decodeAudioData() handles MP3 natively.
    """
    if not text or not text.strip():
        return b""
        
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(tmp_path)

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
