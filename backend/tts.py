import edge_tts
import tempfile
import os

# Indian English & Telugu Neural Voices
VOICE_EN = "en-IN-NeerjaNeural"
VOICE_TE = "te-IN-ShrutiNeural"

def has_telugu(text: str) -> bool:
    """Check if text contains Telugu script characters (\u0c00 - \u0c7f)."""
    return any('\u0c00' <= char <= '\u0c7f' for char in text)

async def synthesize(text: str) -> bytes:
    """Returns MP3 audio bytes using Microsoft Edge Neural TTS.
    Automatically picks Telugu voice (te-IN-ShrutiNeural) for Telugu text 
    or Indian English voice (en-IN-NeerjaNeural) for English text.
    """
    if not text or not text.strip():
        return b""

    voice = VOICE_TE if has_telugu(text) else VOICE_EN
        
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

