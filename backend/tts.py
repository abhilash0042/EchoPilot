import edge_tts
import tempfile
import os

# Indian English & Telugu Expressive Neural Voices
VOICE_EN = "en-IN-NeerjaNeural"  # Professional, natural human Indian English voice
VOICE_TE = "te-IN-ShrutiNeural"  # Fluent, natural Telugu voice

def has_telugu(text: str) -> bool:
    """Check if text contains Telugu script characters (\u0c00 - \u0c7f)."""
    return any('\u0c00' <= char <= '\u0c7f' for char in text)

async def synthesize(text: str) -> bytes:
    """Returns MP3 audio bytes using Microsoft Edge Neural TTS.
    Uses rate='+0%' for a smooth, natural conversational pace.
    """
    if not text or not text.strip():
        return b""

    voice = VOICE_TE if has_telugu(text) else VOICE_EN
        
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Use rate="+0%" for a smoother, less robotic conversational pace
        communicate = edge_tts.Communicate(text, voice, rate="+0%")
        await communicate.save(tmp_path)

        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        # Fallback to Neerja
        try:
            communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural", rate="+0%")
            await communicate.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception:
            return b""
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

