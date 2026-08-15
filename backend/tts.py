import edge_tts
import tempfile
import os

# Indian English & Telugu Expressive Neural Voices
VOICE_EN = "en-IN-KavyaNeural"  # Soft, warm, human Indian English voice
VOICE_TE = "te-IN-ShrutiNeural"  # Fluent, natural Telugu voice

def has_telugu(text: str) -> bool:
    """Check if text contains Telugu script characters (\u0c00 - \u0c7f)."""
    return any('\u0c00' <= char <= '\u0c7f' for char in text)

async def synthesize(text: str) -> bytes:
    """Returns MP3 audio bytes using Microsoft Edge Neural TTS.
    Uses rate='-3%' for smooth, natural human speaking rhythm without robotic staccato.
    """
    if not text or not text.strip():
        return b""

    voice = VOICE_TE if has_telugu(text) else VOICE_EN
        
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Use rate="-3%" for a warmer, smoother conversational pace
        communicate = edge_tts.Communicate(text, voice, rate="-3%")
        await communicate.save(tmp_path)

        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        # Fallback to Neerja if Kavya is unavailable in any region
        try:
            communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural", rate="-3%")
            await communicate.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception:
            return b""
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

