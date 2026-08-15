import edge_tts
import tempfile
import os

# Indian English & Telugu Expressive Neural Voices
VOICE_EN = "en-IN-NeerjaNeural"  # Professional, natural human Indian English voice
VOICE_TE = "te-IN-ShrutiNeural"  # Fluent, natural Telugu voice

def has_telugu(text: str) -> bool:
    """Check if text contains Telugu script characters (\u0c00 - \u0c7f)."""
    return any('\u0c00' <= char <= '\u0c7f' for char in text)

def inject_ai_watermark(audio_bytes: bytes) -> bytes:
    """Injects an ID3v2 metadata header into the MP3 stream to declare it as machine-generated AI speech."""
    if not audio_bytes or audio_bytes.startswith(b"ID3"):
        return audio_bytes
        
    artist = b"EchoPilot Lumina AI"
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

async def synthesize(text: str) -> bytes:
    """Returns MP3 audio bytes using Microsoft Edge Neural TTS with AI metadata watermark.
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
            return inject_ai_watermark(f.read())
    except Exception as e:
        # Fallback to Neerja
        try:
            communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural", rate="+0%")
            await communicate.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return inject_ai_watermark(f.read())
        except Exception:
            return b""
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

