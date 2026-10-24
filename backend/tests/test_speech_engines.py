"""Unit tests for the Groq → Nemotron → local STT engine chain."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.pop("NVIDIA_API_KEY", None)
os.environ.pop("NVIDIA_ASR_URL", None)

from speech_engines import TranscriptResult, list_stt_engines, transcribe_utterance


class _FakeTranscription:
    def __init__(self, text):
        self.text = text
        self.segments = []


class TestSpeechEngines(unittest.IsolatedAsyncioTestCase):
    async def test_groq_primary_wins(self):
        pcm = (np.sin(np.linspace(0, 6, 16000)) * 4000).astype(np.int16)
        groq = MagicMock()
        groq.audio.transcriptions.create.return_value = _FakeTranscription(
            "I need cardiology tomorrow"
        )
        result = await transcribe_utterance(pcm, language="en", groq_client=groq)
        self.assertEqual(result.backend, "groq")
        self.assertIn("cardiology", result.text)

    async def test_groq_empty_skips_fallback(self):
        pcm = (np.sin(np.linspace(0, 6, 16000)) * 4000).astype(np.int16)
        groq = MagicMock()
        groq.audio.transcriptions.create.return_value = _FakeTranscription("")
        with patch("speech_engines._transcribe_nemotron_sync") as nemo:
            result = await transcribe_utterance(pcm, language="en", groq_client=groq)
        self.assertEqual(result.backend, "groq")
        self.assertEqual(result.text, "")
        nemo.assert_not_called()

    async def test_nemotron_fallback_when_groq_fails(self):
        pcm = (np.sin(np.linspace(0, 6, 16000)) * 4000).astype(np.int16)
        groq = MagicMock()
        groq.audio.transcriptions.create.side_effect = RuntimeError("timeout")

        with patch.dict(
            os.environ,
            {
                "NVIDIA_API_KEY": "test-key",
                "NVIDIA_ASR_URL": "https://example.nvidia/v1",
            },
        ):
            with patch(
                "speech_engines._transcribe_nemotron_sync",
                return_value="book a dentist please",
            ):
                result = await transcribe_utterance(pcm, language="en", groq_client=groq)
        self.assertEqual(result.backend, "nemotron")
        self.assertIn("dentist", result.text)

    async def test_local_last_resort(self):
        pcm = (np.sin(np.linspace(0, 6, 8000)) * 4000).astype(np.int16)
        with patch("speech_engines._transcribe_local_sync", return_value="pediatric checkup"):
            result = await transcribe_utterance(pcm, language="en", groq_client=None)
        self.assertEqual(result, TranscriptResult(text="pediatric checkup", backend="local"))

    def test_engine_inventory(self):
        status = list_stt_engines()
        self.assertIn("groq", status)
        self.assertIn("nemotron", status)
        self.assertIn("local", status)

    def test_silent_wav_skips_whisper(self):
        from speech_engines import _wav_is_silence, transcribe_media_bytes
        from stt import pcm_to_wav_bytes

        wav = pcm_to_wav_bytes(np.zeros(16000, dtype=np.int16))
        self.assertTrue(_wav_is_silence(wav))
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as client:
                text = transcribe_media_bytes(wav, "silence.wav")
        self.assertEqual(text, "")
        client.assert_not_called()

    def test_file_transcription_pins_english(self):
        from speech_engines import transcribe_media_bytes
        from stt import pcm_to_wav_bytes

        wav = pcm_to_wav_bytes((np.sin(np.linspace(0, 40, 16000)) * 8000).astype(np.int16))
        created = MagicMock()
        created.audio.transcriptions.create.return_value = _FakeTranscription(
            "I need a cardiology appointment tomorrow at 10 AM"
        )
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("openai.OpenAI", return_value=created):
                text = transcribe_media_bytes(wav, "note.wav")
        kwargs = created.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(kwargs["language"], "en")
        self.assertEqual(kwargs["temperature"], 0)
        self.assertIn("cardiology", kwargs["prompt"])
        self.assertIn("cardiology", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
