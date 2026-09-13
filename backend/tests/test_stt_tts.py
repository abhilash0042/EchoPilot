"""Offline tests for STT audio prep, transcript cleanup, and TTS speak-out."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stt import (
    cleanup_transcript,
    is_low_confidence,
    pick_stt_language,
    prepare_utterance,
    trim_silence,
)
from tts import prepare_speech_text


class TestSttPrep(unittest.TestCase):
    def test_trim_keeps_speech_core(self):
        silence = np.zeros(16000, dtype=np.int16)
        tone = (np.sin(2 * np.pi * 300 * np.linspace(0, 0.4, 6400)) * 4000).astype(np.int16)
        pcm = np.concatenate([silence, tone, silence])
        trimmed = trim_silence(pcm, noise_floor=80.0)
        self.assertLess(len(trimmed), len(pcm))
        self.assertGreater(len(trimmed), 4000)

    def test_prepare_raises_quiet_speech(self):
        quiet = (np.sin(2 * np.pi * 220 * np.linspace(0, 0.5, 8000)) * 400).astype(np.int16)
        prepared = prepare_utterance(quiet, noise_floor=50.0)
        rms = float(np.sqrt(np.mean(prepared.astype(np.float32) ** 2)))
        self.assertGreater(rms, 800)

    def test_transcript_cleanup(self):
        self.assertIn("ophthalmology", cleanup_transcript("book opthalmology tomorrow"))
        self.assertIn("tomorrow", cleanup_transcript("to morrow morning"))
        self.assertIn("AM", cleanup_transcript("11 a.m."))

    def test_language_pin(self):
        self.assertEqual(pick_stt_language("I need cardiology"), "en")
        self.assertIsNone(pick_stt_language("నమస్కారం checkup"))

    def test_confidence_gate(self):
        self.assertTrue(is_low_confidence(0.92, -1.4))
        self.assertFalse(is_low_confidence(0.92, -0.2))
        self.assertFalse(is_low_confidence(None, None))


class TestTtsSpeakable(unittest.TestCase):
    def test_dates_times_phones(self):
        spoken = prepare_speech_text(
            "Cardiology on 2026-09-14 at 14:30, phone 5551234567"
        )
        self.assertIn("September", spoken)
        self.assertIn("14th", spoken)
        self.assertIn("2:30 PM", spoken)
        self.assertIn("5 5 5 1 2 3 4 5 6 7", spoken)
        self.assertNotIn("2026-09-14", spoken)
        self.assertTrue(spoken.endswith("."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
