"""Offline tests for STT audio prep, transcript cleanup, and TTS speak-out."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stt import (
    cleanup_transcript,
    correct_clinic_words,
    is_echo_transcript,
    is_hallucination,
    is_ignore_utterance,
    is_low_confidence,
    pcm_to_wav_bytes,
    pick_stt_language,
    prepare_utterance,
    trim_silence,
)
from tts import has_telugu, prepare_speech_text, split_spoken_sentences


class TestSttPrep(unittest.TestCase):
    def test_trim_keeps_speech_core(self):
        silence = np.zeros(16000, dtype=np.int16)
        tone = (np.sin(2 * np.pi * 300 * np.linspace(0, 0.4, 6400)) * 4000).astype(np.int16)
        pcm = np.concatenate([silence, tone, silence])
        trimmed = trim_silence(pcm, noise_floor=80.0)
        self.assertLess(len(trimmed), len(pcm))
        self.assertGreater(len(trimmed), 4000)
        # Word-edge pad should keep more than the old 80ms crop.
        self.assertGreaterEqual(len(trimmed), 6400)

    def test_prepare_raises_quiet_speech(self):
        quiet = (np.sin(2 * np.pi * 220 * np.linspace(0, 0.5, 8000)) * 400).astype(np.int16)
        prepared = prepare_utterance(quiet, noise_floor=50.0)
        rms = float(np.sqrt(np.mean(prepared.astype(np.float32) ** 2)))
        self.assertGreater(rms, 800)

    def test_transcript_cleanup(self):
        self.assertIn("ophthalmology", cleanup_transcript("book opthalmology tomorrow"))
        self.assertIn("Abhilash", cleanup_transcript("Hey hi I'm a blast"))
        self.assertEqual(cleanup_transcript("I AM a blush."), "Abhilash")
        self.assertEqual(cleanup_transcript("I am a blush"), "Abhilash")
        self.assertIn("cardiology", correct_clinic_words("book cardialogy tomorrow"))
        self.assertIn("I need", cleanup_transcript("I did want an appointment"))
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

    def test_sentence_split_for_streaming_tts(self):
        chunks = split_spoken_sentences(
            "Got it — cardiology. What day works best for you?"
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith("."))

    def test_mixed_english_not_forced_telugu_voice(self):
        self.assertFalse(has_telugu("I need cardiology tomorrow"))
        self.assertTrue(has_telugu("నమస్కారం నాకు చెకప్ కావాలి"))

    def test_hallucination_filter(self):
        self.assertTrue(is_hallucination("Thanks for watching"))
        self.assertTrue(is_hallucination("Thank you"))
        self.assertTrue(is_hallucination("Thanks."))
        self.assertTrue(is_hallucination("thank you so much"))
        self.assertTrue(is_ignore_utterance("No problem"))
        self.assertTrue(is_ignore_utterance("Cooldown"))
        self.assertFalse(is_hallucination("I need a cardiology appointment tomorrow"))
        self.assertFalse(is_hallucination("Can I book cardiology tomorrow"))

    def test_echo_of_assistant_is_dropped(self):
        greeting = "Hey there! I'm Elena from the health clinic. What can I help you with today?"
        leaked = "I'm here to help you with your surgery."
        self.assertTrue(is_echo_transcript(leaked, greeting))
        self.assertFalse(is_echo_transcript("cardiology tomorrow morning", greeting))

    def test_repeat_transcript_is_dropped(self):
        from stt import is_repeat_transcript

        self.assertTrue(is_repeat_transcript("Thank you.", "thank you"))
        self.assertFalse(is_repeat_transcript("cardiology", "Thank you"))

    def test_pcm_writes_wav_header(self):
        pcm = np.zeros(1600, dtype=np.int16)
        wav = pcm_to_wav_bytes(pcm)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav[:16])


if __name__ == "__main__":
    unittest.main(verbosity=2)
