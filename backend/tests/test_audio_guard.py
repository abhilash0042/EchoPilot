"""Uploaded audio books a real request and ignores Whisper filler."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("edge_tts", types.ModuleType("edge_tts"))

import main
from fastapi.testclient import TestClient


def _wav() -> bytes:
    return b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32


class TestAudioGuard(unittest.TestCase):
    def setUp(self):
        main.chat_sessions.clear()
        self.client = TestClient(main.app)

    def test_stock_phrase_does_not_book(self):
        with patch("speech_engines.transcribe_media_bytes", return_value="Thanks for watching!"):
            with patch.object(main, "handle_turn") as turn:
                response = self.client.post(
                    "/api/chat/upload",
                    data={"session_id": "audio-halluc", "message": "", "kind": "audio", "belfry_enabled": "false"},
                    files={"file": ("note.wav", _wav(), "audio/wav")},
                )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["purpose"], "booking")
        self.assertIsNone(body["slots"]["service"])
        self.assertIn("didn't catch", body["reply"])
        self.assertNotIn("audio-halluc", main.chat_sessions)
        turn.assert_not_called()

    def test_clear_request_still_books(self):
        with patch(
            "speech_engines.transcribe_media_bytes",
            return_value="Please book a cardiology appointment tomorrow at 10 AM",
        ):
            with patch.object(main, "handle_turn", return_value="What is your name?") as turn:
                response = self.client.post(
                    "/api/chat/upload",
                    data={"session_id": "audio-book", "message": "", "kind": "audio", "belfry_enabled": "false"},
                    files={"file": ("note.wav", _wav(), "audio/wav")},
                )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["purpose"], "booking")
        self.assertIn("name", body["reply"].lower())
        turn.assert_called_once()
        spoken = turn.call_args.args[1]
        self.assertIn("cardiology", spoken.lower())
        self.assertIn("audio-book", main.chat_sessions)

    def test_typed_caption_books_when_the_clip_is_silent(self):
        with patch("speech_engines.transcribe_media_bytes", return_value=""):
            with patch.object(main, "handle_turn", return_value="What time works?") as turn:
                response = self.client.post(
                    "/api/chat/upload",
                    data={
                        "session_id": "audio-caption",
                        "message": "I need a dermatology appointment on Friday at 2 PM",
                        "kind": "audio",
                        "belfry_enabled": "false",
                    },
                    files={"file": ("note.wav", _wav(), "audio/wav")},
                )
        self.assertEqual(response.status_code, 200)
        turn.assert_called_once()
        self.assertIn("dermatology", turn.call_args.args[1].lower())
        self.assertNotIn("didn't catch", response.json()["reply"])
