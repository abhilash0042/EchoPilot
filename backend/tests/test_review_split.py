"""File and image review stay off the booking path."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("edge_tts", types.ModuleType("edge_tts"))

import main
from fastapi.testclient import TestClient


class TestReviewSplit(unittest.TestCase):
    def setUp(self):
        main.chat_sessions.clear()
        self.client = TestClient(main.app)

    def test_file_review_does_not_open_a_booking(self):
        with patch.object(main, "_review_with_model", return_value="The note refers Ana to dermatology on Friday at 2 PM."):
            response = self.client.post(
                "/api/review/upload",
                data={"message": "What does this ask for?", "kind": "file", "belfry_enabled": "false"},
                files={"file": ("referral.txt", b"Please refer Ana to dermatology on Friday at 2 PM.", "text/plain")},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["purpose"], "review")
        self.assertIn("dermatology", body["reply"].lower())
        self.assertNotIn("slots", body)
        self.assertEqual(main.chat_sessions, {})

    def test_chat_upload_of_a_file_does_not_book(self):
        with patch.object(main, "_review_with_model", return_value="This file is a referral, not a completed booking."):
            response = self.client.post(
                "/api/chat/upload",
                data={"session_id": "should-stay-empty", "message": "", "kind": "file", "belfry_enabled": "false"},
                files={"file": ("note.txt", b"Referral for Ana, dermatology, Friday.", "text/plain")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["purpose"], "review")
        self.assertNotIn("should-stay-empty", main.chat_sessions)

    def test_text_message_still_books(self):
        with patch.object(main, "handle_turn", return_value="What date works for cardiology?") as turn:
            response = self.client.post(
                "/api/chat/message",
                json={
                    "session_id": "book-text",
                    "message": "I need a cardiology appointment",
                    "belfry_enabled": False,
                },
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("slots", body)
        self.assertNotEqual(body.get("purpose"), "review")
        turn.assert_called_once()
        self.assertIn("book-text", main.chat_sessions)
