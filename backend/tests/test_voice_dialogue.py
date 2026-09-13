"""Offline tests for natural voice turns: multi-slot extract + dialogue."""
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from booking import BookingSession, BookingState
from extraction import (
    extract_service_fast,
    extract_date_fast,
    extract_time_fast,
    extract_phone_fast,
    extract_name_fast,
    extract_confirmation,
    extract_slots,
    clean_spoken_text,
)
from dialogue_manager import handle_turn, handle_confirmation


class TestFastExtractors(unittest.TestCase):
    def test_service_keywords(self):
        self.assertEqual(extract_service_fast("I need a cardiology appointment"), "cardiology")
        self.assertEqual(extract_service_fast("skin rash please"), "dermatology")
        self.assertEqual(extract_service_fast("full body checkup"), "general checkup")

    def test_relative_dates(self):
        today = date(2026, 9, 13)
        self.assertEqual(extract_date_fast("tomorrow morning", today), "2026-09-14")
        self.assertEqual(extract_date_fast("day after tomorrow", today), "2026-09-15")
        monday = extract_date_fast("next Monday", today)
        self.assertEqual(monday, "2026-09-14")  # upcoming Monday from Sunday the 13th

    def test_times(self):
        self.assertEqual(extract_time_fast("11 am"), "11:00")
        self.assertEqual(extract_time_fast("2:30 pm"), "14:30")
        self.assertEqual(extract_time_fast("half past 10"), "10:30")
        self.assertEqual(extract_time_fast("in the evening"), "18:00")
        self.assertEqual(extract_time_fast("day after tomorrow at 10"), "10:00")
        self.assertIsNone(extract_time_fast("I am 32 years old"))

    def test_phone_and_name(self):
        self.assertEqual(extract_phone_fast("reach me at 555-123-4567"), "5551234567")
        self.assertEqual(extract_name_fast("my name is Priya Sharma"), "Priya Sharma")
        self.assertIsNone(extract_name_fast("I am looking for a dentist"))

    def test_confirmation_fast_path(self):
        self.assertEqual(extract_confirmation("yes"), "yes")
        self.assertEqual(extract_confirmation("Sounds good!"), "yes")
        self.assertEqual(extract_confirmation("no"), "no")
        self.assertEqual(extract_confirmation("change"), "no")

    def test_spoken_cleanup(self):
        self.assertEqual(clean_spoken_text('**Sure thing!**'), "Sure thing!")

    def test_multi_slot_fast_only(self):
        slots = extract_slots(
            "I need cardiology tomorrow at 11am, my name is Riya, phone 5551234567"
        )
        self.assertEqual(slots["service"], "cardiology")
        self.assertEqual(slots["time"], "11:00")
        self.assertEqual(slots["name"], "Riya")
        self.assertEqual(slots["phone"], "5551234567")
        self.assertIn("date", slots)


class TestDialogueVoiceTurns(unittest.TestCase):
    def test_one_sentence_fills_multiple_slots(self):
        session = BookingSession()
        with patch("dialogue_manager.extract_slots", return_value={
            "service": "cardiology",
            "date": "2026-09-14",
            "time": "11:00",
        }):
            reply = handle_turn(session, "cardiology tomorrow at 11")
        self.assertEqual(session.slots.service, "cardiology")
        self.assertEqual(session.slots.date, "2026-09-14")
        self.assertEqual(session.slots.time, "11:00")
        self.assertEqual(session.state, BookingState.COLLECT_NAME)
        self.assertIn("name", reply.lower())
        self.assertIn("cardiology", reply.lower())

    def test_full_dump_reaches_confirm(self):
        session = BookingSession()
        with patch("dialogue_manager.extract_slots", return_value={
            "service": "dentist",
            "date": "2026-09-15",
            "time": "10:00",
            "name": "Asha",
            "phone": "5550001111",
        }), patch("dialogue_manager.is_slot_available", return_value=True):
            reply = handle_turn(session, "dentist day after tomorrow at 10 for Asha 5550001111")
        self.assertEqual(session.state, BookingState.CONFIRM)
        self.assertIn("sound good", reply.lower())

    def test_sessions_do_not_share_history(self):
        a = BookingSession()
        b = BookingSession()
        with patch("dialogue_manager.extract_slots", return_value={"service": "cardiology"}):
            handle_turn(a, "cardiology")
        self.assertTrue(a.conversation_history)
        self.assertFalse(b.conversation_history)

    def test_yes_confirms_without_llm(self):
        session = BookingSession()
        session.state = BookingState.CONFIRM
        session.slots.service = "cardiology"
        session.slots.date = "2026-09-14"
        session.slots.time = "11:00"
        session.slots.name = "Asha"
        session.slots.phone = "5550001111"
        with patch("dialogue_manager.save_booking", return_value={"success": True}):
            reply = handle_confirmation(session, "yes")
        self.assertEqual(session.state, BookingState.BOOKED)
        self.assertIn("all set", reply.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
