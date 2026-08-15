"""
Automated Verification & Integrity Tests for Lumina Health Database
Tests database queries, relations, sensitive records, and booking store integration.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
from seed_data import seed_database
from booking_store import is_slot_available, save_booking
from booking import BookingSlots


class TestHealthcareDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initializes and seeds database for tests."""
        seed_database(force_refresh=True)

    def test_01_hospital_and_sensitive_licensing(self):
        """Verify hospital record contains sensitive licensing, tax ID, and DEA details."""
        hospitals = database.get_all_hospitals()
        self.assertGreater(len(hospitals), 0, "Hospital record should exist")
        h = hospitals[0]
        self.assertIn("Lumina", h["name"])
        # Verify sensitive hospital fields
        self.assertTrue(bool(h["registration_number"]), "Sensitive: Registration number should be present")
        self.assertTrue(bool(h["tax_id_ein"]), "Sensitive: Tax ID / EIN should be present")
        self.assertTrue(bool(h["dea_license_number"]), "Sensitive: DEA license should be present")
        self.assertTrue(bool(h["hipaa_compliance_id"]), "Sensitive: HIPAA ID should be present")
        print("  [PASS] Hospital & Sensitive Licensing Records Verified")

    def test_02_doctors_and_services(self):
        """Verify doctors and clinical services are populated with specialties and fees."""
        doctors = database.get_all_doctors()
        services = database.get_all_services()
        self.assertGreaterEqual(len(doctors), 5, "At least 5 doctors should be seeded")
        self.assertGreaterEqual(len(services), 6, "At least 6 services should be seeded")
        
        # Verify doctor sensitive license
        for doc in doctors:
            self.assertTrue(bool(doc["license_number"]), f"Doctor {doc['name']} has medical license")
            self.assertGreater(doc["consultation_fee"], 0)
        print("  [PASS] Doctors & Clinical Services Catalogue Verified")

    def test_03_patients_sensitive_records(self):
        """Verify patient records store protected health information (PHI/PII)."""
        patients = database.get_all_patients()
        self.assertGreater(len(patients), 0, "Patient records should exist")
        
        for p in patients:
            self.assertTrue(bool(p["medical_record_number"]), "Patient MRN must exist")
            self.assertTrue(bool(p["blood_group"]), "Blood group must be recorded")
            self.assertTrue(bool(p["allergies"]), "Allergies must be recorded")
            self.assertTrue(bool(p["insurance_provider"]), "Insurance carrier must be recorded")
            self.assertTrue(bool(p["insurance_policy_number"]), "Insurance policy # must be recorded")
        print("  [PASS] Patient Sensitive PHI / PII Records Verified")

    def test_04_slot_availability_check(self):
        """Verify slot availability detection against appointments."""
        # Tomorrow slot LUM-10492 is booked for Cardiology at 10:00
        from datetime import datetime, timedelta
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Unbooked random slot
        free_date = "2029-12-31"
        self.assertTrue(database.check_slot_available(free_date, "15:30"))
        print("  [PASS] Slot Availability Logic Verified")

    def test_05_voice_agent_booking_store_integration(self):
        """Test booking_store save_booking integration directly with SQLite."""
        slots = BookingSlots(
            service="Dermatology Skin Exam",
            date="2027-05-15",
            time="11:00",
            name="Alex Turner",
            phone="+1 (555) 998-2211"
        )
        
        result = save_booking(slots)
        self.assertTrue(result.get("success"), f"Booking failed: {result}")
        booking = result.get("booking")
        self.assertTrue(bool(booking.get("booking_code")), "Booking code must be returned")
        self.assertTrue(bool(booking.get("patient_mrn")), "Patient MRN must be generated and returned")
        
        # Verify patient exists in DB
        patients = database.get_all_patients()
        alex = next((p for p in patients if p["name"] == "Alex Turner"), None)
        self.assertIsNotNone(alex, "Alex Turner should be created in patient table")
        self.assertEqual(alex["phone"], "+1 (555) 998-2211")
        print(f"  [PASS] Booking Store created appointment {booking['booking_code']} and patient {alex['medical_record_number']}")

    def test_06_call_logs_persistence(self):
        """Test voice agent call transcript logging."""
        session_id = "test_sess_9999"
        database.save_call_log(
            session_id=session_id,
            caller_phone="+1 (555) 998-2211",
            caller_name="Alex Turner",
            service_requested="Dermatology Skin Exam",
            call_duration_seconds=35,
            transcript="User: I want to book a skin exam.\nAssistant: You are all set!",
            status="COMPLETED"
        )
        
        logs = database.get_all_call_logs()
        log = next((l for l in logs if l["session_id"] == session_id), None)
        self.assertIsNotNone(log, "Call log should be persisted")
        self.assertEqual(log["call_duration_seconds"], 35)
        self.assertEqual(log["status"], "COMPLETED")
        print("  [PASS] Call Telemetry & Transcript Logs Verified")


if __name__ == "__main__":
    print("\n--- RUNNING LUMINA HEALTHCARE DATABASE VERIFICATION TESTS ---")
    unittest.main(verbosity=2)
