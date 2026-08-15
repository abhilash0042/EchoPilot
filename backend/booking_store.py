import os
import requests
import logging
from typing import Dict, Any

from database import check_slot_available, create_appointment, get_or_create_patient

logger = logging.getLogger(__name__)

# Optional external webhook integration (e.g., n8n, Zapier, Google Calendar)
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL")

def is_slot_available(date: str, time: str) -> bool:
    """Checks real-time appointment availability against the SQLite database."""
    try:
        return check_slot_available(date, time)
    except Exception as e:
        logger.error(f"Error checking slot availability: {e}")
        return True

def save_booking(slots) -> Dict[str, Any]:
    """Saves booking into the database, updates/creates patient record, and optionally triggers webhook."""
    try:
        # 1. Persist directly into SQLite database
        appointment = create_appointment(
            service_name=slots.service or "General Consultation",
            date=slots.date,
            time=slots.time,
            patient_name=slots.name or "Guest Patient",
            patient_phone=slots.phone or "000-000-0000",
            booked_via="AI_VOICE_AGENT"
        )
        
        booking_data = {
            "booking_code": appointment.get("booking_code"),
            "service": slots.service,
            "date": slots.date,
            "time": slots.time,
            "name": slots.name,
            "phone": slots.phone,
            "doctor": appointment.get("doctor_name"),
            "patient_mrn": appointment.get("patient_mrn"),
            "status": "CONFIRMED"
        }

        # 2. Optionally forward to external webhook (n8n / CRM) if configured
        if N8N_WEBHOOK_URL:
            try:
                response = requests.post(N8N_WEBHOOK_URL, json=booking_data, timeout=5)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data.get("available") is False:
                            return {"success": False, "message": "Slot is no longer available."}
                    except ValueError:
                        pass
            except Exception as ex:
                logger.warning(f"Optional n8n webhook notification error: {ex}")

        return {"success": True, "booking": booking_data, "appointment": appointment}
    except Exception as e:
        logger.error(f"Failed to save booking to database: {e}")
        return {"success": False, "message": str(e)}

