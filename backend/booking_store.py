import os
import requests
import logging

logger = logging.getLogger(__name__)

# Placeholder booking store. Replace with Postgres/Firebase/Google Calendar in Phase 7.
_bookings = []

# We use the N8N_WEBHOOK_URL from the environment
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL")

def is_slot_available(date: str, time: str) -> bool:
    # Basic local check. The real availability check happens via n8n 
    # during the save_booking step after confirmation.
    for b in _bookings:
        if b.get("date") == date and b.get("time") == time:
            return False
    return True

def save_booking(slots) -> dict:
    booking_data = {
        "service": slots.service,
        "date": slots.date,
        "time": slots.time,
        "name": slots.name,
        "phone": slots.phone,
    }
    
    if N8N_WEBHOOK_URL:
        try:
            # Send data to n8n webhook
            response = requests.post(N8N_WEBHOOK_URL, json=booking_data, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Expecting something like {"available": true} or {"available": false}
                    if data.get("available") is False:
                        return {"success": False, "message": "Slot is no longer available."}
                except ValueError:
                    # If it's not JSON, assume success if status code is 200
                    pass
            else:
                logger.warning(f"n8n webhook returned status code {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling n8n webhook: {e}")
            # If webhook fails, we can fallback or handle it.
            # We'll proceed with local fallback for now.
            
    # Save to local fallback store
    _bookings.append(booking_data)
    return {"success": True, "booking": booking_data}
