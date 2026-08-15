# Placeholder booking store. Replace with Postgres/Firebase/Google Calendar in Phase 7.
_bookings = []

def is_slot_available(date: str, time: str) -> bool:
    for b in _bookings:
        if b.get("date") == date and b.get("time") == time:
            return False
    return True

def save_booking(slots) -> dict:
    booking = {
        "service": slots.service,
        "date": slots.date,
        "time": slots.time,
        "name": slots.name,
        "phone": slots.phone,
    }
    _bookings.append(booking)
    return booking
