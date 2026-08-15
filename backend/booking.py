from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class BookingState(str, Enum):
    GREETING = "GREETING"
    COLLECT_SERVICE = "COLLECT_SERVICE"
    COLLECT_DATE = "COLLECT_DATE"
    COLLECT_TIME = "COLLECT_TIME"
    COLLECT_NAME = "COLLECT_NAME"
    COLLECT_PHONE = "COLLECT_PHONE"
    CONFIRM = "CONFIRM"
    BOOKED = "BOOKED"

@dataclass
class BookingSlots:
    service: Optional[str] = None
    date: Optional[str] = None       # ISO format YYYY-MM-DD
    time: Optional[str] = None       # HH:MM (24hr)
    name: Optional[str] = None
    phone: Optional[str] = None

@dataclass
class BookingSession:
    state: BookingState = BookingState.GREETING
    slots: BookingSlots = field(default_factory=BookingSlots)

    def next_prompt_field(self) -> str:
        """What field are we currently trying to fill?"""
        mapping = {
            BookingState.COLLECT_SERVICE: "service",
            BookingState.COLLECT_DATE: "date",
            BookingState.COLLECT_TIME: "time",
            BookingState.COLLECT_NAME: "name",
            BookingState.COLLECT_PHONE: "phone",
        }
        return mapping.get(self.state)

    def advance(self):
        order = [
            BookingState.GREETING,
            BookingState.COLLECT_SERVICE,
            BookingState.COLLECT_DATE,
            BookingState.COLLECT_TIME,
            BookingState.COLLECT_NAME,
            BookingState.COLLECT_PHONE,
            BookingState.CONFIRM,
            BookingState.BOOKED,
        ]
        idx = order.index(self.state)
        if idx < len(order) - 1:
            self.state = order[idx + 1]
