from booking import BookingSession, BookingState
from extraction import (
    extract_slots,
    get_conversational_reply,
    extract_confirmation,
    add_to_history,
)
from booking_store import is_slot_available, save_booking
from datetime import datetime

PROMPTS = {
    BookingState.GREETING: "Hey there! I'm Elena from the health clinic. What can I help you with today?",
    BookingState.COLLECT_SERVICE: "So what kind of appointment are you looking for?",
    BookingState.COLLECT_DATE: "Cool! And what day works best for you?",
    BookingState.COLLECT_TIME: "Got it! What time would you like to come in?",
    BookingState.COLLECT_NAME: "Awesome! Can I get your name please?",
    BookingState.COLLECT_PHONE: "Perfect, and what's a good phone number to reach you at?",
}

_FIELD_ORDER = ["service", "date", "time", "name", "phone"]


def _history(session: BookingSession) -> list:
    return session.conversation_history


def _remember(session: BookingSession, role: str, content: str) -> None:
    add_to_history(role, content, history=_history(session))


def _spoken_date(value: str) -> str:
    try:
        date_obj = datetime.strptime(value, "%Y-%m-%d")
        return date_obj.strftime("%A, %B ") + str(date_obj.day)
    except Exception:
        return value


def _spoken_time(value: str) -> str:
    try:
        time_obj = datetime.strptime(value, "%H:%M")
        return time_obj.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return value


def _remaining_fields(session: BookingSession) -> list[str]:
    if session.state == BookingState.GREETING:
        return list(_FIELD_ORDER)
    current = session.next_prompt_field()
    if not current:
        return []
    start = _FIELD_ORDER.index(current)
    return [field for field in _FIELD_ORDER[start:] if not getattr(session.slots, field)]


def _acknowledge_and_ask(session: BookingSession, newly_filled: list[str]) -> str:
    s = session.slots
    bits = []
    if "service" in newly_filled and s.service:
        bits.append(s.service)
    if "date" in newly_filled and s.date:
        bits.append(_spoken_date(s.date))
    if "time" in newly_filled and s.time:
        bits.append(f"at {_spoken_time(s.time)}")
    if "name" in newly_filled and s.name:
        bits.append(f"for {s.name}")
    ack = ("Got it — " + ", ".join(bits) + ". ") if bits else ""
    next_prompt = PROMPTS.get(session.state, "Tell me more!")
    return f"{ack}{next_prompt}".strip()


def _confirmation_reply(session: BookingSession) -> str:
    s = session.slots
    if not is_slot_available(s.date, s.time):
        session.state = BookingState.COLLECT_TIME
        reply = (
            f"Sorry, {_spoken_time(s.time)} on {_spoken_date(s.date)} is already booked. "
            "What other time works for you?"
        )
        _remember(session, "assistant", reply)
        return reply

    reply = (
        f"Alright, just to make sure I got everything right: "
        f"{s.service} appointment for {s.name} on {_spoken_date(s.date)} "
        f"at {_spoken_time(s.time)}, and I'll reach you at {s.phone}. Sound good?"
    )
    _remember(session, "assistant", reply)
    return reply


def handle_turn(session: BookingSession, user_text: str) -> str:
    """Given the current session and what the user just said, advance the
    state machine and return the assistant's next spoken reply."""

    if session.state == BookingState.GREETING:
        session.advance()

    remaining = _remaining_fields(session)
    newly_filled: list[str] = []
    if remaining:
        extracted = extract_slots(user_text, remaining)
        print(f"[NLU] fields={remaining} transcript='{user_text[:100]}' extracted={extracted}")
        for field, value in extracted.items():
            if value:
                setattr(session.slots, field, value)
                newly_filled.append(field)
        session.skip_filled()

    if newly_filled:
        _remember(session, "user", user_text)
        if session.state == BookingState.CONFIRM:
            return _confirmation_reply(session)
        reply = _acknowledge_and_ask(session, newly_filled)
        _remember(session, "assistant", reply)
        return reply

    clean_text = user_text.strip().lower().rstrip(".!?,")
    politeness_tokens = {
        "thank you", "thanks", "thank you so much", "thank you very much",
        "thx", "okay thank you", "ok thank you", "i want to thank you",
        "thank you and thank others",
    }
    if clean_text in politeness_tokens or clean_text.startswith("thank you"):
        reply = f"Happy to help! {PROMPTS[session.state]}"
        _remember(session, "user", user_text)
        _remember(session, "assistant", reply)
        return reply

    if session.state == BookingState.CONFIRM:
        return _confirmation_reply(session)

    if session.state == BookingState.BOOKED:
        session.state = BookingState.COLLECT_SERVICE
        session.slots = type(session.slots)()
        reply = get_conversational_reply(
            user_text,
            "Is there anything else I can help you with?",
            allow_freeform=True,
            history=_history(session),
        )
        _remember(session, "user", user_text)
        _remember(session, "assistant", reply)
        return reply

    reply = get_conversational_reply(user_text, PROMPTS[session.state], history=_history(session))
    _remember(session, "user", user_text)
    _remember(session, "assistant", reply)
    return reply


def handle_confirmation(session: BookingSession, user_text: str) -> str:
    """Called specifically when we're in CONFIRM state, waiting for yes/no."""
    conf = extract_confirmation(user_text)

    if conf == "yes":
        result = save_booking(session.slots)
        if result.get("success"):
            session.state = BookingState.BOOKED
            reply = (
                "You're all set! I've booked that for you. "
                "You'll get a confirmation message shortly. Is there anything else you need?"
            )
        else:
            session.state = BookingState.COLLECT_TIME
            reply = (
                f"Sorry, I couldn't lock that slot. "
                f"What other time works for you instead of {session.slots.time} on {session.slots.date}?"
            )

        _remember(session, "user", user_text)
        _remember(session, "assistant", reply)
        return reply
    elif conf == "no":
        session.state = BookingState.COLLECT_SERVICE
        session.slots = type(session.slots)()
        reply = "No worries at all! Let's start fresh. What kind of appointment are you looking for?"
        _remember(session, "user", user_text)
        _remember(session, "assistant", reply)
        return reply
    else:
        reply = get_conversational_reply(
            user_text,
            "Does everything look good? Just say yes or no and we're done!",
            history=_history(session),
        )
        _remember(session, "user", user_text)
        _remember(session, "assistant", reply)
        return reply
