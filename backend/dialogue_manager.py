import re

from booking import BookingSession, BookingState
from extraction import (
    extract_date_fast,
    extract_service_fast,
    extract_time_fast,
    extract_slots,
    get_conversational_reply,
    extract_confirmation,
    add_to_history,
)
from booking_store import is_slot_available, save_booking
from database import answer_catalog_question
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
    if ack:
        # The acknowledgement already opens the sentence; drop the prompt's own opener.
        next_prompt = re.sub(r"^(?:Got it|Cool|Awesome|Perfect|Sure)[!,.]\s*", "", next_prompt)
        next_prompt = next_prompt[:1].upper() + next_prompt[1:]
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


_WEAK_UTTERANCE = re.compile(r"^(?:hmm+|uh+|um+|maybe|wait)(?:\s+\w+){0,1}[.!?]*$", re.I)

_GREETING = re.compile(
    r"^(?:h+e+y+|h+i+|h+e+l+o+|hai+|hii+|namaste|namaskaram|good\s+(?:morning|afternoon|evening)|"
    r"yo|sup)\b[\s,!.\w]{0,24}$",
    re.I,
)
_THANKS = re.compile(r"^(?:thanks?|thank\s+you|thx|ty|tq|cheers)\b[\s,!.\w]{0,24}$", re.I)
_BYE = re.compile(r"^(?:bye|goodbye|see\s+you|that'?s\s+all|nothing\s+else|no\s+thanks?)\b[\s,!.\w]{0,24}$", re.I)
_CAPABILITY = re.compile(
    r"\b(?:what\s+(?:can|do)\s+(?:you|u)\s+do|how\s+can\s+you\s+help|wt\s+can\s+u\s+do|"
    r"what\s+do\s+you\s+offer|what\s+services)\b",
    re.I,
)
_START_OVER = re.compile(r"\b(?:start\s+over|start\s+again|restart|cancel\s+(?:this|that|it)|never\s*mind)\b", re.I)

_CAPABILITY_REPLY = (
    "I can book check-ups and specialist visits, tell you service prices, doctors and "
    "clinic hours, and take down your details. What would you like to do?"
)


def _smalltalk_reply(session: BookingSession, user_text: str) -> str | None:
    """Answer greetings, thanks, goodbyes, and 'what can you do' without a model call."""
    text = (user_text or "").strip()
    if not text:
        return None
    follow_up = PROMPTS.get(session.state)
    if _CAPABILITY.search(text):
        return _CAPABILITY_REPLY
    if _GREETING.match(text):
        if session.state in (BookingState.GREETING, BookingState.COLLECT_SERVICE):
            return "Hi there! I'm Elena from Meridian Health. I can book an appointment or answer questions about our services. How can I help?"
        return f"Hi! I'm still here. {follow_up}" if follow_up else "Hi! How can I help?"
    if _THANKS.match(text):
        return "You're very welcome! Is there anything else I can help you with?"
    if _BYE.match(text):
        return "Take care, and have a lovely day! Reach out any time you need an appointment."
    return None


def handle_turn(session: BookingSession, user_text: str) -> str:
    """Given the current session and what the user just said, advance the
    state machine and return the assistant's next spoken reply.

    Safety blocks come from the Belfry gateway on the model call, not from
    a local keyword refusal.
    """

    if session.state == BookingState.GREETING:
        session.advance()

    if _START_OVER.search(user_text or "") and session.state != BookingState.COLLECT_SERVICE:
        session.state = BookingState.COLLECT_SERVICE
        session.slots = type(session.slots)()
        reply = "No problem, let's start fresh. What kind of appointment are you looking for?"
        _remember(session, "user", user_text)
        _remember(session, "assistant", reply)
        return reply

    # Pure small talk ("thank you", "bye") must not be read as a name or service.
    # Anything that also carries a service, date, or time still goes to slot filling.
    carries_booking_info = bool(
        extract_service_fast(user_text) or extract_date_fast(user_text) or extract_time_fast(user_text)
    )
    if not carries_booking_info:
        early = _smalltalk_reply(session, user_text)
        if early:
            _remember(session, "user", user_text)
            _remember(session, "assistant", early)
            return early

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

    if session.state == BookingState.CONFIRM:
        return _confirmation_reply(session)

    if session.state == BookingState.BOOKED:
        session.state = BookingState.COLLECT_SERVICE
        session.slots = type(session.slots)()
        reply = "You're already booked. Want to schedule another appointment?"
        _remember(session, "user", user_text)
        _remember(session, "assistant", reply)
        return reply

    prompt = PROMPTS.get(session.state, "Could you say that again?")
    smalltalk = _smalltalk_reply(session, user_text)
    if smalltalk:
        reply = smalltalk
    elif _WEAK_UTTERANCE.match((user_text or "").strip()):
        reply = prompt
    else:
        reply = answer_catalog_question(user_text) or get_conversational_reply(
            user_text, prompt, history=_history(session)
        ) or prompt
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
