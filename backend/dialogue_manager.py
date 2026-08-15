from booking import BookingSession, BookingState
from extraction import extract_field, get_conversational_reply, extract_confirmation, add_to_history, reset_history
from booking_store import is_slot_available, save_booking
from datetime import datetime

PROMPTS = {
    BookingState.GREETING: "Hey there! I'm Lumina from the health clinic. What can I help you with today?",
    BookingState.COLLECT_SERVICE: "So what kind of appointment are you looking for?",
    BookingState.COLLECT_DATE: "Cool! And what day works best for you?",
    BookingState.COLLECT_TIME: "Got it! What time would you like to come in?",
    BookingState.COLLECT_NAME: "Awesome! Can I get your name please?",
    BookingState.COLLECT_PHONE: "Perfect, and what's a good phone number to reach you at?",
}

def handle_turn(session: BookingSession, user_text: str) -> str:
    """Given the current session and what the user just said, advance the
    state machine and return the assistant's next spoken reply."""

    if session.state == BookingState.GREETING:
        session.advance()  # move to COLLECT_SERVICE, but the user's first
                            # utterance might already contain the service —
                            # so fall through and try extracting immediately

    field = session.next_prompt_field()

    if field:
        value = extract_field(field, user_text)
        if value:
            setattr(session.slots, field, value)
            session.advance()
        else:
            # Couldn't extract — use LLM to chit-chat and steer back naturally
            reply = get_conversational_reply(user_text, PROMPTS[session.state])
            add_to_history("user", user_text)
            add_to_history("assistant", reply)
            return reply

    # Check if we've just filled the last slot and should move to confirm
    if session.state == BookingState.CONFIRM:
        s = session.slots
        if not is_slot_available(s.date, s.time):
            # Slot taken — bump back to re-ask for time
            session.state = BookingState.COLLECT_TIME
            return f"Sorry, {s.time} on {s.date} is already booked. What other time works for you?"
            
        # Format date and time for TTS
        try:
            date_obj = datetime.strptime(s.date, "%Y-%m-%d")
            spoken_date = date_obj.strftime("%A, %B ") + str(date_obj.day)
        except:
            spoken_date = s.date
            
        try:
            time_obj = datetime.strptime(s.time, "%H:%M")
            spoken_time = time_obj.strftime("%I:%M %p").lstrip('0')
        except:
            spoken_time = s.time

        return (
            f"Alright, just to make sure I got everything right: "
            f"{s.service} appointment for {s.name} on {spoken_date} "
            f"at {spoken_time}, and I'll reach you at {s.phone}. Sound good?"
        )

    if session.state == BookingState.BOOKED:
        session.state = BookingState.COLLECT_SERVICE
        session.slots = type(session.slots)()
        # Freeform mode: just chat naturally, don't push for another booking
        reply = get_conversational_reply(user_text, "Is there anything else I can help you with?", allow_freeform=True)
        add_to_history("user", user_text)
        add_to_history("assistant", reply)
        return reply

    # Still mid-flow — ask for the next field
    next_field_prompt = PROMPTS.get(session.state)
    reply = next_field_prompt or "Tell me more!"
    add_to_history("user", user_text)
    add_to_history("assistant", reply)
    return reply

def handle_confirmation(session: BookingSession, user_text: str) -> str:
    """Called specifically when we're in CONFIRM state, waiting for yes/no."""
    conf = extract_confirmation(user_text)
    
    if conf == "yes":
        save_booking(session.slots)
        session.state = BookingState.BOOKED
        reply = "You're all set! I've booked that for you. You'll get a confirmation message shortly. Is there anything else you need?"
        add_to_history("user", user_text)
        add_to_history("assistant", reply)
        return reply
    elif conf == "no":
        session.state = BookingState.COLLECT_SERVICE
        session.slots = type(session.slots)()  # reset slots
        reply = "No worries at all! Let's start fresh. What kind of appointment are you looking for?"
        add_to_history("user", user_text)
        add_to_history("assistant", reply)
        return reply
    else:
        # Ambiguous response — use LLM to politely clarify
        reply = get_conversational_reply(user_text, "Does everything look good? Just say yes or no and we're done!")
        add_to_history("user", user_text)
        add_to_history("assistant", reply)
        return reply
