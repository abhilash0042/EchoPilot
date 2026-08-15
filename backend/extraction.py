import os
import json
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if GROQ_API_KEY:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
    MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.2-3b-preview")
    print(f"[LLM] Connected to Groq Cloud API ({MODEL_NAME})")
else:
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    MODEL_NAME = "llama3.2"
    print(f"[LLM] Connected to local Ollama ({MODEL_NAME})")


FIELD_PROMPTS = {
    "service": """Extract the medical service/department the caller wants (e.g. "general checkup",
"dentist", "cardiology", "eye checkup"). If unclear, return null.""",

    "date": """Extract the appointment date the caller mentioned, and convert it to ISO format
YYYY-MM-DD. Today's date is {today}. Handle relative terms like "tomorrow",
"next Monday", "day after tomorrow". If unclear or not mentioned, return null.""",

    "time": """Extract the appointment time and convert to 24-hour HH:MM format.
Handle terms like "morning" (assume 10:00), "afternoon" (assume 14:00), "evening" (assume 18:00),
or specific times like "3pm" -> "15:00". If unclear, return null.""",

    "name": """Extract the caller's full name. If unclear or not mentioned, return null.""",

    "phone": """Extract the caller's phone number, digits only, no spaces or symbols.
If unclear or not mentioned, return null.""",
}

def extract_field(field_name: str, user_text: str) -> str | None:
    instruction = FIELD_PROMPTS.get(field_name, "")
    if field_name == "date":
        instruction = instruction.format(today=date.today().isoformat())
        
    system = f"""{instruction}
Respond with ONLY a JSON object like {{"value": "..."}} or {{"value": null}}.
No other text, no markdown formatting."""

    try:
        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            max_tokens=60,
            temperature=0,
            response_format={"type": "json_object"}
        )
        raw = result.choices[0].message.content.strip()
        # Strip accidental markdown fences if the model adds them
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("value")
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

# Shared persona that makes the AI sound like a real, friendly human
PERSONA = """You are Lumina, a warm and friendly health clinic receptionist. 
You talk like a real person — casual, kind, and helpful. Think of how a friendly 
receptionist at a neighborhood clinic would talk to a patient they see regularly.
Rules:
- Use casual, natural language. Say things like "Sure thing!", "Got it!", "No worries!", "Awesome!"
- NEVER sound robotic or scripted. No corporate jargon.
- Keep responses SHORT — 1 to 2 sentences max. People hate long monologues on the phone.
- If the caller says hi or makes small talk, engage with it naturally! Chat like a friend.
- If they ask a random question (weather, sports, jokes), answer it genuinely and briefly.
- If they say "thanks", "bye", "no thanks", or similar, respond warmly and say goodbye.
- Do NOT use markdown, bullet points, or any formatting. Just speak naturally.
- Do NOT repeat yourself word-for-word. Rephrase if you need to re-ask.
- You are a REAL person having a REAL conversation. Act like it.
"""

# Conversation history for multi-turn context
_conversation_history: list[dict] = []

def add_to_history(role: str, content: str):
    """Track conversation so the LLM has context of what was already said."""
    _conversation_history.append({"role": role, "content": content})
    # Keep only the last 10 turns to avoid token overflow
    if len(_conversation_history) > 20:
        _conversation_history.pop(0)
        _conversation_history.pop(0)

def reset_history():
    _conversation_history.clear()

def get_conversational_reply(user_text: str, current_prompt: str, allow_freeform: bool = False) -> str:
    if allow_freeform:
        system = f"""{PERSONA}
The caller is chatting with you. There's no urgent question you need to ask right now.
Just have a natural conversation. If they seem to want to book something or need help, 
offer to assist. If they say bye or thanks, say a warm goodbye."""
    else:
        system = f"""{PERSONA}
The caller just said something that didn't directly answer your question.
Your current goal is to ask them: "{current_prompt}"
First, respond naturally to what they said — acknowledge it, react to it, be human about it.
Then smoothly bring the conversation back to your question. Don't be pushy about it."""

    messages = [{"role": "system", "content": system}]
    messages.extend(_conversation_history[-10:])  # recent context
    messages.append({"role": "user", "content": user_text})

    try:
        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=100,
            temperature=0.8,
        )
        reply = result.choices[0].message.content.strip()
        # Strip any quotes the model might wrap the response in
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]
        return reply
    except Exception as e:
        print(f"Conversation error: {e}")
        return current_prompt

def extract_confirmation(user_text: str) -> str | None:
    system = """A clinic receptionist just read out an appointment summary and asked the caller to confirm.
Based on the caller's response, determine if they mean YES (confirm) or NO (reject/change).
Examples of YES: "yes", "yeah", "yep", "sure", "sounds good", "perfect", "that's right", "go ahead", "book it"
Examples of NO: "no", "nope", "wrong", "cancel", "change", "start over", "wait", "hold on", "not right"
If it's clearly unrelated or ambiguous (like "what time does the clinic close?"), return null.
Respond with ONLY a JSON object: {"value": "yes"} or {"value": "no"} or {"value": null}."""

    try:
        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            max_tokens=20,
            temperature=0,
        )
        raw = result.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("value")
    except:
        return None
