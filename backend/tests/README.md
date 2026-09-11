# EchoPilot Voice Agent — Test Suite

## Structure

```
backend/tests/
├── corpus.json                   # 30 fixture definitions (ground truth)
├── test_session_state_machine.py # State machine unit tests (VAD, EMA, cooldown, buffer cap)
├── test_pipeline.py              # STT + NLU regression harness (WER + slot accuracy)
├── generate_synthetic.py         # Bootstraps corpus with TTS-generated audio
├── test_concurrency.py           # Concurrent WebSocket session isolation test
├── test_chaos.py                 # Failure injection (Groq down, silence, disconnect, buffer cap)
└── fixtures/
    └── audio/                    # WAV files land here (16kHz mono)
```

---

## The 4-Tier Test Architecture

| Tier | Test File | Runtime | What It Validates |
|---|---|---|---|
| **1. State Machine** | `test_session_state_machine.py` | **~30ms** | AudioSession state machine: RMS VAD boundaries (900ms), mid-sentence pause preservation, barge-in (250ms confirm), 150ms cooldown discarding TTS tail-echo, rolling EMA noise floor recalibration, 15s buffer cap truncation, peak-RMS silence gate. Runs offline with zero external dependencies. |
| **2. Fast Gate (STT + NLU)** | `test_pipeline.py` | **~30s** | Transcription accuracy (WER) and NLU slot extraction (24 fixtures). Pins prompts, relative dates, time phrases, and language tags across Groq Whisper-large-v3 and local fallback. |
| **3. Session & Chaos** | `test_chaos.py` & `test_concurrency.py` | **~1-2m** | Groq API drop recovery, fallback continuity, 7s/9s inactivity prompts, oversized audio caps, concurrent multi-user session state isolation. |
| **4. Acoustics & Hardware** | Live Pre-Flight Call Matrix | Manual | Physical Bluetooth A2DP→HFP profile switching, ambient room reverberation/noise floor shifts, physical microphone AGC. Cannot be reliably simulated synthetically. |

---

## Quickstart

### Step 1 — Run the State Machine Unit Tests (Immediate, Offline)

```powershell
cd backend
venv\Scripts\python tests\test_session_state_machine.py
```
Validates VAD timing, barge-in cooldown discarding, EMA noise recalibration, and buffer cap in 30 milliseconds.

### Step 2 — Generate synthetic bootstrap fixtures

```powershell
cd backend
venv\Scripts\python tests\generate_synthetic.py
```

This calls Edge TTS to create WAV files for all non-noisy, non-device fixtures.
**Synthetic fixtures validate STT decoder behaviour only** — not human speech variability or device codecs.

### Step 3 — Run the STT+NLU regression suite

```powershell
# Full suite against Groq (primary) + local fallback
venv\Scripts\python tests\test_pipeline.py

# Only local Whisper (useful when Groq is unavailable)
venv\Scripts\python tests\test_pipeline.py --local-only

# One fixture or category
venv\Scripts\python tests\test_pipeline.py --id domain_01
venv\Scripts\python tests\test_pipeline.py --category domain
```

### Step 4 — Run chaos tests (requires running backend)

```powershell
# Terminal 1
cd backend && venv\Scripts\uvicorn main:app --port 8000

# Terminal 2
venv\Scripts\python tests\test_chaos.py
venv\Scripts\python tests\test_chaos.py --test groq_failure   # in-process, no server needed
```

### Step 5 — Run concurrency test (requires running backend)

```powershell
venv\Scripts\python tests\test_concurrency.py --sessions 5
```

---

## Recording Real Fixtures

For each category below, record WAV files and save them to `tests/fixtures/audio/<id>.wav`.
WAV spec: **16kHz, mono, int16** (same as what the backend receives from the browser).

| Category | What to capture | Priority |
|---|---|---|
| `clean` | Normal-pace speech in a quiet room | ⭐⭐⭐ |
| `rushed` | Fast clipped speech, minimal articulation | ⭐⭐⭐ |
| `paused` | Include 1–2s natural thinking pauses | ⭐⭐⭐ |
| `domain` | Domain terms: ophthalmology, half past ten, haan theek hai | ⭐⭐⭐ |
| `noisy` | TV/traffic/fan at 60–70 dB with speech at ~75 dB | ⭐⭐ |
| `device` | Same phrase via Bluetooth, laptop mic, wired headset | ⭐⭐ |
| `silence` | Pure silence and delayed response | ⭐ |

Convert any microphone recording to 16kHz mono WAV:
```powershell
# ffmpeg (install once: winget install Gyan.FFmpeg)
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 tests/fixtures/audio/clean_01.wav
```

---

## Reading the Report

```
ID                     Backend  WER  Slots  STT ms  NLU ms  Transcript
────────────────────────────────────────────────────────────────────────────
clean_01               groq     0.000   ✓     342     210  'I need to book a cardiology ap…'
domain_01              groq     0.100   ✗     310     180  'half past 10'
  ↳ time: expected='10:30' got=None          ← NLU failed, not STT
```

**Green WER ≤ 0.10** — acceptable transcription quality  
**Yellow WER 0.10–0.30** — minor errors, slot fill may still work  
**Red WER > 0.30** — STT is failing significantly for this category  

If `Slots ✗` but transcript looks correct → **NLU parsing bug**, not audio  
If `Slots ✗` and transcript is wrong → **STT bug** upstream  

---

## Adding New Fixtures

1. Add an entry to `corpus.json`
2. Record (or synthesize) the WAV file
3. Re-run `test_pipeline.py` — new fixture is automatically picked up

---

## Interpreting STT Backend Comparison

If `groq` and `local-fallback` WER differs significantly for the same category,
it means accuracy meaningfully degrades when Groq is unavailable.
Consider upgrading the local model from `small` → `medium` if fallback accuracy matters.
