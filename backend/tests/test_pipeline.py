#!/usr/bin/env python3
"""
Voice agent regression test harness.

Replays every fixture in corpus.json through the real STT → NLU pipeline
and reports per-fixture and per-category:
  - WER  (word error rate on raw transcript vs ground truth)
  - Slot accuracy  (did NLU extract the right value?)
  - STT backend  (groq / local / none)
  - End-to-end latency  (ms from audio load to slot extracted)

Usage:
    # Run all fixtures against Groq + local fallback (default)
    python tests/test_pipeline.py

    # Skip Groq, only test local Whisper
    python tests/test_pipeline.py --local-only

    # Run a single fixture by id
    python tests/test_pipeline.py --id clean_01

    # Run a specific category
    python tests/test_pipeline.py --category domain

    # Save detailed results to JSON
    python tests/test_pipeline.py --out results.json
"""

import argparse
import io
import json
import os
import site
import sys
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ── Force UTF-8 stdout on Windows (PowerShell defaults to cp1252) ─────────────
# Must run before any print() that uses Unicode box-drawing / tick chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── path setup ────────────────────────────────────────────────────────────────
TESTS_DIR  = Path(__file__).parent
BACKEND_DIR = TESTS_DIR.parent
AUDIO_DIR  = TESTS_DIR / "fixtures" / "audio"
CORPUS_PATH = TESTS_DIR / "corpus.json"

sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

# ── Windows CUDA DLL fix (mirrors main.py startup) ────────────────────────────
# Must run BEFORE faster_whisper is imported, otherwise cublas64_12.dll
# is not found and WhisperModel() raises RuntimeError.
if os.name == "nt":
    for site_dir in site.getsitepackages():
        for lib in ["cublas", "cudnn"]:
            bin_path = os.path.join(site_dir, "nvidia", lib, "bin")
            if os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(bin_path)


# ── imports from backend ──────────────────────────────────────────────────────
from extraction import extract_field                     # NLU slot extractor
from extraction import extract_confirmation              # for confirmation fixtures

try:
    from openai import OpenAI
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    ) if GROQ_API_KEY else None
except Exception:
    groq_client = None

try:
    from faster_whisper import WhisperModel
    _local_model = None
    def get_local_model():
        global _local_model
        if _local_model is None:
            try:
                _local_model = WhisperModel("small", device="cuda", compute_type="float16")
            except Exception:
                try:
                    _local_model = WhisperModel("small", device="cpu", compute_type="int8")
                except Exception:
                    pass
        return _local_model
except ImportError:
    def get_local_model():
        return None

# ── constants ─────────────────────────────────────────────────────────────────
WHISPER_PROMPT = (
    "Healthcare voice conversation in English and Telugu at Meridian Clinic. "
    "Appointment with doctor, general physician, doctor consultation, checkup, "
    "head surgery, brain surgery, cardiology, dermatology, orthopedic, "
    "pediatric, ophthalmology, dentist, blood test, X-ray, "
    "tomorrow, Monday, Tuesday, Wednesday, morning, afternoon, evening, "
    "10 AM, 11 AM, 2 PM, haan, nahi, theek hai, kal, parso, "
    "Abhilash, confirm, cancel, change."
)

# ── audio loading ─────────────────────────────────────────────────────────────
def load_wav_16k_mono(path: Path) -> np.ndarray:
    """Load any WAV file, mix to mono, resample to 16 kHz, return int16 PCM."""
    with wave.open(str(path), "rb") as wf:
        n_ch    = wf.getnchannels()
        sw      = wf.getsampwidth()
        rate    = wf.getframerate()
        raw     = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    pcm = np.frombuffer(raw, dtype=dtype)

    # Promote to int16 range
    if sw == 1:
        pcm = (pcm.astype(np.int16) - 128) * 256
    elif sw == 4:
        pcm = (pcm.astype(np.int32) >> 16).astype(np.int16)

    # Stereo → mono
    if n_ch == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif n_ch > 2:
        pcm = pcm.reshape(-1, n_ch)[:, 0]

    # Resample to 16 kHz via linear interpolation (good enough for test validation)
    if rate != 16000:
        new_len = int(len(pcm) * 16000 / rate)
        indices = np.linspace(0, len(pcm) - 1, new_len)
        pcm = np.interp(indices, np.arange(len(pcm)), pcm.astype(np.float32)).astype(np.int16)

    return pcm


def get_peak_rms(pcm: np.ndarray, frame_ms: int = 30) -> float:
    """Computes peak RMS across frame_ms windows of PCM audio.
    
    Using peak frame RMS instead of whole-buffer average prevents silently
    dropping utterances that contain genuine speech followed by trailing silence.
    """
    if len(pcm) == 0:
        return 0.0
    frame_samples = int(16000 * frame_ms / 1000)
    if len(pcm) < frame_samples:
        return float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))
    num_frames = len(pcm) // frame_samples
    frames = pcm[:num_frames * frame_samples].reshape(num_frames, frame_samples).astype(np.float32)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.max(frame_rms))


# ── WER (word-level Levenshtein, punctuation-normalized) ─────────────────────
import re as _re

def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace — then split into words.

    Whisper systematically adds trailing periods and commas that the ground-truth
    transcripts don't have.  Without normalizing, "Yes, that sounds good." vs
    "Yes that sounds good" scores WER=0.5 even though the content is identical.
    """
    text = text.lower()
    text = _re.sub(r"[^\w\s]", " ", text)   # replace punctuation with space
    text = _re.sub(r"\s+", " ", text).strip()
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return normalized WER in [0, ∞). 0 = perfect match after normalization."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else float("inf")

    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp

    return dp[n] / len(ref)


# ── STT helpers ───────────────────────────────────────────────────────────────
def transcribe_groq(pcm: np.ndarray) -> tuple[str, float]:
    """Returns (transcript, latency_ms)."""
    if not groq_client:
        return "", 0.0
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    buf.name = "audio.wav"

    t0 = time.time()
    result = groq_client.audio.transcriptions.create(
        file=buf,
        model="whisper-large-v3",
        language="en",
        temperature=0,
        prompt=WHISPER_PROMPT,
    )
    return result.text.strip(), (time.time() - t0) * 1000


def transcribe_local(pcm: np.ndarray) -> tuple[str, float]:
    """Returns (transcript, latency_ms)."""
    model = get_local_model()
    if not model:
        return "", 0.0
    pcm_f = pcm.astype(np.float32) / 32768.0
    t0 = time.time()
    segs, _ = model.transcribe(
        pcm_f,
        language="en",
        temperature=0,
        vad_filter=False,
        beam_size=5,
        initial_prompt=WHISPER_PROMPT,
    )
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, (time.time() - t0) * 1000


# ── NLU helpers ───────────────────────────────────────────────────────────────
def extract_slots(transcript: str, expected_slots: dict) -> tuple[dict, float]:
    """Run extract_field / extract_confirmation for each expected slot.

    Returns ({field: {expected, extracted, correct}}, total_nlu_latency_ms).
    """
    results = {}
    total_ms = 0.0

    for field, expected in expected_slots.items():
        t0 = time.time()
        if field == "confirmation":
            extracted = extract_confirmation(transcript)
        else:
            extracted = extract_field(field, transcript) if transcript else None
        total_ms += (time.time() - t0) * 1000

        extracted_norm = (extracted or "").lower().strip()
        expected_norm  = (expected  or "").lower().strip()
        results[field] = {
            "expected":  expected,
            "extracted": extracted,
            "correct":   extracted_norm == expected_norm,
        }

    return results, total_ms


# ── per-fixture runner ────────────────────────────────────────────────────────
def run_fixture(fixture: dict, local_only: bool = False) -> dict:
    audio_path = AUDIO_DIR / fixture["file"]

    base = {
        "id":       fixture["id"],
        "category": fixture.get("category", "unknown"),
        "file":     str(audio_path),
    }

    if not audio_path.exists():
        return {**base, "status": "MISSING"}

    try:
        pcm = load_wav_16k_mono(audio_path)
    except Exception as e:
        return {**base, "status": "LOAD_ERROR", "error": str(e)}

    # ── STT ──────────────────────────────────────────────────────────────────
    transcript, stt_ms, backend = "", 0.0, "none"

    # Peak-RMS silence gate — matches production pipeline behavior in main.py to prevent
    # Whisper hallucinating stock phrases ("Thank you for watching") on silent audio buffers.
    peak_rms = get_peak_rms(pcm, frame_ms=30)
    if peak_rms < 120.0:
        transcript, stt_ms, backend = "", 0.0, "silence-gate"
    elif not local_only and groq_client:
        transcript, stt_ms = transcribe_groq(pcm)
        backend = "groq"
        if not transcript:
            transcript, stt_ms = transcribe_local(pcm)
            backend = "local-fallback"
    else:
        transcript, stt_ms = transcribe_local(pcm)
        backend = "local"

    expected_tx = fixture.get("expected_transcript", "")
    wer = word_error_rate(expected_tx, transcript)

    # ── NLU ──────────────────────────────────────────────────────────────────
    slot_results, nlu_ms = extract_slots(transcript, fixture.get("expected_slots", {}))
    slots_all_correct = all(r["correct"] for r in slot_results.values()) if slot_results else True

    return {
        **base,
        "status":              "OK",
        "stt_backend":         backend,
        "expected_transcript": expected_tx,
        "actual_transcript":   transcript,
        "wer":                 round(wer, 3),
        "slots":               slot_results,
        "slots_correct":       slots_all_correct,
        "stt_latency_ms":      round(stt_ms),
        "nlu_latency_ms":      round(nlu_ms),
        "total_latency_ms":    round(stt_ms + nlu_ms),
    }


# ── report printer ────────────────────────────────────────────────────────────
def print_report(results: list[dict]):
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    YELLOW= "\033[93m"
    RESET = "\033[0m"

    ok = [r for r in results if r["status"] == "OK"]
    missing = [r for r in results if r["status"] == "MISSING"]
    errors  = [r for r in results if r["status"] not in ("OK", "MISSING")]

    print(f"\n{BOLD}══════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}  EchoPilot Voice Agent — Regression Test Report{RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════{RESET}\n")

    if missing:
        print(f"{YELLOW}⚠  {len(missing)} fixture(s) MISSING (not recorded yet):{RESET}")
        for r in missing:
            print(f"     - {r['id']}  ({r['file']})")
        print()

    if errors:
        print(f"{RED}✗  {len(errors)} fixture(s) errored:{RESET}")
        for r in errors:
            print(f"     - {r['id']}: {r.get('error','unknown error')}")
        print()

    if not ok:
        print("No runnable fixtures found. Record audio files and re-run.\n")
        return

    # ── per-fixture table ─────────────────────────────────────────────────────
    print(f"{'ID':<22} {'Backend':<8} {'WER':>5} {'Slots':>6} {'STT ms':>7} {'NLU ms':>7}  Transcript")
    print("─" * 100)
    for r in ok:
        wer_color = GREEN if r["wer"] <= 0.1 else (YELLOW if r["wer"] <= 0.3 else RED)
        slot_icon = f"{GREEN}✓{RESET}" if r["slots_correct"] else f"{RED}✗{RESET}"
        tx_preview = r["actual_transcript"][:50] + ("…" if len(r["actual_transcript"]) > 50 else "")
        print(
            f"{r['id']:<22} {r['stt_backend']:<8} "
            f"{wer_color}{r['wer']:>5.3f}{RESET} "
            f"  {slot_icon}  "
            f"{r['stt_latency_ms']:>6}  {r['nlu_latency_ms']:>6}  "
            f"'{tx_preview}'"
        )
        # Print slot details for failures
        for field, slot in r["slots"].items():
            if not slot["correct"]:
                print(
                    f"  {'':>22} {RED}↳ {field}: expected={slot['expected']!r} "
                    f"got={slot['extracted']!r}{RESET}"
                )

    print()

    # ── per-category summary ──────────────────────────────────────────────────
    categories = {}
    for r in ok:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)

    print(f"\n{BOLD}Category Summary:{RESET}")
    print(f"{'Category':<12} {'Fixtures':>8} {'Avg WER':>8} {'Slot Acc':>9} {'Avg ms':>8}")
    print("\u2500" * 55)
    for cat, items in sorted(categories.items()):
        finite_wers = [i["wer"] for i in items if i["wer"] != float("inf")]
        avg_wer  = sum(finite_wers) / len(finite_wers) if finite_wers else float("inf")
        inf_count = len(items) - len(finite_wers)
        slot_acc = sum(1 for i in items if i["slots_correct"]) / len(items)
        avg_ms   = sum(i["total_latency_ms"] for i in items) / len(items)
        wer_col  = GREEN if avg_wer <= 0.1 else (YELLOW if avg_wer <= 0.3 else RED)
        acc_col  = GREEN if slot_acc >= 0.9 else (YELLOW if slot_acc >= 0.7 else RED)
        wer_str  = f"{avg_wer:>8.3f}" if avg_wer != float("inf") else "     inf"
        inf_note = f" (+{inf_count} halluc)" if inf_count else ""
        print(
            f"{cat:<12} {len(items):>8} "
            f"{wer_col}{wer_str}{RESET}{inf_note} "
            f"{acc_col}{slot_acc:>8.0%}{RESET} "
            f"{avg_ms:>7.0f}ms"
        )

    # ── backend comparison ────────────────────────────────────────────────────
    backends = {}
    for r in ok:
        b = r["stt_backend"]
        backends.setdefault(b, []).append(r)

    if len(backends) > 1:
        print(f"\n{BOLD}STT Backend Comparison:{RESET}")
        for b, items in sorted(backends.items()):
            finite_wers = [i["wer"] for i in items if i["wer"] != float("inf")]
            avg_wer = sum(finite_wers) / len(finite_wers) if finite_wers else float("inf")
            acc = sum(1 for i in items if i["slots_correct"]) / len(items)
            avg_ms = sum(i["stt_latency_ms"] for i in items) / len(items)
            wer_str = f"{avg_wer:.3f}" if avg_wer != float("inf") else "inf"
            print(f"  {b:<16}: n={len(items):>2}  avg_wer={wer_str}  slot_acc={acc:.0%}  avg_stt={avg_ms:.0f}ms")

    # ── overall ───────────────────────────────────────────────────────────────
    total = len(ok)
    passed_slots = sum(1 for r in ok if r["slots_correct"])
    finite_wers  = [r["wer"] for r in ok if r["wer"] != float("inf")]
    halluc_count = total - len(finite_wers)  # fixtures where ref=empty but STT returned text
    avg_wer_all  = sum(finite_wers) / len(finite_wers) if finite_wers else float("inf")
    avg_ms_all   = sum(r["total_latency_ms"] for r in ok) / total

    overall_color = GREEN if passed_slots / total >= 0.9 and avg_wer_all <= 0.1 else RED
    print(f"\n{BOLD}Overall  ({total} fixtures run):{RESET}")
    wer_display = f"{avg_wer_all:.3f}" if avg_wer_all != float("inf") else "inf"
    print(f"  WER (finite) : {overall_color}{wer_display}{RESET}"
          + (f"  ({halluc_count} hallucination(s) on silence excluded)" if halluc_count else ""))
    print(f"  Slot acc     : {overall_color}{passed_slots}/{total} ({passed_slots/total:.0%}){RESET}")
    print(f"  Avg latency  : {avg_ms_all:.0f} ms\n")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EchoPilot regression test harness")
    parser.add_argument("--local-only",  action="store_true", help="Skip Groq, use local Whisper only")
    parser.add_argument("--id",          help="Run a single fixture by id")
    parser.add_argument("--category",    help="Run all fixtures of a given category")
    parser.add_argument("--out",         help="Write full results JSON to this file")
    args = parser.parse_args()

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    # Filter
    if args.id:
        corpus = [fx for fx in corpus if fx["id"] == args.id]
        if not corpus:
            print(f"No fixture with id={args.id!r}")
            sys.exit(1)
    if args.category:
        corpus = [fx for fx in corpus if fx.get("category") == args.category]

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(corpus)
    for i, fixture in enumerate(corpus, 1):
        status_char = "." if (AUDIO_DIR / fixture["file"]).exists() else "?"
        print(f"[{i:>2}/{total}] {status_char} {fixture['id']} ...", end=" ", flush=True)
        r = run_fixture(fixture, local_only=args.local_only)
        results.append(r)
        if r["status"] == "MISSING":
            print("MISSING (record and re-run)")
        elif r["status"] == "OK":
            wer_str = f"WER={r['wer']:.3f}"
            slot_str = "slots✓" if r["slots_correct"] else "slots✗"
            print(f"{wer_str}  {slot_str}  [{r['stt_backend']}]  {r['total_latency_ms']}ms")
        else:
            print(f"ERROR: {r.get('error','')}")

    print_report(results)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Full results written to: {out_path}\n")


if __name__ == "__main__":
    main()
