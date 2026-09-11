#!/usr/bin/env python3
"""
Session state machine & audio layer unit tests.

Exercises the core AudioSession logic and VAD state machine in isolation,
without requiring live WebSockets, external TTS, or network calls:
  1. Adaptive noise-floor calibration & dynamic threshold clamping
  2. Utterance boundary detection at 900ms silence threshold
  3. Mid-sentence pause resilience (<900ms does not split utterances)
  4. Barge-in detection (250ms confirmation) & 150ms tail-echo cooldown discard
  5. Rolling EMA noise-floor recalibration during confirmed silence
  6. 15-second buffer cap truncation (drops head, keeps tail)
  7. Peak-RMS silence gate (prevents false-positive drops on trailing silence)

Usage:
    python tests/test_session_state_machine.py
"""

import io
import os
import site
import sys
import unittest
import numpy as np
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TESTS_DIR = Path(__file__).parent
BACKEND_DIR = TESTS_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# Windows DLL setup for faster_whisper / ctranslate2 if present
if os.name == "nt":
    for site_dir in site.getsitepackages():
        for lib in ["cublas", "cudnn"]:
            bin_path = os.path.join(site_dir, "nvidia", lib, "bin")
            if os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(bin_path)

from main import (
    AudioSession,
    has_speech,
    get_peak_rms,
    SAMPLE_RATE,
    FRAME_MS,
    FRAME_SIZE,
    SILENCE_MS_TO_FINALIZE,
    BARGE_IN_CONFIRM_MS,
)


# ── Audio generation helpers ──────────────────────────────────────────────────

def make_silence(duration_s: float) -> bytes:
    """Zero PCM samples for duration_s."""
    n_samples = int(SAMPLE_RATE * duration_s)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def make_tone(duration_s: float, freq_hz: float = 300.0, rms: float = 800.0) -> bytes:
    """Sine wave with precise RMS amplitude."""
    n_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    amplitude = rms * np.sqrt(2)
    wave = np.sin(2 * np.pi * freq_hz * t) * amplitude
    return np.clip(wave, -32768, 32767).astype(np.int16).tobytes()


def make_noise(duration_s: float, rms: float = 150.0, seed: int = 42) -> bytes:
    """Gaussian noise with target RMS amplitude."""
    n_samples = int(SAMPLE_RATE * duration_s)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, rms, n_samples)
    return np.clip(noise, -32768, 32767).astype(np.int16).tobytes()


# ── State Machine Step Simulator ──────────────────────────────────────────────

class MockWebSocketSession:
    """Simulates the main.py audio_socket receive-and-process loop with controllable time."""

    def __init__(self):
        self.session = AudioSession()
        self.sim_time = 1000.0
        self.finalized_utterances = []
        self.barge_in_interrupts = 0
        self.skipped_silence_count = 0

    def feed_chunk(self, chunk: bytes, duration_s: float = 0.15):
        """Feeds an audio chunk, simulating one 150ms websocket receive cycle in main.py."""
        # Check cooldown discard (mirrors main.py: time.time() >= session.interrupt_cooldown_until)
        if self.sim_time >= self.session.interrupt_cooldown_until:
            self.session.add_chunk(chunk)

        now = self.sim_time
        pcm = self.session.get_full_pcm()
        if len(pcm) == 0:
            self.sim_time += duration_s
            return

        # 1. Barge-in check (while assistant is speaking)
        if self.session.assistant_speaking:
            barge_threshold = max(self.session.speech_threshold, 200.0)
            speaking_detected = has_speech(pcm, window_ms=150, threshold=barge_threshold)
            if speaking_detected:
                self.session.barge_in_speech_ms += 150
                if self.session.barge_in_speech_ms >= BARGE_IN_CONFIRM_MS and not self.session.interrupted:
                    self.session.interrupted = True
                    self.session.assistant_speaking = False
                    self.session.interrupt_cooldown_until = now + 0.15
                    # Mirror production: keep confirmed speech, discard pre-barge audio
                    self.session.reset_for_interrupt(self.session.barge_in_speech_ms)
                    self.session.silence_ms = 0
                    self.session.last_frame_had_speech = True
                    self.barge_in_interrupts += 1
            else:
                self.session.barge_in_speech_ms = 0
            self.sim_time += duration_s
            return

        # 2. VAD check (normal user turn)
        speaking_now = has_speech(pcm, window_ms=300, threshold=self.session.speech_threshold)
        if speaking_now:
            self.session.silence_ms = 0
            self.session.last_frame_had_speech = True
        else:
            self.session.silence_ms += 150

            # Rolling recalibration during confirmed silence (>10s)
            if (self.session.silence_ms >= 1000
                    and len(pcm) >= FRAME_SIZE
                    and now - self.session.last_recalibration_time > 10.0):
                tail = pcm[-FRAME_SIZE * int(300 / FRAME_MS):]
                ambient_rms = float(np.sqrt(np.mean(np.square(tail.astype(np.float32)))))
                old_floor = self.session.noise_floor_rms
                self.session.noise_floor_rms = 0.7 * old_floor + 0.3 * ambient_rms
                self.session.last_recalibration_time = now

        # 3. Utterance boundary check
        if self.session.last_frame_had_speech and self.session.silence_ms >= SILENCE_MS_TO_FINALIZE:
            final_pcm = self.session.get_full_pcm()
            self.session.reset_after_transcript()
            self.session.silence_ms = 0
            self.session.last_frame_had_speech = False

            if len(final_pcm) >= 1600:
                peak_rms = get_peak_rms(final_pcm, frame_ms=30)
                silence_threshold = max(self.session.noise_floor_rms * 1.5, 120.0)
                if peak_rms < silence_threshold:
                    self.skipped_silence_count += 1
                else:
                    self.finalized_utterances.append({
                        "pcm": final_pcm,
                        "duration_s": len(final_pcm) / SAMPLE_RATE,
                        "finalized_at": now,
                        "peak_rms": peak_rms,
                    })

        self.sim_time += duration_s


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestSessionStateMachine(unittest.TestCase):

    def test_01_noise_floor_calibration_and_dynamic_threshold(self):
        """First 500ms should calibrate noise floor and derive dynamic 3x speech threshold."""
        session = AudioSession()
        self.assertFalse(session.threshold_calibrated)
        self.assertEqual(session.noise_floor_rms, 150.0)

        # Feed 300ms (9600 bytes) of quiet noise (RMS=80) — not yet calibrated
        session.add_chunk(make_noise(0.3, rms=80.0, seed=1))
        self.assertFalse(session.threshold_calibrated)

        # Feed another 200ms (6400 bytes) — total 500ms (16000 bytes) triggers calibration
        session.add_chunk(make_noise(0.2, rms=80.0, seed=2))
        self.assertTrue(session.threshold_calibrated)
        self.assertAlmostEqual(session.noise_floor_rms, 80.0, delta=10.0)
        # Dynamic threshold: max(150.0, min(600.0, 80 * 3 = 240)) -> ~240
        self.assertAlmostEqual(session.speech_threshold, session.noise_floor_rms * 3.0, delta=1.0)
        self.assertGreaterEqual(session.speech_threshold, 150.0)
        self.assertLessEqual(session.speech_threshold, 600.0)

    def test_02_dynamic_threshold_clamping(self):
        """Speech threshold clamps between 150.0 (quiet room) and 600.0 (loud room)."""
        session_quiet = AudioSession()
        session_quiet.noise_floor_rms = 30.0  # 30 * 3 = 90 -> clamped to 150
        self.assertEqual(session_quiet.speech_threshold, 150.0)

        session_loud = AudioSession()
        session_loud.noise_floor_rms = 300.0  # 300 * 3 = 900 -> clamped to 600
        self.assertEqual(session_loud.speech_threshold, 600.0)

    def test_03_utterance_boundary_at_900ms_silence(self):
        """Speech followed by 900ms silence triggers exactly one utterance boundary."""
        sim = MockWebSocketSession()
        # Calibrate with 500ms ambient noise
        sim.feed_chunk(make_noise(0.5, rms=60.0))

        # 0.9s initial silence — no speech occurred yet, so no utterance boundary
        for _ in range(6):
            sim.feed_chunk(make_silence(0.15))
        self.assertEqual(len(sim.finalized_utterances), 0)

        # 1.05s speech (tone at 800 RMS > speech_threshold)
        for _ in range(7):
            sim.feed_chunk(make_tone(0.15, freq_hz=300, rms=800.0))
        self.assertTrue(sim.session.last_frame_had_speech)
        self.assertEqual(sim.session.silence_ms, 0)
        self.assertEqual(len(sim.finalized_utterances), 0)

        # 600ms silence (4 chunks):
        # Due to has_speech's 300ms window, the first 150ms chunk still contains 150ms
        # of speech in its trailing window (RMS > threshold), so silence_ms begins
        # accumulating on the second chunk (silence_ms = (4 - 1) * 150 = 450ms).
        for _ in range(4):
            sim.feed_chunk(make_silence(0.15))
        self.assertEqual(len(sim.finalized_utterances), 0)
        self.assertEqual(sim.session.silence_ms, 450)

        # Feed 3 more silence chunks (450ms) -> silence_ms reaches 900ms -> FINALIZE!
        for _ in range(3):
            sim.feed_chunk(make_silence(0.15))
        self.assertEqual(len(sim.finalized_utterances), 1)

        utt = sim.finalized_utterances[0]
        # Utterance contains initial silence + speech + trailing silence
        self.assertGreaterEqual(utt["duration_s"], 1.5)
        self.assertFalse(sim.session.last_frame_had_speech)
        self.assertEqual(sim.session.silence_ms, 0)

        # Additional silence does NOT trigger another utterance
        for _ in range(6):
            sim.feed_chunk(make_silence(0.15))
        self.assertEqual(len(sim.finalized_utterances), 1)

    def test_04_mid_sentence_pause_does_not_split(self):
        """Natural pauses < 900ms (e.g. 600ms 'umm') do NOT split an utterance into two."""
        sim = MockWebSocketSession()
        sim.feed_chunk(make_noise(0.5, rms=60.0))

        # Part 1: speech for 600ms (4 chunks)
        for _ in range(4):
            sim.feed_chunk(make_tone(0.15, rms=800.0))

        # Thinking pause: silence for 600ms (4 chunks)
        # silence_ms reaches 450ms < 900ms threshold
        for _ in range(4):
            sim.feed_chunk(make_silence(0.15))
        self.assertEqual(len(sim.finalized_utterances), 0)
        self.assertEqual(sim.session.silence_ms, 450)

        # Part 2: speech resumes for 600ms (4 chunks) -> silence_ms resets to 0
        for _ in range(4):
            sim.feed_chunk(make_tone(0.15, rms=800.0))
        self.assertEqual(sim.session.silence_ms, 0)
        self.assertTrue(sim.session.last_frame_had_speech)

        # End of turn: silence until finalization (7 chunks)
        for _ in range(7):
            sim.feed_chunk(make_silence(0.15))

        # Must be exactly ONE single combined utterance, NOT two split halves
        self.assertEqual(len(sim.finalized_utterances), 1)
        combined_utt = sim.finalized_utterances[0]
        # Total duration spans: 0.5s cal + 0.6s speech1 + 0.6s pause + 0.6s speech2 + trailing silence
        self.assertGreaterEqual(combined_utt["duration_s"], 2.7)

    def test_05_barge_in_and_cooldown_discards_tts_tail(self):
        """Barge-in confirms after 250ms; cooldown discards ONLY subsequent TTS tail, not user speech."""
        sim = MockWebSocketSession()
        sim.feed_chunk(make_noise(0.5, rms=60.0))

        # Assistant starts speaking TTS
        sim.session.assistant_speaking = True
        sim.session.barge_in_speech_ms = 0
        sim.session.interrupted = False

        # User speaks chunk 1 (150ms) -> 150ms < 250ms confirmation threshold
        sim.feed_chunk(make_tone(0.15, rms=900.0))
        self.assertFalse(sim.session.interrupted)
        self.assertTrue(sim.session.assistant_speaking)
        self.assertEqual(sim.barge_in_interrupts, 0)

        # User speaks chunk 2 (150ms) -> 300ms >= 250ms -> INTERRUPT FIRES!
        sim.feed_chunk(make_tone(0.15, rms=900.0))
        self.assertTrue(sim.session.interrupted)
        self.assertFalse(sim.session.assistant_speaking)
        self.assertEqual(sim.barge_in_interrupts, 1)

        # Cooldown window is set to now + 0.15s
        cooldown_until = sim.session.interrupt_cooldown_until
        self.assertAlmostEqual(cooldown_until, sim.sim_time - 0.15 + 0.15, delta=0.01)

        # Buffer was cleared of all pre-barge audio, but KEEPS the confirmed speech frames.
        # reset_for_interrupt() preserves the last barge_in_speech_ms of the user's opening words.
        # (300ms of speech at 16kHz int16 = 9600 bytes)
        self.assertGreater(len(sim.session.pcm_buffer), 0,
                           "Confirmed speech must be kept, not blanked")
        self.assertLessEqual(len(sim.session.pcm_buffer),
                             int(16000 * (BARGE_IN_CONFIRM_MS / 1000.0 + 0.15)) * 2 + 100,
                             "Buffer should only contain the confirmed speech window")

        # TTS tail echo arrives while still inside cooldown
        # Simulate feeding a chunk while sim_time < cooldown_until
        buffer_size_before_echo = len(sim.session.pcm_buffer)
        sim.sim_time = cooldown_until - 0.05
        sim.feed_chunk(make_tone(0.05, rms=600.0), duration_s=0.05)
        # Cooldown must have discarded the echo — buffer must not grow!
        self.assertEqual(len(sim.session.pcm_buffer), buffer_size_before_echo,
                         "TTS tail echo must be discarded by cooldown (buffer must not grow)")

        # Advance time past cooldown window
        sim.sim_time = cooldown_until + 0.01
        # Real post-interrupt user speech arrives
        sim.feed_chunk(make_tone(0.15, rms=900.0), duration_s=0.15)
        # Buffer now captures the fresh utterance!
        self.assertGreater(len(sim.session.pcm_buffer), 0)

    def test_09_barge_in_confirmation_window_audio_is_preserved(self):
        """Confirmed 250ms of user speech is kept in buffer at interrupt time, not discarded.

        The 250ms confirmation window IS the user's opening words.
        Blanking the buffer on interrupt would clip the first word of every interruption.
        reset_for_interrupt() must retain exactly those confirmed frames.
        """
        sim = MockWebSocketSession()
        sim.feed_chunk(make_noise(0.5, rms=60.0))
        sim.session.noise_floor_rms = 60.0
        sim.session.threshold_calibrated = True

        # Assistant speaking TTS
        sim.session.assistant_speaking = True
        sim.session.barge_in_speech_ms = 0
        sim.session.interrupted = False

        # User's first 150ms word — added to buffer BEFORE barge-in logic runs,
        # but confirmation not yet met (150 < 250)
        chunk1 = make_tone(0.15, rms=900.0)
        sim.feed_chunk(chunk1)
        bytes_after_chunk1 = len(sim.session.pcm_buffer)
        self.assertGreater(bytes_after_chunk1, 0, "Chunk 1 must be buffered before confirmation")

        # User's second 150ms word — confirmation fires (300 >= 250ms)
        # Buffer should NOT be blanked — chunk1 + chunk2 should be retained
        chunk2 = make_tone(0.15, rms=900.0)
        sim.feed_chunk(chunk2)

        self.assertTrue(sim.session.interrupted)
        self.assertEqual(sim.barge_in_interrupts, 1)

        # CRITICAL: confirmation window audio must survive the interrupt.
        # barge_in_speech_ms was 300ms when interrupt fired, so 300ms of tail should be kept.
        expected_keep_bytes = int(16000 * (300 / 1000.0)) * 2  # 300ms at 16kHz int16
        # Buffer must have content (first words preserved), not be empty
        self.assertGreater(
            len(sim.session.pcm_buffer), 0,
            "Confirmation-window audio (user's opening words) must not be blanked on interrupt"
        )
        # Buffer should be at most expected_keep_bytes (the tail of confirmed speech)
        self.assertLessEqual(
            len(sim.session.pcm_buffer), expected_keep_bytes + 100,  # small tolerance for rounding
            "Buffer should contain at most the confirmed speech window, not all prior buffered audio"
        )

    def test_06_rolling_ema_noise_floor_recalibration(self):
        """EMA recalibration shifts noise_floor_rms during confirmed silence (>10s)."""
        sim = MockWebSocketSession()
        # Calibrate initial noise floor at 100.0 RMS
        sim.feed_chunk(make_noise(0.5, rms=100.0))
        sim.session.noise_floor_rms = 100.0
        sim.session.last_recalibration_time = sim.sim_time

        # Simulate AC unit turning on, background noise level rising to 200.0 RMS
        # Feed 11 seconds of high-ambient silence (>10s recalibration interval)
        for _ in range(74):  # 74 * 0.15s = 11.1s
            sim.feed_chunk(make_noise(0.15, rms=200.0))

        # Expected EMA: 0.7 * old (100) + 0.3 * new (~200) = ~130
        self.assertGreater(sim.session.noise_floor_rms, 115.0)
        self.assertLess(sim.session.noise_floor_rms, 155.0)
        # Speech threshold must update accordingly
        self.assertAlmostEqual(sim.session.speech_threshold, sim.session.noise_floor_rms * 3.0, delta=1.0)

    def test_07_buffer_cap_15s_drops_head(self):
        """Buffer cap truncates at 480,000 bytes (15s), preserving the recent tail."""
        session = AudioSession()
        # 16 seconds of audio = 16 * 16000 * 2 = 512,000 bytes
        large_chunk = b"\x01\x02" * (16 * 16000)
        session.add_chunk(large_chunk)

        # Cap must be exactly 480,000 bytes
        self.assertEqual(len(session.pcm_buffer), 480000)

    def test_08_peak_rms_silence_gate_math(self):
        """Peak-RMS gate accepts quiet speech with trailing silence, but rejects pure silence."""
        session = AudioSession()
        session.noise_floor_rms = 100.0
        silence_threshold = max(session.noise_floor_rms * 1.5, 120.0)  # 150.0

        # Case A: Pure silence (1.5s of zeros)
        pure_silence = np.frombuffer(make_silence(1.5), dtype=np.int16)
        peak_silence = get_peak_rms(pure_silence, frame_ms=30)
        self.assertEqual(peak_silence, 0.0)
        self.assertLess(peak_silence, silence_threshold)

        # Case B: 200ms quiet speech (RMS=300) + 1000ms trailing silence (RMS=50)
        speech_pcm = np.frombuffer(make_tone(0.2, freq_hz=300, rms=300.0), dtype=np.int16)
        tail_silence = np.frombuffer(make_noise(1.0, rms=50.0), dtype=np.int16)
        mixed_pcm = np.concatenate([speech_pcm, tail_silence])

        # Buffer-wide average RMS: ~124 (would falsely drop if buffer-average were used!)
        buffer_avg_rms = float(np.sqrt(np.mean(np.square(mixed_pcm.astype(np.float32)))))
        self.assertLess(buffer_avg_rms, silence_threshold,
                        "Buffer average should be below threshold (demonstrating the danger)")

        # Peak frame RMS: ~300 (safely passes threshold!)
        peak_mixed = get_peak_rms(mixed_pcm, frame_ms=30)
        self.assertGreaterEqual(peak_mixed, silence_threshold,
                               "Peak frame RMS must clear threshold, preserving quiet speech")


# ── Standalone CLI Runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    print(f"\n{BOLD}══════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}  EchoPilot AudioSession State Machine Test Suite{RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════{RESET}\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSessionStateMachine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print(f"\n{BOLD}{GREEN}✓ All {result.testsRun} state machine tests passed successfully!{RESET}\n")
        sys.exit(0)
    else:
        sys.exit(1)
