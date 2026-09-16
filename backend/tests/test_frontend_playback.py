"""Guard the TTS playback globals that silently mute Elena if missing."""
import unittest
from pathlib import Path

APP_JS = Path(__file__).resolve().parents[2] / "frontend" / "app.js"


class TestFrontendPlaybackGlobals(unittest.TestCase):
    def test_playback_queue_is_declared(self):
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("const playbackQueue = []", source)
        self.assertIn("let playbackActive = false", source)
        self.assertIn("let ttsStreamOpen = false", source)
        self.assertIn("function enqueuePlayback", source)
        self.assertLess(
            source.index("const playbackQueue = []"),
            source.index("function enqueuePlayback"),
            "playbackQueue must be declared before enqueuePlayback runs",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
