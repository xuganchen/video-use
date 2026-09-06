"""Engine layer: cache keying and the payload contract every engine must meet.

The contract matters because render.py keeps only entries whose type == "word"
and pack_transcripts breaks phrases on the gaps between them. An engine that
returns a differently-shaped word list produces an empty render, silently.
"""

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "helpers"))

import engines  # noqa: E402
from transcribe import transcript_path  # noqa: E402


def silent_wav(path: Path, seconds: float = 2.0, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


class TranscriptPathTests(unittest.TestCase):
    """Two engines on the same source must not collide in the cache."""

    def setUp(self):
        self.edit = Path("/tmp/edit")
        self.video = Path("/src/DJI_0001.MP4")

    def test_elevenlabs_keeps_the_legacy_plain_name(self):
        # Transcripts made before --engine existed stay valid.
        for engine in ("elevenlabs", None):
            self.assertEqual(
                transcript_path(self.edit, self.video, 0, engine).name,
                "DJI_0001.json",
            )

    def test_local_engines_are_keyed_by_name(self):
        self.assertEqual(
            transcript_path(self.edit, self.video, 0, "mlx-whisper").name,
            "DJI_0001.mlx-whisper.json",
        )
        self.assertEqual(
            transcript_path(self.edit, self.video, 0, "vibevoice").name,
            "DJI_0001.vibevoice.json",
        )

    def test_engines_never_share_a_path(self):
        paths = {transcript_path(self.edit, self.video, 0, e) for e in engines.ENGINES}
        self.assertEqual(len(paths), len(engines.ENGINES))

    def test_audio_track_and_engine_compose(self):
        self.assertEqual(
            transcript_path(self.edit, self.video, 1, "mlx-whisper").name,
            "DJI_0001.track1.mlx-whisper.json",
        )


class MlxWhisperMappingTests(unittest.TestCase):
    """mlx-whisper returns nested segments; we flatten to a flat word list."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = silent_wav(Path(self.tmp.name) / "a.wav", seconds=3.0)
        # mlx_whisper is Apple-Silicon-only; stub it so the contract is testable anywhere.
        sys.modules["mlx_whisper"] = SimpleNamespace(transcribe=lambda *a, **k: {
            "language": "zh",
            "text": " 你好 世界 ",
            "segments": [
                {"start": 0.0, "end": 1.0, "words": [
                    {"word": " 你好", "start": 0.10, "end": 0.55},
                    {"word": "", "start": 0.55, "end": 0.56},      # dropped: empty
                ]},
                {"start": 1.0, "end": 2.0, "words": [
                    {"word": " 世界", "start": 1.20, "end": 1.80},
                ]},
            ],
        })
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: sys.modules.pop("mlx_whisper", None))

    def test_flattens_segments_and_drops_empty_tokens(self):
        out = engines.transcribe_audio("mlx-whisper", self.wav)
        self.assertEqual([w["text"] for w in out["words"]], ["你好", "世界"])

    def test_words_carry_the_type_render_filters_on(self):
        out = engines.transcribe_audio("mlx-whisper", self.wav)
        # render.py: `if w.get("type") != "word": continue`
        self.assertTrue(all(w["type"] == "word" for w in out["words"]))

    def test_word_level_granularity_and_duration(self):
        out = engines.transcribe_audio("mlx-whisper", self.wav)
        self.assertEqual(out["timestamp_granularity"], "word")
        self.assertEqual(out["engine"], "mlx-whisper")
        self.assertAlmostEqual(out["audio_duration_secs"], 3.0, places=3)

    def test_leading_whitespace_is_stripped_from_tokens(self):
        # Whisper emits " word" with a leading space; unstripped it corrupts
        # the phrase text pack_transcripts joins.
        out = engines.transcribe_audio("mlx-whisper", self.wav)
        self.assertFalse(any(w["text"] != w["text"].strip() for w in out["words"]))

    def test_timings_are_monotonic(self):
        out = engines.transcribe_audio("mlx-whisper", self.wav)
        for w in out["words"]:
            self.assertLessEqual(w["start"], w["end"])


class ContractTests(unittest.TestCase):
    def test_every_engine_is_dispatchable(self):
        for name in engines.ENGINES:
            self.assertIn(name, engines._DISPATCH)

    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(RuntimeError):
            engines.transcribe_audio("whisper.cpp", Path("/nope.wav"))

    def test_only_elevenlabs_needs_a_key(self):
        self.assertTrue(engines.needs_api_key("elevenlabs"))
        self.assertFalse(engines.needs_api_key("mlx-whisper"))
        self.assertFalse(engines.needs_api_key("vibevoice"))

    def test_default_engine_is_local_and_word_level(self):
        self.assertEqual(engines.DEFAULT_ENGINE, "mlx-whisper")
        self.assertFalse(engines.needs_api_key(engines.DEFAULT_ENGINE))

    def test_vibevoice_missing_package_gives_install_guidance(self):
        # transformers alone is not enough: the vibevoice package registers the
        # custom architecture. Without this guard from_pretrained dies with an
        # opaque "does not recognize this architecture" ValueError after a
        # pointless network round-trip.
        if "vibevoice" in sys.modules:
            self.skipTest("vibevoice actually installed")
        with tempfile.TemporaryDirectory() as d:
            wav = silent_wav(Path(d) / "a.wav")
            with self.assertRaises(RuntimeError) as cm:
                engines.transcribe_audio("vibevoice", wav)
            msg = str(cm.exception)
            self.assertIn("git clone", msg)
            self.assertIn("mlx-whisper", msg)

    def test_elevenlabs_without_a_key_fails_before_the_network(self):
        with tempfile.TemporaryDirectory() as d:
            wav = silent_wav(Path(d) / "a.wav")
            with self.assertRaises(RuntimeError):
                engines.transcribe_audio("elevenlabs", wav, api_key=None)


if __name__ == "__main__":
    unittest.main()
