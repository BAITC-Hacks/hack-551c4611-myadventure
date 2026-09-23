import contextlib
import io
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from speech import PipelineDiagnostics, PipelineError, processMeetingAudio
from speech.alignment import SpeechPipelineResult
from speech.fixture_suite import CASES, FIXTURES, build_derived_fixtures, run_fixture_suite


class FixtureTests(unittest.TestCase):
    def test_five_small_fixtures_present(self) -> None:
        self.assertEqual(len(CASES), 5)
        self.assertLess(sum((FIXTURES / case.filename).stat().st_size for case in CASES), 3 * 1024 * 1024)
        for case in CASES:
            if not case.invalid:
                with wave.open(str(FIXTURES / case.filename), "rb") as audio:
                    self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
                    self.assertGreater(audio.getnframes(), 0)

    def test_derived_fixtures_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ("ru.wav", "kk.wav"):
                shutil.copyfile(FIXTURES / name, directory / name)
            build_derived_fixtures(directory)
            for name in ("mixed.wav", "multi-speaker.wav", "invalid.wav"):
                self.assertEqual((directory / name).read_bytes(), (FIXTURES / name).read_bytes())

    def test_invalid_fixture_is_controlled_before_model_access(self) -> None:
        with patch("speech.pipeline.transcribeAudio") as stt:
            with self.assertRaises(PipelineError) as caught:
                processMeetingAudio(FIXTURES / "invalid.wav")
            stt.assert_not_called()
        self.assertEqual((caught.exception.stage, caught.exception.code), ("preprocessing", "UNSUPPORTED_FORMAT"))

    def test_runner_checks_results_and_marks_demo(self) -> None:
        # Test runner mechanics only, not recognition accuracy.
        def fake_pipeline(path: Path, *, mode: str, diagnostics: PipelineDiagnostics, demo_speakers: int, demo_turn_seconds: float) -> SpeechPipelineResult:
            diagnostics.mode = "DEMO"
            case = next(case for case in CASES if case.filename == path.name)
            if case.invalid:
                raise PipelineError("preprocessing", "UNSUPPORTED_FORMAT", "fixture rejected")
            return {
                "durationSeconds": 2,
                "detectedSpeakers": case.min_speakers,
                "segments": [{"id": f"seg-{index}", "speakerId": f"SPEAKER_{index:02d}", "start": float(index), "end": float(index + 1), "text": " ".join(case.keywords)} for index in range(case.min_speakers)],
            }

        with patch("speech.fixture_suite.processMeetingAudio", side_effect=fake_pipeline) as pipeline, contextlib.redirect_stdout(io.StringIO()):
            report = run_fixture_suite("demo")
        self.assertEqual(pipeline.call_count, 5)
        self.assertTrue(report["passed"])
        self.assertFalse(report["speaker_separation_ai_evaluated"])
        self.assertEqual(pipeline.call_args_list[2].kwargs["demo_turn_seconds"], 14.68)
        self.assertEqual(pipeline.call_args_list[3].kwargs["demo_turn_seconds"], 13.3)

    def test_runner_does_not_hide_failures(self) -> None:
        with patch("speech.fixture_suite.processMeetingAudio", side_effect=RuntimeError("test fault")) as pipeline, contextlib.redirect_stdout(io.StringIO()):
            report = run_fixture_suite("demo")
        self.assertEqual(pipeline.call_count, 5)
        self.assertFalse(report["passed"])
