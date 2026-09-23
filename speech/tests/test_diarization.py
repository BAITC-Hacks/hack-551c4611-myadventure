import io
import json
import math
import os
import socket
import subprocess
import tempfile
import unittest
import wave
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speech import DiarizationError, PreparedAudio, PyannoteLocalBackend, diarizeAudio, prepareAudio
from speech._diarization_worker import run
from speech.diarization import DiarizationResult, SpeakerTurn


class StubBackend:
    """Test double only: these labels are not inferred from audio."""

    def __init__(self, turns: Iterable[SpeakerTurn]) -> None:
        self.turns = turns

    def diarize(self, audio: PreparedAudio) -> Iterable[SpeakerTurn]:
        return self.turns


class DiarizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.make_audio(16000)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("No network allowed"))
        network.start()
        self.addCleanup(network.stop)

    def make_audio(self, frames: int) -> PreparedAudio:
        path = self.root / "input.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * frames)
        return prepareAudio(path)

    def assert_valid(self, result: DiarizationResult, duration: float) -> None:
        starts = [item["start"] for item in result["segments"]]
        self.assertEqual(starts, sorted(starts))
        for item in result["segments"]:
            self.assertRegex(item["speakerId"], r"^SPEAKER_\d{2,}$")
            self.assertTrue(math.isfinite(item["start"]))
            self.assertGreaterEqual(item["start"], 0)
            self.assertGreaterEqual(item["end"], item["start"])
            self.assertLessEqual(item["end"], duration)

    def test_one_speaker_demo(self) -> None:
        audio = prepareAudio(Path(__file__).parent / "fixtures" / "ru.wav")
        result = diarizeAudio(audio, mode="demo")
        self.assertEqual(result["mode"], "DEMO")
        self.assertIn("not AI", result["warning"] or "")
        self.assertEqual({item["speakerId"] for item in result["segments"]}, {"SPEAKER_00"})
        self.assert_valid(result, audio.duration_seconds)

    def test_two_speakers_demo_is_explicit_and_deterministic(self) -> None:
        # Composite speech fixture; speaker count is supplied, not inferred.
        with wave.open(str(self.audio.path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            for name in ["ru.wav", "kk.wav"]:
                with wave.open(str(Path(__file__).parent / "fixtures" / name), "rb") as source:
                    output.writeframes(source.readframes(source.getnframes()))
        audio = prepareAudio(self.audio.path)
        interval = audio.duration_seconds / 4
        result = diarizeAudio(audio, mode="demo", demo_speakers=2, demo_turn_seconds=interval)
        self.assertEqual([s["speakerId"] for s in result["segments"]], ["SPEAKER_00", "SPEAKER_01"] * 2)
        self.assertEqual(result, diarizeAudio(audio, mode="demo", demo_speakers=2, demo_turn_seconds=interval))
        self.assertEqual(result["mode"], "DEMO")
        self.assert_valid(result, audio.duration_seconds)

    def test_one_frame_audio(self) -> None:
        audio = self.make_audio(1)
        result = diarizeAudio(audio, mode="demo", demo_speakers=2)
        self.assertEqual(len(result["segments"]), 1)
        self.assert_valid(result, audio.duration_seconds)

    def test_rapid_switches(self) -> None:
        result = diarizeAudio(self.audio, mode="demo", demo_speakers=3, demo_turn_seconds=0.01)
        self.assertEqual(len(result["segments"]), 100)
        self.assertEqual(len({s["speakerId"] for s in result["segments"]}), 3)
        self.assert_valid(result, 1)

    def test_model_unavailable_no_automatic_demo(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DiarizationError) as caught:
                diarizeAudio(self.audio)
        self.assertEqual(caught.exception.code, "MODEL_MISSING")
        with self.assertRaises(DiarizationError) as caught:
            PyannoteLocalBackend(self.root / "missing")
        self.assertEqual(caught.exception.code, "MODEL_MISSING")

    def test_stable_ids_by_first_occurrence_and_overlap_preserved(self) -> None:
        backend = StubBackend([
            SpeakerTurn("alice-cluster", 0.7, 1), SpeakerTurn("bob-cluster", 0, 0.5),
            SpeakerTurn("alice-cluster", 0.4, 0.8), SpeakerTurn("bob-cluster", 0.8, 1),
        ])
        result = diarizeAudio(self.audio, backend=backend)
        self.assertEqual(result["mode"], "LOCAL")
        self.assertIsNone(result["warning"])
        self.assertEqual([s["speakerId"] for s in result["segments"]], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_00"])
        self.assertEqual(len(result["segments"]), 4)
        self.assert_valid(result, 1)

    def test_single_cluster_and_empty_local_result(self) -> None:
        result = diarizeAudio(self.audio, backend=StubBackend([SpeakerTurn("cluster-z", 0, 1)]))
        self.assertEqual(result["segments"][0]["speakerId"], "SPEAKER_00")
        self.assertEqual(diarizeAudio(self.audio, backend=StubBackend([]))["segments"], [])

    def test_invalid_backend_timestamps_and_label(self) -> None:
        for turn in [SpeakerTurn("", 0, 1), SpeakerTurn(" ", 0, 1), SpeakerTurn("a", -1, 1), SpeakerTurn("a", 1, 0), SpeakerTurn("a", 0, float("nan")), SpeakerTurn("a", float("inf"), 1)]:
            with self.subTest(turn=turn), self.assertRaises(DiarizationError) as caught:
                diarizeAudio(self.audio, backend=StubBackend([turn]))
            self.assertEqual(caught.exception.code, "INFERENCE_FAILED")

    def test_out_of_bounds_turns(self) -> None:
        result = diarizeAudio(self.audio, backend=StubBackend([SpeakerTurn("a", 0, 2), SpeakerTurn("b", 2, 3)]))
        self.assertEqual(result["segments"], [{"speakerId": "SPEAKER_00", "start": 0, "end": 1}])

    def test_lazy_backend_error_is_controlled(self) -> None:
        def broken() -> Iterable[SpeakerTurn]:
            yield SpeakerTurn("a", 0, 0.5)
            raise RuntimeError("failed")

        with self.assertRaises(DiarizationError) as caught:
            diarizeAudio(self.audio, backend=StubBackend(broken()))
        self.assertEqual(caught.exception.code, "INFERENCE_FAILED")

    def test_invalid_audio_including_demo(self) -> None:
        self.audio.path.write_bytes(b"broken")
        with self.assertRaises(DiarizationError) as caught:
            diarizeAudio(self.audio, mode="demo")
        self.assertEqual(caught.exception.code, "INVALID_AUDIO")

    def test_changed_metadata(self) -> None:
        with self.assertRaises(DiarizationError) as caught:
            diarizeAudio(replace(self.audio, duration_seconds=20), mode="demo")
        self.assertEqual(caught.exception.code, "INVALID_AUDIO")

    def test_invalid_demo_configuration(self) -> None:
        for interval in [0, -1, float("nan"), float("inf"), 1e-300]:
            with self.subTest(interval=interval), self.assertRaises(DiarizationError) as caught:
                diarizeAudio(self.audio, mode="demo", demo_turn_seconds=interval)
            self.assertEqual(caught.exception.code, "INVALID_CONFIG")
        for speakers in [0, -1, True]:
            with self.subTest(speakers=speakers), self.assertRaises(DiarizationError):
                diarizeAudio(self.audio, mode="demo", demo_speakers=speakers)
        with self.assertRaises(DiarizationError):
            diarizeAudio(self.audio, demo_speakers=2)
        with self.assertRaises(DiarizationError):
            diarizeAudio(self.audio, mode="demo", backend=StubBackend([]))

    def test_input_unchanged_and_no_temp_files(self) -> None:
        contents = self.audio.path.read_bytes()
        before = set(self.root.iterdir())
        diarizeAudio(self.audio, mode="demo")
        self.assertEqual(self.audio.path.read_bytes(), contents)
        self.assertEqual(set(self.root.iterdir()), before)

    def backend(self) -> PyannoteLocalBackend:
        (self.root / "config.yaml").write_text("invalid-bundle", encoding="utf-8")
        return PyannoteLocalBackend(self.root)

    def test_worker_command_is_local_and_offline(self) -> None:
        backend = self.backend()
        original_environment = dict(os.environ)
        response = subprocess.CompletedProcess([], 0, stdout='[{"speaker":"x","start":0,"end":1}]', stderr="")
        with patch("speech.diarization.subprocess.run", return_value=response) as runner:
            result = diarizeAudio(self.audio, backend=backend)
        self.assert_valid(result, 1)
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(kwargs["env"]["PYANNOTE_METRICS_ENABLED"], "0")
        self.assertEqual(kwargs["env"]["OTEL_SDK_DISABLED"], "true")
        self.assertNotIn("shell", kwargs)
        self.assertEqual(runner.call_args.args[0][1:3], ["-m", "speech._diarization_worker"])
        self.assertEqual(dict(os.environ), original_environment)

    def test_worker_error_codes_and_bad_output(self) -> None:
        backend = self.backend()
        for code, expected in [(2, "MODEL_LOADING_FAILED"), (3, "INFERENCE_FAILED")]:
            with patch("speech.diarization.subprocess.run", return_value=subprocess.CompletedProcess([], code, stdout="", stderr="details")):
                with self.assertRaises(DiarizationError) as caught:
                    diarizeAudio(self.audio, backend=backend)
            self.assertEqual(caught.exception.code, expected)
        for data in ["not json", "{}", "[{}]", '[{"speaker":"x","start":true,"end":1}]']:
            with patch("speech.diarization.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=data, stderr="")):
                with self.assertRaises(DiarizationError) as caught:
                    diarizeAudio(self.audio, backend=backend)
            self.assertEqual(caught.exception.code, "INFERENCE_FAILED")

    def test_worker_timeout(self) -> None:
        backend = self.backend()
        with patch("speech.diarization.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 1)):
            with self.assertRaises(DiarizationError) as caught:
                diarizeAudio(self.audio, backend=backend)
        self.assertEqual(caught.exception.code, "INFERENCE_FAILED")

    def test_real_subprocess_without_usable_model(self) -> None:
        # Real process/error transport, NOT a real model quality test.
        with self.assertRaises(DiarizationError) as caught:
            diarizeAudio(self.audio, backend=self.backend())
        self.assertEqual(caught.exception.code, "MODEL_LOADING_FAILED")

    def test_worker_pyannote_adapter_and_explicit_no_token(self) -> None:
        self.backend()
        output = io.StringIO()
        with patch.dict(os.environ), patch("speech._diarization_worker.importlib.import_module") as importer:
            loader = importer.return_value.Pipeline.from_pretrained
            loader.return_value.return_value.speaker_diarization.itertracks.return_value = [
                (SimpleNamespace(start=0, end=1), None, "cluster"),
            ]
            with redirect_stdout(output):
                code = run(self.root, self.audio.path, 2)
            loader.assert_called_once_with(str(self.root), token=False)
            loader.return_value.assert_called_once_with(str(self.audio.path), num_speakers=2)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["PYANNOTE_METRICS_ENABLED"], "0")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), [{"speaker": "cluster", "start": 0, "end": 1}])

    def test_worker_load_and_inference_failures(self) -> None:
        self.backend()
        with patch.dict(os.environ), patch("speech._diarization_worker.importlib.import_module", side_effect=ImportError("not installed")), redirect_stderr(io.StringIO()):
            self.assertEqual(run(self.root, self.audio.path, None), 2)
        with patch.dict(os.environ), patch("speech._diarization_worker.importlib.import_module") as importer, redirect_stderr(io.StringIO()):
            importer.return_value.Pipeline.from_pretrained.return_value.side_effect = RuntimeError("failed")
            self.assertEqual(run(self.root, self.audio.path, None), 3)
