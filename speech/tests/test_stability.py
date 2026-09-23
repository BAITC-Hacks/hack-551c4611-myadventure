import contextlib
import io
import os
import socket
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speech import PipelineDiagnostics, PipelineError, processMeetingAudio
from speech.stt import _load_model


class StabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio.wav"
        with wave.open(str(self.audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 16000)
        self.model = self.root / "model"
        self.model.mkdir()
        for name in ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.json"):
            (self.model / name).write_text("fixture", encoding="utf-8")
        _load_model.cache_clear()
        self.addCleanup(_load_model.cache_clear)

    def test_model_is_cached_across_repeated_transcription(self) -> None:
        segment = SimpleNamespace(start=0.0, end=1.0, text="Добрый день, коллеги.")
        model = SimpleNamespace(model=SimpleNamespace(is_multilingual=True))
        model.transcribe = lambda *_args, **_kwargs: ([segment], object())
        module = SimpleNamespace(WhisperModel=lambda *_args, **_kwargs: model)
        with patch.dict(os.environ, {"JINALYS_STT_MODEL_DIR": str(self.model)}), patch(
            "speech.stt.importlib.import_module", return_value=module,
        ) as importer:
            for _ in range(2):
                diagnostics = PipelineDiagnostics()
                result = processMeetingAudio(
                    self.audio, mode="demo", diagnostics=diagnostics,
                    demo_turn_seconds=1,
                )
                self.assertTrue(diagnostics.completed)
                self.assertEqual(result["segments"][0]["speakerId"], "SPEAKER_00")
        self.assertEqual(importer.call_count, 1)
        self.assertEqual((_load_model.cache_info().misses, _load_model.cache_info().hits), (1, 1))

    def test_missing_model_error_is_clear_and_does_not_attempt_network(self) -> None:
        with patch.dict(os.environ, {"JINALYS_STT_MODEL_DIR": str(self.root / "missing")}, clear=True):
            with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                with self.assertRaises(PipelineError) as caught:
                    processMeetingAudio(self.audio, mode="demo", diagnostics=PipelineDiagnostics())
        self.assertEqual((caught.exception.stage, caught.exception.code), ("stt", "MODEL_MISSING"))
        self.assertIn("missing or incomplete", str(caught.exception))

    def test_two_requests_do_not_create_files_or_corrupt_state(self) -> None:
        before = {path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        with patch("speech.pipeline.transcribeAudio", return_value=[{"start": 0, "end": 1, "text": "Добрый день."}]):
            diagnostics = PipelineDiagnostics()
            first = processMeetingAudio(self.audio, mode="demo", diagnostics=diagnostics)
            diagnostics.warnings.append("caller state")
            second = processMeetingAudio(self.audio, mode="demo", diagnostics=diagnostics)
        after = {path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertNotIn("caller state", diagnostics.warnings)
        self.assertTrue(diagnostics.completed)

    def test_demo_pipeline_does_not_load_diarization_backend(self) -> None:
        with patch("speech.pipeline.transcribeAudio", return_value=[]), patch("speech.diarization.PyannoteLocalBackend") as backend:
            with contextlib.redirect_stdout(io.StringIO()):
                processMeetingAudio(self.audio, mode="demo", diagnostics=PipelineDiagnostics())
        backend.assert_not_called()
