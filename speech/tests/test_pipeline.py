import json
import os
import socket
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from speech import PipelineDiagnostics, PipelineError, processMeetingAudio
from speech.alignment import SpeechPipelineResult
from speech.diarization import DiarizationError
from speech.stt import TranscriptionError

FIXTURE = Path(__file__).parent / "fixtures" / "ru.wav"


def validate_schema(result: SpeechPipelineResult) -> None:
    script = """
import {readFileSync} from 'node:fs';
import Ajv from 'ajv';
const validate = new Ajv({strict:true}).compile(JSON.parse(readFileSync('schema.json','utf8')));
if (!validate(JSON.parse(readFileSync(0,'utf8')))) throw Error(JSON.stringify(validate.errors));
"""
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[2] / "contracts" / "speech",
        input=json.dumps(result, ensure_ascii=True, allow_nan=False),
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=True,
    )


class PipelineTests(unittest.TestCase):
    def test_pipeline_order_and_exact_contract(self) -> None:
        # Only backend inference is doubled; actual preprocessing/alignment run.
        with patch("speech.pipeline.transcribeAudio") as stt, patch("speech.pipeline.diarizeAudio") as diarizer:
            calls: list[str] = []

            def transcribe(_audio: object) -> list[dict[str, object]]:
                calls.append("stt")
                return [{"start": 0, "end": 1, "text": "Подготовьте итоговый отчёт."}]

            def diarize(_audio: object, **_options: object) -> dict[str, object]:
                calls.append("diarization")
                return {"mode": "LOCAL", "segments": [{"start": 0, "end": 2, "speakerId": "SPEAKER_00"}], "warning": None}

            stt.side_effect = transcribe
            diarizer.side_effect = diarize
            diagnostics = PipelineDiagnostics()
            result = processMeetingAudio(FIXTURE, diagnostics=diagnostics)
        self.assertEqual(calls, ["stt", "diarization"])
        self.assertTrue(diagnostics.completed)
        self.assertEqual(set(result), {"durationSeconds", "detectedSpeakers", "segments"})
        self.assertEqual(result["detectedSpeakers"], 1)
        self.assertEqual(result["segments"][0]["language"], "ru")
        validate_schema(result)

    def test_invalid_input_stops_before_inference(self) -> None:
        with patch("speech.pipeline.transcribeAudio") as stt:
            with self.assertRaises(PipelineError) as caught:
                processMeetingAudio(FIXTURE.parent / "missing.wav")
            stt.assert_not_called()
        self.assertEqual((caught.exception.stage, caught.exception.code), ("preprocessing", "INVALID_FILE"))

    def test_stt_error_and_model_missing(self) -> None:
        for code in ["MODEL_MISSING", "INFERENCE_FAILED"]:
            error = TranscriptionError("MODEL_MISSING" if code == "MODEL_MISSING" else "INFERENCE_FAILED", "test failure")
            with patch("speech.pipeline.transcribeAudio", side_effect=error), patch("speech.pipeline.diarizeAudio") as diarizer:
                with self.assertRaises(PipelineError) as caught:
                    processMeetingAudio(FIXTURE)
                diarizer.assert_not_called()
            self.assertEqual((caught.exception.stage, caught.exception.code), ("stt", code))

    def test_real_missing_stt_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(PipelineError) as caught:
            processMeetingAudio(FIXTURE)
        self.assertEqual((caught.exception.stage, caught.exception.code), ("stt", "MODEL_MISSING"))

    def test_diarization_error_no_automatic_demo(self) -> None:
        for error in [DiarizationError("MODEL_MISSING", "missing"), DiarizationError("INFERENCE_FAILED", "failed")]:
            with patch("speech.pipeline.transcribeAudio", return_value=[]), patch("speech.pipeline.diarizeAudio", side_effect=error):
                with self.assertRaises(PipelineError) as caught:
                    processMeetingAudio(FIXTURE)
            self.assertEqual(caught.exception.stage, "diarization")
            self.assertEqual(caught.exception.code, error.code)

    def test_unexpected_errors_are_contained(self) -> None:
        for function, stage in [("prepareAudio", "preprocessing"), ("transcribeAudio", "stt"), ("diarizeAudio", "diarization"), ("alignTranscript", "alignment")]:
            with patch("speech.pipeline.transcribeAudio", return_value=[]), patch("speech.pipeline.diarizeAudio", return_value={"mode": "LOCAL", "segments": [], "warning": None}):
                with patch("speech.pipeline." + function, side_effect=RuntimeError("private backend details")):
                    with self.assertRaises(PipelineError) as caught:
                        processMeetingAudio(FIXTURE)
            self.assertEqual((caught.exception.stage, caught.exception.code), (stage, "STAGE_FAILED"))
            self.assertNotIn("private backend details", str(caught.exception))

    def test_demo_requires_provenance_holder(self) -> None:
        with self.assertRaises(PipelineError) as caught:
            processMeetingAudio(FIXTURE, mode="demo")
        self.assertEqual(caught.exception.code, "DEMO_REQUIRES_DIAGNOSTICS")

    def test_demo_marker_and_empty_transcript(self) -> None:
        diagnostics = PipelineDiagnostics(completed=True, warnings=["old"])
        with patch("speech.pipeline.transcribeAudio", return_value=[]):
            result = processMeetingAudio(FIXTURE, mode="demo", diagnostics=diagnostics)
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["detectedSpeakers"], 0)
        self.assertTrue(diagnostics.completed)
        self.assertEqual(diagnostics.mode, "DEMO")
        self.assertIn("DEMO", " ".join(diagnostics.warnings))
        self.assertNotIn("old", diagnostics.warnings)
        validate_schema(result)

    def test_failed_call_resets_completed(self) -> None:
        diagnostics = PipelineDiagnostics(completed=True)
        with self.assertRaises(PipelineError):
            processMeetingAudio(FIXTURE.parent / "missing", diagnostics=diagnostics)
        self.assertFalse(diagnostics.completed)

    def test_explicit_demo_turn_duration(self) -> None:
        with patch("speech.pipeline.transcribeAudio", return_value=[
            {"start": 0, "end": 2, "text": "Добрый день."},
            {"start": 8, "end": 10, "text": "Спасибо, коллеги."},
        ]):
            result = processMeetingAudio(FIXTURE, mode="demo", demo_speakers=2, demo_turn_seconds=7, diagnostics=PipelineDiagnostics())
        self.assertEqual(result["detectedSpeakers"], 2)
        with patch("speech.pipeline.transcribeAudio") as stt:
            with self.assertRaises(PipelineError):
                processMeetingAudio(FIXTURE, mode="demo", demo_turn_seconds=0, diagnostics=PipelineDiagnostics())
            stt.assert_not_called()


@unittest.skipUnless(os.environ.get("JINALYS_STT_MODEL_DIR"), "Real local STT weights not configured")
class PipelineEndToEndTests(unittest.TestCase):
    def assert_output(self, result: SpeechPipelineResult) -> None:
        self.assertGreaterEqual(result["durationSeconds"], 0)
        self.assertGreaterEqual(result["detectedSpeakers"], 1)
        self.assertIsInstance(result["segments"], list)
        self.assertTrue(result["segments"])
        ids = [item["id"] for item in result["segments"]]
        self.assertEqual(len(ids), len(set(ids)))
        starts = [item["start"] for item in result["segments"]]
        self.assertEqual(starts, sorted(starts))
        for item in result["segments"]:
            self.assertTrue(item["id"])
            self.assertTrue(item["speakerId"])
            self.assertGreaterEqual(item["start"], 0)
            self.assertGreaterEqual(item["end"], item["start"])
            self.assertIsInstance(item["text"], str)
        validate_schema(result)

    def test_real_audio_real_stt_explicit_demo_diarization(self) -> None:
        diagnostics = PipelineDiagnostics()
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline inference only")):
            result = processMeetingAudio(FIXTURE, mode="demo", diagnostics=diagnostics)
        self.assert_output(result)
        self.assertEqual(diagnostics.mode, "DEMO")
        self.assertTrue(diagnostics.completed)
        self.assertIn("DEMO", " ".join(diagnostics.warnings))
        self.assertIn("ночного", " ".join(item["text"] for item in result["segments"]).lower())

    @unittest.skipUnless(os.environ.get("JINALYS_DIARIZATION_MODEL_DIR"), "Real diarization bundle not configured")
    def test_fully_local_model_pipeline(self) -> None:
        diagnostics = PipelineDiagnostics()
        result = processMeetingAudio(FIXTURE, diagnostics=diagnostics)
        self.assert_output(result)
        self.assertEqual(diagnostics.mode, "LOCAL")
