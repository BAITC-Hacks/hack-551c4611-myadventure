import copy
import json
import subprocess
import unittest
from pathlib import Path
from typing import cast

from speech import AlignmentError, AlignmentResult, alignTranscript
from speech.alignment import MAX_TRANSCRIPT_SEGMENT_JSON_CHARS
from speech.diarization import DiarizationResult, DiarizationSegment
from speech.stt import TranscriptionSegment


def stt(start: float, end: float, text: str = "До пятницы подготовьте итоговый отчёт.") -> TranscriptionSegment:
    return {"start": start, "end": end, "text": text}


def turn(speaker: str, start: float, end: float) -> DiarizationSegment:
    return {"speakerId": speaker, "start": start, "end": end}


def diarization(*turns: DiarizationSegment) -> DiarizationResult:
    return {"mode": "LOCAL", "segments": list(turns), "warning": None}


class AlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.speakers = diarization(turn("SPEAKER_00", 0, 5), turn("SPEAKER_01", 5, 10))

    def align(self, *segments: TranscriptionSegment) -> AlignmentResult:
        return alignTranscript(segments, self.speakers, duration_seconds=10)

    def test_inside_speaker_interval(self) -> None:
        result = self.align(stt(1, 3)).result
        self.assertEqual(result["segments"], [{"id": "seg-1", "speakerId": "SPEAKER_00", **stt(1, 3), "language": "ru"}])
        self.assertEqual(result["detectedSpeakers"], 1)

    def test_crosses_two_intervals_largest_overlap_wins(self) -> None:
        result = self.align(stt(4, 8)).result
        self.assertEqual(result["segments"][0]["speakerId"], "SPEAKER_01")
        self.assertEqual(len(result["segments"]), 1)

    def test_boundary_and_equal_overlap_tiebreak(self) -> None:
        result = self.align(stt(4, 5), stt(5, 6), stt(5, 5), stt(4, 6)).result
        self.assertEqual([s["speakerId"] for s in result["segments"]], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_01"])

    def test_empty_transcript(self) -> None:
        self.assertEqual(self.align().result, {"durationSeconds": 10, "detectedSpeakers": 0, "segments": []})

    def test_empty_diarization_unknown_is_not_detected(self) -> None:
        aligned = alignTranscript([stt(1, 2)], diarization(), duration_seconds=10)
        self.assertEqual(aligned.result["segments"][0]["speakerId"], "SPEAKER_UNKNOWN")
        self.assertEqual(aligned.result["detectedSpeakers"], 0)
        self.assertTrue(aligned.warnings)

    def test_no_overlap_nearest_and_earlier_tie(self) -> None:
        speakers = diarization(turn("SPEAKER_00", 0, 1), turn("SPEAKER_01", 5, 10))
        result = alignTranscript([stt(2.5, 3.5), stt(4, 4.5)], speakers, duration_seconds=10)
        self.assertEqual([s["speakerId"] for s in result.result["segments"]], ["SPEAKER_00", "SPEAKER_01"])
        self.assertTrue(result.warnings)

    def test_unique_ids_sorted_and_input_unchanged(self) -> None:
        transcript = [stt(8, 9), stt(1, 2), stt(1, 2), stt(3, 4)]
        speakers = copy.deepcopy(self.speakers)
        original = copy.deepcopy(transcript)
        aligned = alignTranscript(transcript, speakers, duration_seconds=10)
        segments = aligned.result["segments"]
        self.assertEqual([s["start"] for s in segments], [1, 1, 3, 8])
        self.assertEqual([s["id"] for s in segments], ["seg-1", "seg-2", "seg-3", "seg-4"])
        self.assertEqual(transcript, original)
        self.assertEqual(speakers, self.speakers)
        self.assertEqual(aligned, alignTranscript(transcript, speakers, duration_seconds=10))
        self.assertEqual(aligned.result["detectedSpeakers"], 2)

    def test_unsorted_diarization_and_repeated_speaker(self) -> None:
        speakers = diarization(turn("SPEAKER_02", 6, 10), turn("SPEAKER_01", 3, 6), turn("SPEAKER_02", 0, 3))
        result = alignTranscript([stt(1, 2), stt(7, 8)], speakers, duration_seconds=10).result
        self.assertEqual([s["speakerId"] for s in result["segments"]], ["SPEAKER_02", "SPEAKER_02"])
        self.assertEqual(result["detectedSpeakers"], 1)

    def test_overlapping_speakers_are_deterministic(self) -> None:
        first = turn("SPEAKER_02", 0, 10)
        second = turn("SPEAKER_01", 0, 10)
        result = alignTranscript([stt(1, 2)], diarization(first, second), duration_seconds=10)
        reversed_result = alignTranscript([stt(1, 2)], diarization(second, first), duration_seconds=10)
        self.assertEqual(result, reversed_result)
        self.assertEqual(result.result["segments"][0]["speakerId"], "SPEAKER_01")

    def test_text_preserved_verbatim(self) -> None:
        text = "  Әріптестер, отчёт жұмаға дейін.\n"
        self.assertEqual(self.align(stt(1, 2, text)).result["segments"][0]["text"], text)

    def test_long_segment_is_bounded_without_inventing_timestamps(self) -> None:
        text = ("До пятницы подготовьте итоговый отчёт. " * 180).strip()
        aligned = self.align(stt(1, 9, text))
        segments = aligned.result["segments"]
        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segment["text"] for segment in segments), text)
        self.assertEqual({segment["speakerId"] for segment in segments}, {"SPEAKER_01"})
        self.assertEqual({(segment["start"], segment["end"]) for segment in segments}, {(1, 9)})
        self.assertEqual([segment["id"] for segment in segments], [f"seg-{index}" for index in range(1, len(segments) + 1)])
        self.assertTrue(all(
            len(json.dumps(segment, ensure_ascii=False, separators=(",", ":")))
            <= MAX_TRANSCRIPT_SEGMENT_JSON_CHARS
            for segment in segments
        ))
        self.assertIn("no subsegment timestamps", " ".join(aligned.warnings))

    def test_end_of_30_minute_recording_keeps_global_time(self) -> None:
        task = "До конца дня отправьте подписанный договор."
        speakers = diarization(
            turn("SPEAKER_00", 0, 900), turn("SPEAKER_01", 900, 1800),
        )
        result = alignTranscript([stt(1792, 1799, task)], speakers, duration_seconds=1800).result
        self.assertEqual(result["durationSeconds"], 1800)
        self.assertEqual(result["segments"], [{
            "id": "seg-1", "speakerId": "SPEAKER_01", "start": 1792,
            "end": 1799, "text": task, "language": "ru",
        }])

    def test_protocol_ai_total_limits_are_controlled(self) -> None:
        with self.assertRaises(AlignmentError) as characters:
            self.align(stt(1, 2, "а" * 120_001))
        self.assertEqual(characters.exception.code, "TRANSCRIPT_TOO_LARGE")
        many = [stt(1, 2, "") for _ in range(5001)]
        with self.assertRaises(AlignmentError) as segments:
            alignTranscript(many, self.speakers, duration_seconds=10)
        self.assertEqual(segments.exception.code, "TRANSCRIPT_TOO_LARGE")

    def test_demo_provenance_is_retained_outside_public_json(self) -> None:
        speakers = copy.deepcopy(self.speakers)
        speakers["mode"] = "DEMO"
        result = alignTranscript([stt(1, 2)], speakers, duration_seconds=10)
        self.assertEqual(result.diarization_mode, "DEMO")
        self.assertIn("DEMO", " ".join(result.warnings))
        self.assertEqual(set(result.result), {"durationSeconds", "detectedSpeakers", "segments"})

    def test_invalid_transcript_is_controlled(self) -> None:
        malformed: list[object] = [stt(-1, 2), stt(2, 1), stt(0, 11), stt(0, float("nan")), {"start": 0, "end": 1, "text": 7}, {"start": 0, "end": 1}, {"start": True, "end": 1, "text": "x"}]
        for segment in malformed:
            with self.subTest(segment=segment), self.assertRaises(AlignmentError) as caught:
                self.align(cast(TranscriptionSegment, segment))
            self.assertEqual(caught.exception.code, "INVALID_TRANSCRIPT")

    def test_invalid_diarization_is_controlled(self) -> None:
        for segment in [turn("", 0, 1), turn("SPEAKER_00", 3, 2), turn("SPEAKER_00", 0, float("inf")), turn("SPEAKER_UNKNOWN", 0, 1)]:
            with self.subTest(segment=segment), self.assertRaises(AlignmentError) as caught:
                alignTranscript([stt(0, 1)], diarization(segment), duration_seconds=10)
            self.assertEqual(caught.exception.code, "INVALID_DIARIZATION")
        with self.assertRaises(AlignmentError):
            alignTranscript([], cast(DiarizationResult, {"mode": "invalid", "segments": []}), duration_seconds=0)

    def test_invalid_duration_and_empty_zero_duration(self) -> None:
        for duration in [-1, float("nan"), float("inf"), True]:
            with self.subTest(duration=duration), self.assertRaises(AlignmentError) as caught:
                alignTranscript([], diarization(), duration_seconds=duration)
            self.assertEqual(caught.exception.code, "INVALID_DURATION")
        self.assertEqual(alignTranscript([], diarization(), duration_seconds=0).result["segments"], [])

    def test_outputs_validate_against_existing_json_schema(self) -> None:
        # Use the existing JS contract validator; no second schema or validator dependency.
        demo = copy.deepcopy(self.speakers)
        demo["mode"] = "DEMO"
        outputs = [
            self.align(stt(1, 3)).result, self.align(stt(4, 8)).result,
            self.align(stt(5, 5)).result, self.align().result,
            alignTranscript([stt(1, 2)], diarization(), duration_seconds=10).result,
            alignTranscript([stt(1, 2)], demo, duration_seconds=10).result,
        ]
        for text, language in [
            ("Сегодня нужно подготовить итоговый отчёт.", "ru"),
            ("Әріптестер, бүгін жиналысты бастаймыз.", "kk"),
            ("Асқар, осы аптада подрядчикпен сөйлесіп, новый график жасап беріңіз.", "mixed"),
            ("ОК", None),
        ]:
            result = self.align(stt(1, 2, text)).result
            self.assertEqual(result["segments"][0].get("language"), language)
            if language is None:
                self.assertNotIn("language", result["segments"][0])
            self.assertEqual(result["segments"][0]["text"], text)
            outputs.append(result)
        validator = """
import { readFileSync } from 'node:fs';
import Ajv from 'ajv';
const schema = JSON.parse(readFileSync('schema.json', 'utf8'));
const validate = new Ajv({strict: true}).compile(schema);
for (const value of JSON.parse(readFileSync(0, 'utf8'))) {
  if (!validate(value)) throw new Error(JSON.stringify(validate.errors));
}
"""
        completed = subprocess.run(
            ["node", "--input-type=module", "-e", validator],
            cwd=Path(__file__).resolve().parents[2] / "contracts" / "speech",
            input=json.dumps(outputs, ensure_ascii=True, allow_nan=False),
            capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
