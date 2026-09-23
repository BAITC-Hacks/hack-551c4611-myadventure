"""Deterministic timestamp alignment; no model, network or text rewriting."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from .diarization import DiarizationResult, DiarizationSegment
from .language import detectSegmentLanguage
from .stt import TranscriptionSegment


class TranscriptSegment(TypedDict):
    """Python representation of the existing contracts/speech public schema."""

    id: str
    speakerId: str
    speakerName: NotRequired[str]
    start: float
    end: float
    text: str
    language: NotRequired[Literal["ru", "kk", "mixed"]]


class SpeechPipelineResult(TypedDict):
    durationSeconds: float
    detectedSpeakers: int
    segments: list[TranscriptSegment]


@dataclass(frozen=True)
class AlignmentResult:
    """Only result is the external JSON; retain provenance in internal job state."""

    result: SpeechPipelineResult
    diarization_mode: Literal["LOCAL", "DEMO"]
    warnings: tuple[str, ...]


class AlignmentError(Exception):
    def __init__(
        self, code: Literal["INVALID_TRANSCRIPT", "INVALID_DIARIZATION", "INVALID_DURATION"], message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


def _valid_time(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _interval_valid(start: object, end: object, duration: float) -> bool:
    return (
        isinstance(start, (int, float)) and isinstance(end, (int, float))
        and _valid_time(start) and _valid_time(end) and start <= end <= duration
    )


def _match(
    segment: TranscriptionSegment, turns: Sequence[DiarizationSegment],
) -> tuple[str, bool]:
    midpoint = segment["start"] + (segment["end"] - segment["start"]) / 2

    def rank(turn: DiarizationSegment) -> tuple[float, bool, float, float, float, str]:
        overlap = max(0.0, min(segment["end"], turn["end"]) - max(segment["start"], turn["start"]))
        contains_midpoint = turn["start"] <= midpoint < turn["end"]
        distance = max(turn["start"] - midpoint, midpoint - turn["end"], 0.0)
        return (-overlap, not contains_midpoint, distance, turn["start"], turn["end"], turn["speakerId"])

    winner = min(turns, key=rank)
    score = rank(winner)
    # Zero-duration segments can match an interval containing their timestamp.
    nearest_only = score[0] == 0 and score[1]
    return winner["speakerId"], nearest_only


def alignTranscript(
    transcript: Sequence[TranscriptionSegment], diarization: DiarizationResult, *, duration_seconds: float,
) -> AlignmentResult:
    """Assign one speaker to each complete STT segment by maximum interval overlap.

    Ties prefer midpoint containment (half-open speaker intervals), then distance,
    earlier intervals and lexical speaker ID. No-overlap segments use the nearest
    interval. Empty diarization uses SPEAKER_UNKNOWN, which is not a detected voice.
    Source text and timestamps are preserved; no splitting or translation occurs.
    """
    if not _valid_time(duration_seconds):
        raise AlignmentError("INVALID_DURATION", "duration_seconds must be finite and nonnegative.")
    for segment in transcript:
        if (
            not isinstance(segment, dict) or not isinstance(segment.get("text"), str)
            or not _interval_valid(segment.get("start"), segment.get("end"), duration_seconds)
        ):
            raise AlignmentError("INVALID_TRANSCRIPT", "Each STT segment needs string text and 0 <= start <= end <= duration.")
    if (
        not isinstance(diarization, dict) or diarization.get("mode") not in ("LOCAL", "DEMO")
        or not isinstance(diarization.get("segments"), list)
        or (diarization.get("warning") is not None and not isinstance(diarization.get("warning"), str))
    ):
        raise AlignmentError("INVALID_DIARIZATION", "Diarization requires a valid mode, segment list and optional warning.")
    turns = diarization["segments"]
    for turn in turns:
        if (
            not isinstance(turn, dict) or not isinstance(turn.get("speakerId"), str)
            or not turn["speakerId"].strip() or turn["speakerId"] == "SPEAKER_UNKNOWN"
            or not _interval_valid(turn.get("start"), turn.get("end"), duration_seconds)
        ):
            raise AlignmentError("INVALID_DIARIZATION", "Each speaker interval needs a nonempty known speakerId and valid timestamps.")
    warnings: list[str] = []
    if diarization["mode"] == "DEMO":
        warnings.append("DEMO: speaker labels are deterministic simulation, not AI diarization.")
    if diarization.get("warning"):
        warnings.append(diarization["warning"] or "")
    ordered = sorted(transcript, key=lambda item: (item["start"], item["end"]))
    aligned: list[TranscriptSegment] = []
    used_nearest = False
    for index, segment in enumerate(ordered, start=1):
        if turns:
            speaker, nearest = _match(segment, turns)
            used_nearest = used_nearest or nearest
        else:
            speaker = "SPEAKER_UNKNOWN"
        output_segment: TranscriptSegment = {
            "id": f"seg-{index}", "speakerId": speaker,
            "start": segment["start"], "end": segment["end"], "text": segment["text"],
        }
        language = detectSegmentLanguage(segment["text"])
        if language is not None:
            output_segment["language"] = language
        aligned.append(output_segment)
    if aligned and not turns:
        warnings.append("No diarization intervals: SPEAKER_UNKNOWN is unassigned and excluded from detectedSpeakers.")
    if used_nearest:
        warnings.append("Some STT segments had no overlapping speaker interval; nearest interval was used.")
    return AlignmentResult(
        result={
            "durationSeconds": duration_seconds,
            "detectedSpeakers": len({segment["speakerId"] for segment in aligned if segment["speakerId"] != "SPEAKER_UNKNOWN"}),
            "segments": aligned,
        },
        diarization_mode=diarization["mode"], warnings=tuple(warnings),
    )
