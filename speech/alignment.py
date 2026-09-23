"""Deterministic timestamp alignment; no model, network or text rewriting."""

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from .diarization import DiarizationResult, DiarizationSegment
from .language import detectSegmentLanguage
from .stt import TranscriptionSegment

MAX_TRANSCRIPT_SEGMENT_JSON_CHARS = 6000
MAX_TRANSCRIPT_CHARACTERS = 120_000
MAX_TRANSCRIPT_SEGMENTS = 5000
# Even if every character needs a six-character JSON escape, the full object
# remains below the protocol AI limit. Normal RU/KZ JSON is much smaller.
MAX_TRANSCRIPT_TEXT_CHARS = 800


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
        self, code: Literal[
            "INVALID_TRANSCRIPT", "INVALID_DIARIZATION", "INVALID_DURATION", "TRANSCRIPT_TOO_LARGE",
        ], message: str,
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


def _split_text(text: str) -> list[str]:
    """Prefer sentence/word boundaries while preserving every source character."""
    if len(text) <= MAX_TRANSCRIPT_TEXT_CHARS:
        return [text]
    result: list[str] = []
    start = 0
    while len(text) - start > MAX_TRANSCRIPT_TEXT_CHARS:
        hard_end = start + MAX_TRANSCRIPT_TEXT_CHARS
        window = text[start:hard_end]
        sentence_ends = [match.end() for match in re.finditer(r"[.!?…][\"'»)]*\s+", window)]
        preferred = sentence_ends[-1] if sentence_ends and sentence_ends[-1] >= MAX_TRANSCRIPT_TEXT_CHARS // 2 else -1
        if preferred < 0:
            whitespace = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
            preferred = whitespace + 1 if whitespace >= MAX_TRANSCRIPT_TEXT_CHARS // 2 else len(window)
        end = start + preferred
        if end <= start:
            end = hard_end
        result.append(text[start:end])
        start = end
    result.append(text[start:])
    return result


def _json_length(segment: TranscriptSegment) -> int:
    return len(json.dumps(segment, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def alignTranscript(
    transcript: Sequence[TranscriptionSegment], diarization: DiarizationResult, *, duration_seconds: float,
) -> AlignmentResult:
    """Assign one speaker to each complete STT segment by maximum interval overlap.

    Ties prefer midpoint containment (half-open speaker intervals), then distance,
    earlier intervals and lexical speaker ID. No-overlap segments use the nearest
    interval. Empty diarization uses SPEAKER_UNKNOWN, which is not a detected voice.
    Source text is never translated. Oversized model segments are split at a
    nearby sentence/word boundary and retain the model segment's original time
    bounds, avoiding invented subsegment timestamps.
    """
    if not _valid_time(duration_seconds):
        raise AlignmentError("INVALID_DURATION", "duration_seconds must be finite and nonnegative.")
    for segment in transcript:
        if (
            not isinstance(segment, dict) or not isinstance(segment.get("text"), str)
            or not _interval_valid(segment.get("start"), segment.get("end"), duration_seconds)
        ):
            raise AlignmentError("INVALID_TRANSCRIPT", "Each STT segment needs string text and 0 <= start <= end <= duration.")
    if sum(len(segment["text"]) for segment in transcript) > MAX_TRANSCRIPT_CHARACTERS:
        raise AlignmentError(
            "TRANSCRIPT_TOO_LARGE", "Transcript exceeds the 120000-character protocol AI limit.",
        )
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
    next_id = 1
    split_segments = False
    for segment in ordered:
        if turns:
            speaker, nearest = _match(segment, turns)
            used_nearest = used_nearest or nearest
        else:
            speaker = "SPEAKER_UNKNOWN"
        chunks = _split_text(segment["text"])
        split_segments = split_segments or len(chunks) > 1
        for text in chunks:
            if next_id > MAX_TRANSCRIPT_SEGMENTS:
                raise AlignmentError(
                    "TRANSCRIPT_TOO_LARGE", "Transcript exceeds the 5000-segment protocol AI limit.",
                )
            output_segment: TranscriptSegment = {
                "id": f"seg-{next_id}", "speakerId": speaker,
                "start": segment["start"], "end": segment["end"], "text": text,
            }
            language = detectSegmentLanguage(text)
            if language is not None:
                output_segment["language"] = language
            if _json_length(output_segment) > MAX_TRANSCRIPT_SEGMENT_JSON_CHARS:
                raise AlignmentError("INVALID_TRANSCRIPT", "Transcript segment exceeds the 6000-character JSON limit.")
            aligned.append(output_segment)
            next_id += 1
    if aligned and not turns:
        warnings.append("No diarization intervals: SPEAKER_UNKNOWN is unassigned and excluded from detectedSpeakers.")
    if used_nearest:
        warnings.append("Some STT segments had no overlapping speaker interval; nearest interval was used.")
    if split_segments:
        warnings.append("Long STT segments were split using original model time bounds; no subsegment timestamps were invented.")
    return AlignmentResult(
        result={
            "durationSeconds": duration_seconds,
            "detectedSpeakers": len({segment["speakerId"] for segment in aligned if segment["speakerId"] != "SPEAKER_UNKNOWN"}),
            "segments": aligned,
        },
        diarization_mode=diarization["mode"], warnings=tuple(warnings),
    )
