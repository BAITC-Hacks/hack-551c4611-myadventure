"""Local speaker turns, with an explicitly selected and marked DEMO mode."""

import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict

from .audio import AudioPreparationError, PreparedAudio, prepareAudio

DiarizationErrorCode = Literal[
    "MODEL_MISSING", "MODEL_LOADING_FAILED", "INVALID_AUDIO", "INFERENCE_FAILED", "INVALID_CONFIG",
]


class DiarizationError(Exception):
    def __init__(self, code: DiarizationErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


class DiarizationSegment(TypedDict):
    speakerId: str
    start: float
    end: float


class DiarizationResult(TypedDict):
    mode: Literal["LOCAL", "DEMO"]
    segments: list[DiarizationSegment]
    warning: str | None


@dataclass(frozen=True)
class SpeakerTurn:
    """Backend label is anonymous and meaningful only inside this recording."""

    speaker: str
    start: float
    end: float


class LocalDiarizationBackend(Protocol):
    """Implementations must process locally and return anonymous speaker turns."""

    def diarize(self, audio: PreparedAudio) -> Iterable[SpeakerTurn]: ...


def _canonical_segments(turns: Iterable[SpeakerTurn], duration: float) -> list[DiarizationSegment]:
    validated: list[SpeakerTurn] = []
    for turn in turns:
        if not isinstance(turn.speaker, str) or not turn.speaker.strip():
            raise ValueError("Diarization returned an empty speaker label")
        if not math.isfinite(turn.start) or not math.isfinite(turn.end) or turn.start < 0 or turn.end < turn.start:
            raise ValueError("Diarization returned invalid timestamps")
        if turn.start < duration:
            validated.append(SpeakerTurn(turn.speaker, turn.start, min(turn.end, duration)))
    validated.sort(key=lambda turn: (turn.start, turn.end, turn.speaker))
    labels: dict[str, str] = {}
    segments: list[DiarizationSegment] = []
    for turn in validated:
        if turn.speaker not in labels:
            labels[turn.speaker] = f"SPEAKER_{len(labels):02d}"
        segments.append({"speakerId": labels[turn.speaker], "start": turn.start, "end": turn.end})
    return segments


class PyannoteLocalBackend:
    """Run pyannote 4.x in an offline worker, isolating heavy dependencies."""

    def __init__(
        self, model_path: str | os.PathLike[str] | None = None, *,
        python_executable: str | None = None, timeout_seconds: float = 600,
        num_speakers: int | None = None,
    ) -> None:
        configured = model_path if model_path is not None else os.environ.get("JINALYS_DIARIZATION_MODEL_DIR")
        if not configured:
            raise DiarizationError("MODEL_MISSING", "Set JINALYS_DIARIZATION_MODEL_DIR to a complete local community-1 bundle.")
        self.model_path = Path(configured).resolve()
        if not (self.model_path / "config.yaml").is_file():
            raise DiarizationError("MODEL_MISSING", "Local diarization bundle is missing config.yaml; no download will be attempted.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise DiarizationError("INVALID_CONFIG", "timeout_seconds must be finite and positive.")
        if num_speakers is not None and (isinstance(num_speakers, bool) or not isinstance(num_speakers, int) or num_speakers < 1):
            raise DiarizationError("INVALID_CONFIG", "num_speakers must be a positive integer.")
        self.python_executable = python_executable or os.environ.get("JINALYS_DIARIZATION_PYTHON") or sys.executable
        self.timeout_seconds = timeout_seconds
        self.num_speakers = num_speakers

    def diarize(self, audio: PreparedAudio) -> list[SpeakerTurn]:
        environment = os.environ.copy()
        environment.update({
            "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "TRANSFORMERS_OFFLINE": "1",
            "PYANNOTE_METRICS_ENABLED": "0", "OTEL_SDK_DISABLED": "true",
            "PYTHONIOENCODING": "utf-8",
        })
        command = [
            self.python_executable, "-m", "speech._diarization_worker",
            "--model", str(self.model_path), "--audio", str(audio.path.resolve()),
        ]
        if self.num_speakers is not None:
            command.extend(["--num-speakers", str(self.num_speakers)])
        try:
            completed = subprocess.run(
                command, cwd=Path(__file__).resolve().parent.parent, env=environment,
                capture_output=True, text=True, encoding="utf-8", timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DiarizationError("INFERENCE_FAILED", "Local diarization worker could not run or exceeded its time limit.") from error
        if completed.returncode != 0:
            if completed.returncode == 2:
                raise DiarizationError("MODEL_LOADING_FAILED", "Cannot load local diarization model; check bundle and optional dependencies.")
            raise DiarizationError("INFERENCE_FAILED", "Local diarization inference failed.")
        try:
            payload: object = json.loads(completed.stdout)
            if not isinstance(payload, list):
                raise ValueError("Expected an array of speaker turns")
            turns: list[SpeakerTurn] = []
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("Expected a speaker turn object")
                speaker, start, end = item.get("speaker"), item.get("start"), item.get("end")
                if not isinstance(speaker, str) or isinstance(start, bool) or isinstance(end, bool):
                    raise ValueError("Invalid speaker turn fields")
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    raise ValueError("Expected numeric timestamps")
                turns.append(SpeakerTurn(speaker, float(start), float(end)))
            return turns
        except (ValueError, TypeError) as error:
            raise DiarizationError("INFERENCE_FAILED", "Local diarization worker returned invalid output.") from error


def diarizeAudio(
    preparedAudio: PreparedAudio, *, mode: Literal["local", "demo"] = "local",
    backend: LocalDiarizationBackend | None = None,
    demo_speakers: int = 1, demo_turn_seconds: float = 5.0,
) -> DiarizationResult:
    """Separate anonymous speakers; no real identity or cloud service is used.

    DEMO assigns fixed-duration turns, irrespective of voices or silence. It is
    never selected automatically when local inference fails. Keep the mode and
    warning alongside segments when displaying or storing demo output.
    """
    if mode not in ("local", "demo"):
        raise DiarizationError("INVALID_CONFIG", "mode must be local or demo.")
    try:
        audio = prepareAudio(preparedAudio.path)
        if audio != preparedAudio:
            raise DiarizationError("INVALID_AUDIO", "Prepared audio no longer matches the source file.")
    except AudioPreparationError as error:
        raise DiarizationError("INVALID_AUDIO", "Cannot diarize invalid audio: " + str(error)) from error
    if mode == "demo":
        if backend is not None:
            raise DiarizationError("INVALID_CONFIG", "A local backend cannot be supplied in DEMO mode.")
        if isinstance(demo_speakers, bool) or not isinstance(demo_speakers, int) or demo_speakers < 1:
            raise DiarizationError("INVALID_CONFIG", "demo_speakers must be a positive integer.")
        if not math.isfinite(demo_turn_seconds) or demo_turn_seconds <= 0:
            raise DiarizationError("INVALID_CONFIG", "demo_turn_seconds must be finite and positive.")
        turn_count = audio.duration_seconds / demo_turn_seconds
        if not math.isfinite(turn_count) or turn_count > 100000:
            raise DiarizationError("INVALID_CONFIG", "DEMO configuration generates too many turns.")
        turns = [
            SpeakerTurn(str(index % demo_speakers), index * demo_turn_seconds,
                        min((index + 1) * demo_turn_seconds, audio.duration_seconds))
            for index in range(math.ceil(turn_count))
        ]
        return {
            "mode": "DEMO", "segments": _canonical_segments(turns, audio.duration_seconds),
            "warning": "DEMO: deterministic time-based speaker labels, not AI diarization or detected speaker identities.",
        }
    if demo_speakers != 1 or demo_turn_seconds != 5.0:
        raise DiarizationError("INVALID_CONFIG", "DEMO options require explicit mode='demo'.")
    try:
        local = backend if backend is not None else PyannoteLocalBackend()
        segments = _canonical_segments(local.diarize(audio), audio.duration_seconds)
        return {"mode": "LOCAL", "segments": segments, "warning": None}
    except DiarizationError:
        raise
    except Exception as error:
        raise DiarizationError("INFERENCE_FAILED", "Local speaker separation failed; no DEMO fallback was substituted.") from error
