"""Model-independent public entry point for meeting transcription."""

import math
import os
from dataclasses import dataclass, field
from typing import Literal

from .alignment import AlignmentError, SpeechPipelineResult, alignTranscript
from .audio import AudioPreparationError, prepareAudio
from .diarization import DiarizationError, diarizeAudio
from .stt import TranscriptionError, transcribeAudio

PipelineStage = Literal["configuration", "preprocessing", "stt", "diarization", "alignment"]


class PipelineError(Exception):
    """Stable boundary for callers; stage/code identify a controlled failure."""

    def __init__(self, stage: PipelineStage, code: str, message: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(message)


@dataclass
class PipelineDiagnostics:
    """Internal job metadata; deliberately excluded from SpeechPipelineResult."""

    mode: Literal["LOCAL", "DEMO"] = "LOCAL"
    warnings: list[str] = field(default_factory=list)
    completed: bool = False


def processMeetingAudio(
    file: str | os.PathLike[str], *, mode: Literal["local", "demo"] = "local",
    diagnostics: PipelineDiagnostics | None = None, demo_speakers: int = 1,
    demo_turn_seconds: float = 5.0,
) -> SpeechPipelineResult:
    """Prepare → local STT → diarize → align/mark language → public JSON.

    Default use needs only a file; worker model settings belong to deployment.
    Explicit demo mode simulates diarization ONLY and still runs real STT. It
    requires a diagnostics holder so DEMO provenance cannot be silently dropped
    by the entry point. The caller must retain/display it alongside the job.
    """
    stage: PipelineStage = "configuration"
    try:
        if diagnostics is not None:
            diagnostics.completed = False
            diagnostics.warnings.clear()
            diagnostics.mode = "DEMO" if mode == "demo" else "LOCAL"
        if mode not in ("local", "demo"):
            raise PipelineError(stage, "INVALID_CONFIG", "mode must be local or demo.")
        if mode == "demo" and diagnostics is None:
            raise PipelineError(stage, "DEMO_REQUIRES_DIAGNOSTICS", "DEMO requires PipelineDiagnostics; preserve its mode/warnings in job state.")
        if isinstance(demo_speakers, bool) or not isinstance(demo_speakers, int) or demo_speakers < 1:
            raise PipelineError(stage, "INVALID_CONFIG", "demo_speakers must be a positive integer.")
        if not math.isfinite(demo_turn_seconds) or demo_turn_seconds <= 0:
            raise PipelineError(stage, "INVALID_CONFIG", "demo_turn_seconds must be finite and positive.")
        if mode == "local" and (demo_speakers != 1 or demo_turn_seconds != 5.0):
            raise PipelineError(stage, "INVALID_CONFIG", "DEMO options require explicit mode='demo'.")
        if mode == "demo" and diagnostics is not None:
            diagnostics.warnings.append("DEMO: diarization uses simulated speaker turns, not detected voices. STT remains local AI.")
        stage = "preprocessing"
        prepared = prepareAudio(file)
        stage = "stt"
        transcript = transcribeAudio(prepared)
        stage = "diarization"
        speakers = diarizeAudio(prepared, mode=mode, demo_speakers=demo_speakers, demo_turn_seconds=demo_turn_seconds)
        stage = "alignment"
        aligned = alignTranscript(transcript, speakers, duration_seconds=prepared.duration_seconds)
        if diagnostics is not None:
            diagnostics.mode = aligned.diarization_mode
            diagnostics.warnings.extend(aligned.warnings)
            diagnostics.completed = True
        return aligned.result
    except PipelineError:
        raise
    except (AudioPreparationError, TranscriptionError, DiarizationError, AlignmentError) as error:
        raise PipelineError(stage, error.code, f"Meeting audio {stage} failed: {error}") from error
    except Exception as error:
        # Unexpected backend faults are contained; raw paths/logs aren't public messages.
        raise PipelineError(stage, "STAGE_FAILED", f"Meeting audio {stage} failed unexpectedly.") from error
