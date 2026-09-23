"""Local multilingual Whisper inference. Model provisioning is a separate step."""

import importlib
import math
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

from .audio import AudioPreparationError, PreparedAudio, prepareAudio

STTErrorCode = Literal["MODEL_MISSING", "MODEL_LOADING_FAILED", "INVALID_AUDIO", "INFERENCE_FAILED"]
Language = Literal["ru", "kk", "mixed"]


class TranscriptionError(Exception):
    def __init__(self, code: STTErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


class TranscriptionSegment(TypedDict):
    start: float
    end: float
    text: str


class _Segment(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...

    @property
    def text(self) -> str: ...


class _ModelInfo(Protocol):
    @property
    def is_multilingual(self) -> bool: ...


class _Model(Protocol):
    @property
    def model(self) -> _ModelInfo: ...

    def transcribe(
        self, audio: str, *, task: str, language: str | None, multilingual: bool,
        vad_filter: bool, without_timestamps: bool, condition_on_previous_text: bool,
        beam_size: int, word_timestamps: bool, hallucination_silence_threshold: float,
    ) -> tuple[Iterable[_Segment], object]: ...


class _ModelFactory(Protocol):
    def __call__(
        self, path: str, *, device: str, compute_type: str, local_files_only: bool,
        use_auth_token: bool,
    ) -> _Model: ...


def _model_directory(model_path: str | os.PathLike[str] | None) -> Path:
    configured = model_path if model_path is not None else os.environ.get("JINALYS_STT_MODEL_DIR")
    if not configured:
        raise TranscriptionError("MODEL_MISSING", "Set JINALYS_STT_MODEL_DIR to a local multilingual faster-whisper model directory.")
    path = Path(configured)
    # tokenizer.json is mandatory: faster-whisper otherwise attempts a download.
    required = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.json")
    try:
        if not path.is_dir() or any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in required):
            raise TranscriptionError("MODEL_MISSING", "Local model is missing or incomplete; expected " + ", ".join(required) + ".")
        return path.resolve()
    except OSError as error:
        raise TranscriptionError("MODEL_LOADING_FAILED", "Cannot read the local model directory.") from error


@lru_cache(maxsize=1)
def _load_model(path: str, device: str, compute_type: str) -> _Model:
    try:
        module = importlib.import_module("faster_whisper")
        # The dependency is untyped; contain it at this explicitly typed boundary.
        factory = cast(_ModelFactory, module.WhisperModel)
        model = factory(path, device=device, compute_type=compute_type, local_files_only=True, use_auth_token=False)
        if not model.model.is_multilingual:
            raise ValueError("English-only models cannot transcribe RU/KZ")
        return model
    except Exception as error:
        raise TranscriptionError(
            "MODEL_LOADING_FAILED",
            "Cannot load local multilingual STT model. Check speech/requirements-stt.txt, model files and device compatibility.",
        ) from error


def transcribeAudio(
    preparedAudio: PreparedAudio, *, model_path: str | os.PathLike[str] | None = None,
    language: Language | None = None, device: Literal["cpu", "cuda"] = "cpu",
    compute_type: str = "int8",
) -> list[TranscriptionSegment]:
    """Return sorted source-language text and timestamps, without translation.

    Defaults to CPU int8 and automatic per-segment language detection. Explicit
    ru/kk hints are for monolingual recordings; mixed keeps detection enabled.
    PyAV in faster-whisper converts input to mono 16 kHz in memory. Silero VAD
    weights ship with the dependency. Runtime never provisions model assets.
    """
    if language not in (None, "ru", "kk", "mixed"):
        raise ValueError("language must be ru, kk, mixed or None")
    try:
        checked = prepareAudio(preparedAudio.path)
        if checked != preparedAudio:
            raise TranscriptionError("INVALID_AUDIO", "Prepared audio metadata no longer matches the source file.")
    except AudioPreparationError as error:
        raise TranscriptionError("INVALID_AUDIO", "Cannot transcribe invalid audio: " + str(error)) from error
    directory = _model_directory(model_path)
    model = _load_model(str(directory), device, compute_type)
    try:
        segments, _ = model.transcribe(
            str(checked.path), task="transcribe",
            language=language if language in ("ru", "kk") else None,
            multilingual=language in (None, "mixed"), vad_filter=True,
            without_timestamps=False, condition_on_previous_text=False, beam_size=5,
            word_timestamps=True, hallucination_silence_threshold=1.0,
        )
        result: list[TranscriptionSegment] = []
        # Iteration performs inference too; keep it inside the error boundary.
        for segment in segments:
            start, end = float(segment.start), float(segment.end)
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError("Model returned invalid timestamps")
            if not isinstance(segment.text, str):
                raise ValueError("Model returned non-text output")
            if start >= checked.duration_seconds:
                continue
            end = min(end, checked.duration_seconds)
            if segment.text.strip():
                result.append({"start": start, "end": end, "text": segment.text.strip()})
        result.sort(key=lambda item: (item["start"], item["end"]))
        return result
    except Exception as error:
        raise TranscriptionError("INFERENCE_FAILED", "Local speech transcription failed; check the model, audio and available memory.") from error
