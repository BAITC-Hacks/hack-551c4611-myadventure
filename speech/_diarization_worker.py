"""Optional pyannote worker. Spawned with offline/telemetry settings by adapter."""

import argparse
import importlib
import json
import os
import sys
from collections.abc import Iterable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Protocol, cast


class _Turn(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...


class _Annotation(Protocol):
    def itertracks(self, *, yield_label: bool) -> Iterable[tuple[_Turn, object, str]]: ...


class _Output(Protocol):
    @property
    def speaker_diarization(self) -> _Annotation: ...


class _Pipeline(Protocol):
    def __call__(self, audio: str, *, num_speakers: int | None) -> _Output: ...


class _Loader(Protocol):
    def __call__(self, checkpoint: str, *, token: bool) -> _Pipeline | None: ...


def _configure_ffmpeg() -> object | None:
    """Make an explicitly configured shared FFmpeg bundle visible to TorchCodec."""
    configured = os.environ.get("JINALYS_FFMPEG_DIR")
    if not configured:
        return None
    directory = Path(configured).resolve()
    if not directory.is_dir():
        raise FileNotFoundError("JINALYS_FFMPEG_DIR is not a directory")
    os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        return cast(object, add_dll_directory(str(directory)))
    return None


def run(model: Path, audio: Path, num_speakers: int | None) -> int:
    """Return 2 for setup failure, 3 for inference failure; stdout is JSON only."""
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYANNOTE_METRICS_ENABLED": "0", "OTEL_SDK_DISABLED": "true",
    })
    try:
        if not (model / "config.yaml").is_file():
            raise FileNotFoundError("Missing local config.yaml")
        # Keep the handle alive until inference completes. Python 3.8+ no longer
        # resolves dependent Windows DLLs from PATH alone.
        ffmpeg_dll_directory = _configure_ffmpeg()
        with redirect_stdout(sys.stderr):
            module = importlib.import_module("pyannote.audio")
            loader = cast(_Loader, module.Pipeline.from_pretrained)
            pipeline = loader(str(model), token=False)
            if pipeline is None:
                raise RuntimeError("Local pipeline was not loaded")
    except Exception:
        print("MODEL_LOADING_FAILED: check local bundle and pyannote dependencies.", file=sys.stderr)
        return 2
    try:
        _ = ffmpeg_dll_directory
        with redirect_stdout(sys.stderr):
            output = pipeline(str(audio), num_speakers=num_speakers)
            result = [
                {"speaker": speaker, "start": float(turn.start), "end": float(turn.end)}
                for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True)
            ]
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception:
        print("INFERENCE_FAILED: local diarization could not process audio.", file=sys.stderr)
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Local-only pyannote diarization worker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--num-speakers", type=int)
    args = parser.parse_args()
    return run(args.model, args.audio, args.num_speakers)


if __name__ == "__main__":
    raise SystemExit(main())
