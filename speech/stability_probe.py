"""Two-run local inference stability check used by Stage 10 verification."""

import argparse
import ctypes
import gc
import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from .alignment import SpeechPipelineResult
from .fixture_suite import validate_contract
from .pipeline import PipelineDiagnostics, processMeetingAudio
from .stt import _load_model

DEFAULT_AUDIO = Path(__file__).parent / "tests" / "fixtures" / "ru.wav"
REPOSITORY = Path(__file__).resolve().parent.parent
MEMORY_GROWTH_LIMIT = 256 * 1024 * 1024


def _working_set_bytes() -> int | None:
    """Return current Windows working set; unsupported platforms return None."""
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.WorkingSetSize)


def _tree_state(directory: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(directory)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(path.resolve())


@contextmanager
def _offline() -> Iterator[None]:
    def blocked_connect(_socket: socket.socket, _address: object) -> None:
        raise AssertionError("Outbound network access is forbidden during local inference")

    with patch.object(socket.socket, "connect", blocked_connect):
        yield


def run_stability_probe(audio: Path = DEFAULT_AUDIO) -> dict[str, object]:
    model_setting = os.environ.get("JINALYS_STT_MODEL_DIR")
    if not model_setting:
        raise RuntimeError("Set JINALYS_STT_MODEL_DIR to the local STT model directory.")
    model_directory = Path(model_setting).resolve()
    model_before = _tree_state(model_directory)
    audio_before = (audio.stat().st_size, audio.stat().st_mtime_ns)
    _load_model.cache_clear()

    with tempfile.TemporaryDirectory(prefix="jinalys-stability-") as folder:
        temp_directory = Path(folder)
        environment = {"TMP": str(temp_directory), "TEMP": str(temp_directory)}
        runs: list[dict[str, object]] = []
        outputs: list[SpeechPipelineResult] = []
        with patch.dict(os.environ, environment), _offline():
            for number in (1, 2):
                diagnostics = PipelineDiagnostics()
                started = time.monotonic()
                result = processMeetingAudio(audio, mode="demo", diagnostics=diagnostics)
                validate_contract(result)
                gc.collect()
                cache = _load_model.cache_info()
                runs.append({
                    "run": number, "status": "PASS", "elapsedSeconds": round(time.monotonic() - started, 3),
                    "workingSetBytes": _working_set_bytes(),
                    "cache": {"hits": cache.hits, "misses": cache.misses, "size": cache.currsize},
                    "segments": len(result["segments"]), "diagnosticsCompleted": diagnostics.completed,
                })
                outputs.append(result)
        temp_files = [str(path.relative_to(temp_directory)) for path in temp_directory.rglob("*") if path.is_file()]

    first_rss = runs[0]["workingSetBytes"]
    second_rss = runs[1]["workingSetBytes"]
    memory_growth = second_rss - first_rss if isinstance(first_rss, int) and isinstance(second_rss, int) else None
    structure_equal = (
        outputs[0].keys() == outputs[1].keys()
        and [segment.keys() for segment in outputs[0]["segments"]]
        == [segment.keys() for segment in outputs[1]["segments"]]
    )
    checks = {
        "run1": runs[0]["status"] == "PASS",
        "run2": runs[1]["status"] == "PASS",
        "modelInitializedOnce": runs[1]["cache"] == {"hits": 1, "misses": 1, "size": 1},
        "modelFilesUnchanged": model_before == _tree_state(model_directory),
        "audioUnchanged": audio_before == (audio.stat().st_size, audio.stat().st_mtime_ns),
        "noTemporaryFilesCreated": not temp_files,
        "outputStructureEqual": structure_equal,
        "outputEqual": outputs[0] == outputs[1],
        "secondRunMemoryGrowthWithinLimit": (
            memory_growth <= MEMORY_GROWTH_LIMIT if memory_growth is not None else os.name != "nt"
        ),
        "networkBlockedDuringRuns": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "audio": _display_path(audio), "modelDirectory": _display_path(model_directory), "runs": runs,
        "checks": checks, "memoryGrowthBytesRun2VsRun1": memory_growth,
        "memoryGrowthLimitBytes": MEMORY_GROWTH_LIMIT, "temporaryFiles": temp_files,
        "note": "Two-run RSS check detects large growth but cannot prove absence of every long-run native leak.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_stability_probe(args.audio)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
