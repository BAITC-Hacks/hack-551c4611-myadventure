"""Reproducible public-fixture pipeline validation. No model download or cloud call."""

import argparse
import hashlib
import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .pipeline import PipelineDiagnostics, PipelineError, processMeetingAudio

FIXTURES = Path(__file__).parent / "tests" / "fixtures"


@dataclass(frozen=True)
class FixtureCase:
    name: str
    filename: str
    keywords: tuple[str, ...] = ()
    min_speakers: int = 1
    invalid: bool = False
    first_source: str | None = None


CASES = (
    FixtureCase("RU", "ru.wav", ("ночного", "добычу")),
    FixtureCase("KZ", "kk.wav", ("ғалымдар", "арқылы")),
    FixtureCase("MIXED", "mixed.wav", ("ночного", "ғалымдар"), first_source="ru.wav"),
    FixtureCase("MULTI SPEAKER", "multi-speaker.wav", ("ночного", "ғалымдар"), min_speakers=2, first_source="kk.wav"),
    FixtureCase("INVALID FILE", "invalid.wav", invalid=True),
)


def build_derived_fixtures(directory: Path = FIXTURES) -> None:
    """Rebuild deterministic montages, retaining attribution of source recordings."""
    for filename, sources in [("mixed.wav", ("ru.wav", "kk.wav")), ("multi-speaker.wav", ("kk.wav", "ru.wav"))]:
        parts: list[bytes] = []
        for source in sources:
            with wave.open(str(directory / source), "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                    raise ValueError("Source fixtures must be mono PCM16 at 16 kHz")
                parts.append(audio.readframes(audio.getnframes()))
        with wave.open(str(directory / filename), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(parts[0] + b"\0\0" * 16000 + parts[1])
    (directory / "invalid.wav").write_bytes(b"This is a synthetic non-audio fixture, not a WAV recording.\n")


def validate_contract(payload: object) -> None:
    script = """
import {readFileSync} from 'node:fs';
import Ajv from 'ajv';
const validate = new Ajv({strict:true}).compile(JSON.parse(readFileSync('schema.json','utf8')));
if (!validate(JSON.parse(readFileSync(0,'utf8')))) throw Error(JSON.stringify(validate.errors));
"""
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parent.parent / "contracts" / "speech",
        input=json.dumps(payload, ensure_ascii=True, allow_nan=False),
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=True,
    )


def run_fixture_suite(mode: Literal["local", "demo"], directory: Path = FIXTURES) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for case in CASES:
        diagnostics = PipelineDiagnostics()
        path = directory / case.filename
        row: dict[str, object] = {"case": case.name, "file": case.filename, "status": "FAIL"}
        try:
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            demo_turn_seconds = 5.0
            if mode == "demo" and case.first_source is not None:
                with wave.open(str(directory / case.first_source), "rb") as source:
                    demo_turn_seconds = source.getnframes() / source.getframerate() + 1.0
                row["demo_turn_seconds"] = demo_turn_seconds
                row["demo_timing_source"] = "Known montage boundary; not inferred from voices"
            result = processMeetingAudio(
                path, mode=mode, diagnostics=diagnostics,
                demo_speakers=2 if mode == "demo" and case.name in ("MIXED", "MULTI SPEAKER") else 1,
                demo_turn_seconds=demo_turn_seconds,
            )
            row["result"] = result
            if case.invalid:
                raise ValueError("Invalid fixture was unexpectedly accepted")
            validate_contract(result)
            segments = result["segments"]
            if result["durationSeconds"] < 0 or result["detectedSpeakers"] < case.min_speakers or not segments:
                raise ValueError("Duration, transcript or assigned speaker count check failed")
            if len({s["id"] for s in segments}) != len(segments):
                raise ValueError("Segment IDs must be unique")
            if [s["start"] for s in segments] != sorted(s["start"] for s in segments):
                raise ValueError("Timestamps must be sorted")
            for segment in segments:
                if not segment["id"] or not segment["speakerId"] or not 0 <= segment["start"] <= segment["end"] <= result["durationSeconds"]:
                    raise ValueError("Invalid segment identity or timestamps")
            text = " ".join(s["text"] for s in segments).casefold()
            if not all(keyword in text for keyword in case.keywords):
                raise ValueError("Expected source-language words were not retained")
            row.update(status="PASS", result=result)
        except PipelineError as error:
            row.update(error_stage=error.stage, error_code=error.code, message=str(error))
            if case.invalid and error.stage == "preprocessing" and error.code == "UNSUPPORTED_FORMAT":
                row["status"] = "PASS"
        except Exception as error:
            row["message"] = str(error)
        row.update(diarization_mode=diagnostics.mode, warnings=diagnostics.warnings)
        results.append(row)
        print(f"{case.name}: {row['status']}", flush=True)
    return {
        "scope": "Real local STT + DEMO speaker assignment" if mode == "demo" else "Local STT and local diarization",
        "speaker_separation_ai_evaluated": mode == "local" and all(row["status"] == "PASS" for row in results),
        "note": "DEMO PASS validates pipeline execution and assigned labels, not AI speaker recognition. Local-model tests are not a DER benchmark.",
        "results": results,
        "passed": all(row["status"] == "PASS" for row in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Rebuild montages and invalid fixture only")
    parser.add_argument("--mode", choices=("local", "demo"), default="local")
    parser.add_argument("--report", type=Path, default=FIXTURES / "results.json")
    args = parser.parse_args()
    if args.build:
        build_derived_fixtures()
        return 0
    report = run_fixture_suite(cast(Literal["local", "demo"], args.mode))
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
