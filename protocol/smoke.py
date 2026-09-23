"""Explicit real-model check on synthetic data; never run by unit tests."""
import json
import sys
import time
from pathlib import Path

from .pipeline import generate_meeting_protocol
from .llm import LocalLLM


class DiagnosticLLM(LocalLLM):
    def complete(self, system, data, schema):
        result = super().complete(system, data, schema)
        if "--diagnostic" in sys.argv:
            print(json.dumps(result, ensure_ascii=True), file=sys.stderr, flush=True)
        return result


def main():
    data = json.loads((Path(__file__).parent / "fixtures/input.demo.json").read_text(encoding="utf-8"))
    if "--long" in sys.argv:
        filler = [{"id": f"context-{i}", "speakerId": "SPEAKER_01", "start": i * 28,
                   "end": (i + 1) * 28, "language": "ru",
                   "text": "Обсуждаем текущую ситуацию. Новых поручений и решений в этой реплике нет. " * 10}
                  for i in range(60)]
        for segment in data["transcript"]:
            segment["start"] += 1750
            segment["end"] += 1750
        data["transcript"][-1]["end"] = 1800
        data["transcript"] = filler + data["transcript"]
    start = time.monotonic()
    result = generate_meeting_protocol(**data, llm=DiagnosticLLM(), on_progress=lambda stage: print(stage, file=sys.stderr, flush=True))
    by_source = {}
    for action in result["actionItems"]:
        for source in action["sourceSegmentIds"]:
            by_source.setdefault(source, []).append(action)
    failures = []
    for source, name, deadline in [("seg-2", "Ерболат", "2026-10-15"), ("seg-4", "Салтанат", None),
                                    ("seg-5", "Асқар", None), ("seg-6", "Айгүл", None), ("seg-7", None, "2026-10-16")]:
        candidates = by_source.get(source, [])
        if not any(a["assignee"] == name and a["deadline"] == deadline for a in candidates):
            failures.append(f"Missing or incorrect action from {source}")
    if "seg-3" in by_source:
        failures.append("Idea incorrectly treated as assignment")
    if not any(a["deadlineText"] == "осы аптада" for a in by_source.get("seg-5", [])):
        failures.append("Mixed-language deadline text was lost")
    if not any("хаттамасын" in a["task"] for a in by_source.get("seg-6", [])):
        failures.append("Kazakh task meaning was not preserved")
    if not result["summary"].strip():
        failures.append("Summary is empty")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if result["transcript"] != data["transcript"]:
        failures.append("Transcript changed or truncated")
    if "--long" in sys.argv:
        result = {"actionItems": result["actionItems"], "segmentCount": len(result["transcript"]),
                  "inputCharacters": sum(len(s["text"]) for s in data["transcript"])}
    print(json.dumps({"seconds": round(time.monotonic() - start, 2), "failures": failures, "result": result}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
