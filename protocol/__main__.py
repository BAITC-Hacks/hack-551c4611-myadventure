import argparse
import json
import sys
from pathlib import Path

from .pipeline import generate_meeting_protocol
from .validation import ProtocolError, validate_input


def main():
    parser = argparse.ArgumentParser(description="Local meeting protocol AI; JSON on stdout, status on stderr")
    parser.add_argument("input", type=Path, help="JSON with transcript and metadata")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8-sig"))
        from .validation import validate, schema_for
        validate(data, schema_for("ProtocolInput"))
        validate_input(data["transcript"], data["metadata"])
        if args.validate_only:
            print("Input valid", file=sys.stderr)
            return 0
        result = generate_meeting_protocol(**data, on_progress=lambda stage: print(stage, file=sys.stderr))
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ProtocolError, OSError, ValueError) as error:
        print(f"Protocol error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
