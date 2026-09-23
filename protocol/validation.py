import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


class ProtocolError(ValueError):
    """Safe, actionable error; never includes transcript content."""


CONTRACT = json.loads((Path(__file__).parent / "contracts/protocol.schema.json").read_text(encoding="utf-8"))


def schema_for(name):
    return {"$schema": CONTRACT["$schema"], "$defs": CONTRACT["$defs"], "$ref": f"#/$defs/{name}"}


def validate(data, schema):
    try:
        json.dumps(data, allow_nan=False)
    except (ValueError, TypeError):
        raise ProtocolError("Data must be JSON-compatible with finite numbers") from None
    error = next(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(data), None)
    if error:
        path = ".".join(map(str, error.absolute_path)) or "root"
        raise ProtocolError(f"Invalid data at {path}: {error.validator}")


def validate_input(transcript, metadata):
    validate({"transcript": transcript, "metadata": metadata}, schema_for("ProtocolInput"))
    ids = [s["id"] for s in transcript]
    if len(ids) != len(set(ids)):
        raise ProtocolError("Duplicate transcript IDs")
    if any(s["end"] < s["start"] for s in transcript):
        raise ProtocolError("Segment end precedes start")


def validate_output(result):
    validate(result, schema_for("MeetingProtocol"))
    validate_input(result["transcript"], {})
    ids = {s["id"] for s in result["transcript"]}
    actions = result["actionItems"]
    if len({a["id"] for a in actions}) != len(actions):
        raise ProtocolError("Duplicate action IDs")
    for action in actions:
        if not set(action["sourceSegmentIds"]) <= ids:
            raise ProtocolError("Action references missing source segments")
        if action["deadlineText"] is None and action["deadline"] is not None:
            raise ProtocolError("Deadline has no original text")
        if (action["assignee"] is None or action["deadline"] is None) and not action["needsReview"]:
            raise ProtocolError("Incomplete action must require review")
