import json
import unittest
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from protocol.deadlines import normalize_deadline
from protocol.llm import LocalLLM
from protocol.pipeline import generate_meeting_protocol, VERDICT
from protocol.validation import ProtocolError, CONTRACT, validate_input
from jsonschema import Draft202012Validator

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures/input.demo.json").read_text(encoding="utf-8"))


def action(segment="seg-2", **overrides):
    result = dict(id="candidate", task="Подготовить отчёт", assignee="Ерболат",
                  deadlineText="до 15 октября 2026 года", deadline=None,
                  sourceSegmentIds=[segment], confidence=0.9, needsReview=False)
    result.update(overrides)
    return result


class ScriptedLLM:
    """Test double only, not a substitute for real AI quality evaluation."""
    def __init__(self, actions, verdicts=None):
        good = {key: True for key in VERDICT["properties"]}
        self.responses = [{"actionItems": actions}, *(verdicts if verdicts is not None else [good] * len(actions)),
                          {"selectedSegmentIds": ["seg-1"], "topics": []}]

    def complete(self, system, data, schema):
        return self.responses.pop(0)


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, actions, verdicts=None):
        return generate_meeting_protocol(**FIXTURE, llm=ScriptedLLM(actions, verdicts))

    def test_schema_and_fixture(self):
        Draft202012Validator.check_schema(CONTRACT)
        validate_input(**FIXTURE)

    def test_complete_action_and_original_preserved(self):
        before = deepcopy(FIXTURE)
        result = self.run_pipeline([action()])
        self.assertEqual(result["actionItems"][0]["deadline"], "2026-10-15")
        self.assertFalse(result["actionItems"][0]["needsReview"])
        self.assertEqual(FIXTURE, before)
        self.assertEqual(result["transcript"], FIXTURE["transcript"])

    def test_missing_deadline(self):
        item = self.run_pipeline([action("seg-4", deadlineText=None)])["actionItems"][0]
        self.assertIsNone(item["deadline"])
        self.assertTrue(item["needsReview"])

    def test_missing_assignee(self):
        item = self.run_pipeline([action(assignee=None)])["actionItems"][0]
        self.assertIsNone(item["assignee"])
        self.assertTrue(item["needsReview"])

    def test_idea_rejected(self):
        verdict = {key: True for key in VERDICT["properties"]}
        verdict["isAction"] = False
        self.assertEqual(self.run_pipeline([action("seg-3")], [verdict])["actionItems"], [])

    def test_mixed_original_deadline_preserved(self):
        item = self.run_pipeline([action("seg-5", assignee="Асқар", deadlineText="осы аптада")])["actionItems"][0]
        self.assertEqual(item["deadlineText"], "осы аптада")
        self.assertIsNone(item["deadline"])

    def test_hallucinated_deadline_removed(self):
        item = self.run_pipeline([action(deadlineText="завтра", deadline="2030-01-01")])["actionItems"][0]
        self.assertIsNone(item["deadlineText"])
        self.assertIsNone(item["deadline"])

    def test_bad_source_fails(self):
        with self.assertRaises(ProtocolError):
            self.run_pipeline([action("nonexistent")])

    def test_duplicate_input_fails(self):
        with self.assertRaises(ProtocolError):
            validate_input([FIXTURE["transcript"][0]] * 2, {})

    def test_nonfinite_timestamp_rejected(self):
        data = deepcopy(FIXTURE)
        data["transcript"][0]["start"] = float("nan")
        with self.assertRaises(ProtocolError):
            validate_input(**data)

    def test_no_actions(self):
        self.assertEqual(self.run_pipeline([])["actionItems"], [])

    def test_summary_is_grounded(self):
        result = self.run_pipeline([])
        self.assertEqual(result["summary"], FIXTURE["transcript"][0]["text"])

    def test_summary_unknown_source_rejected(self):
        llm = ScriptedLLM([])
        llm.responses[-1]["selectedSegmentIds"] = ["missing"]
        with self.assertRaises(ProtocolError):
            generate_meeting_protocol(**FIXTURE, llm=llm)

    def test_malformed_ai_fails(self):
        with self.assertRaises(ProtocolError):
            self.run_pipeline([{"task": "missing fields"}])

    def test_unconfirmed_assignee_removed(self):
        verdict = {key: True for key in VERDICT["properties"]}
        verdict["assigneeSupported"] = False
        item = self.run_pipeline([action()], [verdict])["actionItems"][0]
        self.assertIsNone(item["assignee"])
        self.assertLessEqual(item["confidence"], 0.5)


class DeadlineTests(unittest.TestCase):
    def test_dates(self):
        for text, expected in [("до 2026-10-15", "2026-10-15"), ("15.10.2026", "2026-10-15"),
                               ("до 15 октября 2026 года", "2026-10-15"), ("31 февраля 2026", None),
                               ("до 15 октября", None), ("до пятницы", None), (None, None)]:
            with self.subTest(text=text):
                self.assertEqual(normalize_deadline(text), expected)

    def test_relative(self):
        self.assertEqual(normalize_deadline("через две недели", "2026-09-23"), "2026-10-07")
        self.assertEqual(normalize_deadline("ертең", "2026-09-23"), "2026-09-24")
        self.assertIsNone(normalize_deadline("через две недели"))


class AdapterTests(unittest.TestCase):
    def test_cloud_rejected(self):
        with self.assertRaises(ProtocolError):
            LocalLLM(model="test", base_url="https://example.com")

    def test_unavailable(self):
        adapter = LocalLLM(model="test")
        with patch.object(adapter.opener, "open", side_effect=URLError("offline")):
            with self.assertRaisesRegex(ProtocolError, "unavailable"):
                adapter.complete("test", {}, {})

    def test_structured_http_response(self):
        adapter = LocalLLM(model="test")
        body = json.dumps({"done": True, "message": {"content": '{"ok": true}'}}).encode()
        with patch.object(adapter.opener, "open", return_value=BytesIO(body)) as request:
            self.assertEqual(adapter.complete("system", {}, {"type": "object"}), {"ok": True})
            payload = json.loads(request.call_args.args[0].data)
            self.assertFalse(payload["stream"])
            self.assertFalse(payload["think"])

    def test_malformed_http_response(self):
        adapter = LocalLLM(model="test")
        with patch.object(adapter.opener, "open", return_value=BytesIO(b'not json')):
            with self.assertRaisesRegex(ProtocolError, "malformed"):
                adapter.complete("test", {}, {})

    def test_cloud_model_rejected(self):
        with self.assertRaises(ProtocolError):
            LocalLLM(model="example:cloud")


if __name__ == "__main__":
    unittest.main()
