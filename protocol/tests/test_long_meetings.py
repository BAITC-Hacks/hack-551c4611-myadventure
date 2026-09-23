import json
import unittest

from protocol.pipeline import generate_meeting_protocol, transcript_chunks, CHUNK_CHARS
from protocol.validation import ProtocolError


class BoundaryLLM:
    def __init__(self):
        self.seen = set()
        self.calls = 0

    def complete(self, system, data, schema):
        self.calls += 1
        segments = data["transcript"]
        self.seen.update(s["id"] for s in segments)
        if "actionItems" in schema["properties"]:
            return {"actionItems": [dict(id="candidate", task="Check report", assignee=None,
                    deadlineText=None, deadline=None, sourceSegmentIds=[s["id"]],
                    confidence=0.8, needsReview=True) for s in segments if s["text"].startswith("TASK")]}
        if "isAction" in schema["properties"]:
            return {key: True for key in schema["properties"]}
        return {"selectedSegmentIds": [segments[-1]["id"]], "topics": []}


def meeting(count=120, length=700):
    return [dict(id=f"s-{i}", speakerId="SPEAKER_00", start=i * 1800 / count,
                 end=(i + 1) * 1800 / count, text="TASK " + "а" * length) for i in range(count)]


class LongMeetingTests(unittest.TestCase):
    def test_thirty_minutes_no_loss_and_overlap_deduplicated(self):
        transcript = meeting()
        llm = BoundaryLLM()
        progress = []
        result = generate_meeting_protocol(transcript, llm=llm, on_progress=progress.append)
        self.assertEqual(result["transcript"], transcript)
        self.assertEqual(llm.seen, {s["id"] for s in transcript})
        self.assertEqual(len(result["actionItems"]), len(transcript))
        self.assertEqual(progress.count("done"), 1)
        self.assertTrue(any(s.startswith("chunk:") for s in progress))

    def test_serialized_chunk_budget(self):
        chunks = transcript_chunks(meeting())
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(json.dumps(chunk, ensure_ascii=False)), CHUNK_CHARS)
        self.assertTrue(set(s["id"] for s in chunks[0]) & set(s["id"] for s in chunks[1]))

    def test_limits_fail_before_inference(self):
        for data in (meeting(length=1100), [dict(id="long", speakerId="S", start=0, end=1801, text="test")],
                     [dict(id="large", speakerId="S", start=0, end=1800, text="a" * 6100)]):
            llm = BoundaryLLM()
            with self.subTest(data_size=len(data)), self.assertRaises(ProtocolError):
                generate_meeting_protocol(data, llm=llm)
            self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
