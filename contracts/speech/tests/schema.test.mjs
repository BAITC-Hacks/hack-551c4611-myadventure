import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { URL } from "node:url";
import test from "node:test";
import Ajv from "ajv";
import { toSpeechPipelineResult } from "../index.ts";

const schema = JSON.parse(readFileSync(new URL("../schema.json", import.meta.url), "utf8"));
const ajv = new Ajv({ strict: true });
const validate = ajv.compile(schema);
const segment = { id: "s1", speakerId: "p1", start: 0, end: 1, text: "Сәлем, коллеги" };
const result = { durationSeconds: 1, detectedSpeakers: 1, segments: [segment] };

test("schema compiles and accepts the public examples", () => {
  assert.equal(validate(result), true);
  assert.equal(validate({ durationSeconds: 0, detectedSpeakers: 0, segments: [] }), true);
  for (const language of ["ru", "kk", "mixed"]) {
    assert.equal(validate({ ...result, segments: [{ ...segment, language, speakerName: "Алия" }] }), true);
  }
});

test("schema rejects missing fields, wrong types, languages and internal metadata", () => {
  for (const field of schema.required) {
    const invalid = { ...result };
    delete invalid[field];
    assert.equal(validate(invalid), false);
  }
  for (const field of schema.definitions.TranscriptSegment.required) {
    const invalid = { ...segment };
    delete invalid[field];
    assert.equal(validate({ ...result, segments: [invalid] }), false);
  }
  for (const patch of [{ language: "en" }, { start: "0" }, { speakerName: null }, { confidence: 0.9 }]) {
    assert.equal(validate({ ...result, segments: [{ ...segment, ...patch }] }), false);
  }
  assert.equal(validate({ ...result, model: "internal" }), false);
  assert.equal(validate({ ...result, durationSeconds: "1" }), false);
  assert.equal(validate({ ...result, detectedSpeakers: "1" }), false);
});

test("adapter strips internal fields at both levels without mutating input", () => {
  const internal = {
    ...result, model: "local-model",
    segments: [{ ...segment, language: "mixed", speakerName: "Алия", confidence: 0.9 }],
  };
  const output = toSpeechPipelineResult(internal);
  assert.deepEqual(output, {
    ...result, segments: [{ ...segment, language: "mixed", speakerName: "Алия" }],
  });
  assert.equal(validate(output), true);
  assert.equal(internal.segments[0].confidence, 0.9);
  assert.notEqual(output.segments[0], internal.segments[0]);
  assert.deepEqual(toSpeechPipelineResult(result), result);
});
