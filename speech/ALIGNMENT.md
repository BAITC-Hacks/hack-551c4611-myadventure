# Deterministic speaker/transcript alignment

```python
from speech import alignTranscript, diarizeAudio, prepareAudio, transcribeAudio

audio = prepareAudio("meeting.wav")
transcript = transcribeAudio(audio)
speakers = diarizeAudio(audio)  # requires a provisioned local diarization model
aligned = alignTranscript(transcript, speakers, duration_seconds=audio.duration_seconds)

public_result = aligned.result
protocol_input = public_result["segments"]
# Retain aligned.diarization_mode and aligned.warnings in internal job state.
```

For an explicitly marked simulation, replace the diarization call with
`diarizeAudio(audio, mode="demo", demo_speakers=2)`. Alignment retains the `DEMO`
mode and warning even if the incoming demo warning is missing. Do not present
that output as AI speaker detection. The public contract has no provenance field,
so job state must retain the internal envelope while consumers receive `.result`.

Matching is deliberately simple: each complete STT segment selects the speaker
interval with the largest timestamp overlap. It is not split, translated, trimmed
or rewritten. Repeated intervals from the same speaker are compared individually,
not summed. For equal overlap, choose midpoint containment, then shortest distance
from midpoint to interval, then earlier start/end and lexical speaker ID.
Speaker intervals are half-open for containment (`start <= midpoint < end`),
so a zero-duration STT segment exactly at a speaker change selects the next turn.
Overlapping speaker tracks still yield one speaker per transcript segment.

If no interval overlaps, choose the nearest interval and attach an internal
warning. If diarization is empty, assign `SPEAKER_UNKNOWN` and warn. This is an
unassigned sentinel permitted by the existing string-valued schema, not a newly
detected voice or a change to the contract. No fictitious `SPEAKER_00` is inferred.
`detectedSpeakers` counts unique known IDs assigned to transcript segments,
excluding `SPEAKER_UNKNOWN`. Empty transcripts produce `segments: []` and zero
assigned speakers, preserving the supplied duration.

STT segments are sorted by start/end (original order breaks identical timestamp
ties), then assigned unique `seg-1`, `seg-2`, ... IDs. These IDs are scoped to one
result and deterministic for the same input. Input collections are not modified.
Runtime validation rejects invalid duration, non-string text, empty/unknown source
speaker labels, nonfinite/negative timestamps, reversed intervals and intervals
outside the recording. Failures use `AlignmentError` with `INVALID_DURATION`,
`INVALID_TRANSCRIPT` or `INVALID_DIARIZATION`.

The Python TypedDicts mirror the existing TypeScript contract. `.result` contains
only `durationSeconds`, `detectedSpeakers`, and `segments`; segment fields are
`id`, `speakerId`, `start`, `end`, and `text`, plus optional `language` from the
lightweight text heuristic. Speaker names are not inferred. No runtime model or
network dependency is introduced. Complexity is O(transcript segments × speaker
intervals), suitable for the initial MVP.

## Verification

The schema check uses the existing Ajv installation and reads the original
`contracts/speech/schema.json`; it does not create a second JSON Schema.
Install contract development dependencies once with:

```sh
npm ci --prefix contracts/speech
```

From the repository root:

```sh
python -m unittest discover -s speech/tests -p test_alignment.py -v
python -m ruff check --config speech/pyproject.toml speech
python -m mypy --config-file speech/pyproject.toml speech
python -m compileall -q speech
python -c "from speech import alignTranscript, AlignmentResult"
```

Fifteen tests cover the five requested cases, tie handling, nearest-interval gaps,
duplicate timestamps/unique IDs, sorted output, repeated speaker identities,
overlapping speakers, unchanged source text/input, malformed data, DEMO provenance
and validation of six representative outputs against the unchanged shared schema.

Stage 6 verification: all 15 alignment tests passed. Full Python discovery ran
70 tests: 63 passed and seven opt-in real-STT tests were skipped because
`JINALYS_STT_MODEL_DIR` was not configured for this run. All three existing
TypeScript contract tests passed. Ruff, strict mypy (11 Python files), ESLint,
TypeScript typecheck, Python compilation and import checks passed. This stage
does not claim a new real-model accuracy evaluation.
