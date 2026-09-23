# Public meeting transcription entry point

```python
from speech import PipelineError, processMeetingAudio

try:
    result = processMeetingAudio("meeting.wav")
    # Exact JSON-compatible SpeechPipelineResult:
    # {durationSeconds, detectedSpeakers, segments}
    protocol_input = result["segments"]
except PipelineError as error:
    print(error.stage, error.code, str(error))
```

The caller supplies a file, not a model. The speech worker's deployment config
selects local model directories through `JINALYS_STT_MODEL_DIR` and
`JINALYS_DIARIZATION_MODEL_DIR`; see `.env.example`. Provision dependencies and
weights using the STT and diarization setup documentation. `.env` files are not
automatically loaded. No service, cloud API or API key is introduced.

The entry point runs preprocessing, local STT, diarization, deterministic
alignment and optional text language marking. It returns only the existing public
contract fields. Identifiers are unique within the result and timestamps are
sorted. Original text is preserved. Silence/empty transcripts may produce an
empty list and zero detected speakers. An absent diarization model is an error,
not an automatic simulation. Input remains owned by the caller and is not deleted.

`PipelineError.stage` identifies `configuration`, `preprocessing`, `stt`,
`diarization` or `alignment`. Known errors preserve their underlying stable code,
such as `INVALID_FILE`, `MODEL_MISSING`, `MODEL_LOADING_FAILED` or
`INFERENCE_FAILED`. Unexpected ordinary exceptions are wrapped as `STAGE_FAILED`
with a generic public message and chained cause for internal debugging. Failed
stages stop execution; no partial public result is returned. Process termination,
interrupts and native crashes are not recoverable Python exceptions.

## Explicit DEMO diarization

Real diarization weights are currently unavailable. A demo can exercise the whole
orchestration with real local STT, but speaker turns remain simulated:

```python
from speech import PipelineDiagnostics, processMeetingAudio

diagnostics = PipelineDiagnostics()
result = processMeetingAudio(
    "meeting.wav", mode="demo", demo_speakers=2, diagnostics=diagnostics,
)
assert diagnostics.mode == "DEMO"
assert diagnostics.completed
# Store diagnostics with the job and visibly mark it DEMO.
```

DEMO requires the diagnostics holder; otherwise the entry point raises
`DEMO_REQUIRES_DIAGNOSTICS`. This keeps provenance outside the unchanged public
schema while making it available to the integration layer. Do not present the
simulated speaker count/labels as detected voices. Warnings also retain nearest
speaker matching and unknown-speaker notices. Reused diagnostics are reset at
the start of each call and only marked completed on success. Use one holder per
concurrent job. Normal callers can optionally collect the same diagnostics.

For known synthetic fixture timing, `demo_turn_seconds` configures the simulated
turn duration (default 5 seconds). It is not inferred from audio. Non-default
demo options require explicit DEMO mode and do not affect local-model inference.

## End-to-end verification

`speech/tests/test_pipeline.py` includes actual WAV → preprocessing → local
Whisper → explicit DEMO diarization → alignment/language → schema-validated JSON.
The audio is the attributed 13.68-second RU fixture. No stage function is mocked
in that test; only outbound Python socket connections are blocked. DEMO is an
explicit deterministic implementation, not real AI diarization.

Set `JINALYS_STT_MODEL_DIR` to local weights and run from the repository root:

```sh
python -m unittest speech.tests.test_pipeline.PipelineEndToEndTests.test_real_audio_real_stt_explicit_demo_diarization -v
python -m unittest discover -s speech/tests -v
python -m ruff check --config speech/pyproject.toml speech
python -m mypy --config-file speech/pyproject.toml speech
python -m compileall -q speech
npm test --prefix contracts/speech
npm run lint --prefix contracts/speech
npm run typecheck --prefix contracts/speech
```

Schema tests require `npm ci --prefix contracts/speech`. Python dependencies are
documented in `README.md`. There is no separate application build script; Python
compilation/import and TypeScript no-emit compilation are the available checks.

The fully real-model E2E test is enabled additionally by
`JINALYS_DIARIZATION_MODEL_DIR`. It remains unverified until an authorized complete
diarization bundle and compatible dependencies are installed. A passing DEMO E2E
must not be reported as a passing fully model-based diarization pipeline.

Stage 8 validation (2026-09-23): the real-audio/real-STT/DEMO-diarization E2E test
passed alone and again in the full suite. Full discovery: 88 tests, 87 passed,
one skipped (fully real diarization E2E; local bundle unavailable). All three
contract tests, Ruff, strict mypy for 15 Python files, ESLint, TypeScript typecheck,
Python compilation and public import passed. The shared schema was not changed.
