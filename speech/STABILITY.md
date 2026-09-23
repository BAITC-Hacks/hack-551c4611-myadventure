# Stage 10 local inference stability

Validation date: 2026-09-23. Environment: Windows, CPU int8,
`faster-whisper==1.2.1`, local `large-v3-turbo` CTranslate2 weights. Input:
the attributed 13.68-second RU fixture. Speaker assignment used the explicitly
marked deterministic DEMO mode because local diarization weights are unavailable.
STT was real local inference.

The probe runs `processMeetingAudio` twice in one fresh Python process after
clearing the model cache. It blocks Python socket connections, redirects `TMP`
and `TEMP` to an isolated directory, validates both responses against the shared
JSON Schema, and compares model/audio file state before and after.

| Check | RUN 1 | RUN 2 |
| --- | ---: | ---: |
| Result | PASS | PASS |
| Elapsed | 17.637 s | 14.490 s |
| STT cache | 1 miss, 0 hits | 1 miss, 1 hit |
| Cached models | 1 | 1 |
| Segments | 2 | 2 |
| Working set | 959,438,848 B | 956,514,304 B |

PASS findings:

- The STT model initialized once. The second request reused the same cached model.
- No network connection was attempted; a download would have failed the run.
- Local model files and source audio size/mtime were unchanged.
- Both public responses were exactly equal and had the same field structure.
- No files appeared in the isolated temporary directory.
- Working set decreased by 2,924,544 bytes from RUN 1 to RUN 2.
- Reused `PipelineDiagnostics` state is reset per request in unit coverage.
- Missing or incomplete model configuration remains a controlled
  `PipelineError(stage="stt", code="MODEL_MISSING")` with an actionable message.
- Explicit DEMO does not construct the optional pyannote backend.

The model cache is the existing `functools.lru_cache(maxsize=1)` around local
Whisper model loading. A single configured deployment model remains resident for
reuse. Selecting a different path/device/compute tuple may evict the previous
entry, as expected for this MVP worker architecture.

The RSS check guards against a large second-run increase (limit 256 MiB), and the
measured second run did not grow. Two runs cannot prove the absence of every
long-duration leak in native libraries. Production soak testing over long meeting
audio and many requests remains appropriate before deployment.

Reproduce after provisioning the ignored local model directory:

```sh
set JINALYS_STT_MODEL_DIR=speech/models/large-v3-turbo
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
python -m speech.stability_probe --report speech/tests/fixtures/stability-results.json
```

The committed JSON report contains the complete machine-readable checks. Model
weights remain ignored and are not part of this commit.
