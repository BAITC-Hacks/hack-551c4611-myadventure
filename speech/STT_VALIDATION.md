# Stage 4 validation — 2026-09-23

PASS for the scoped local STT MVP. No cloud inference or API key was used.

Environment: Windows, Python 3.13.6, faster-whisper 1.2.1, CTranslate2 4.8.2,
PyAV 18.1.0, CPU int8. The available GPU was not used.
Model: `dropbox-dash/faster-whisper-large-v3-turbo`, revision
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`.
`model.bin` SHA256:
`e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da`.
Weights are provisioned locally in `speech/models/large-v3-turbo` and ignored by
Git. Downloads were a separate setup step; inference used offline mode.

| Case | Input | Outcome |
| --- | --- | --- |
| RU | FLEURS dev, 13.68 s | Nonempty Russian text; expected source words retained |
| KZ | FLEURS dev, 12.30 s | Nonempty Kazakh text; source-language words retained, no Russian translation |
| Mixed | RU + 1 s silence + KZ, 26.98 s | Both source languages retained in ordered segments |
| Silence | 3 s PCM zeros | Empty segment list, no exception |
| Missing model | Nonexistent directory | Controlled `MODEL_MISSING` |
| Corrupt model | Invalid local weights/config | Controlled `MODEL_LOADING_FAILED` |
| Automatic language | RU and KZ without language hint | Both keyword checks passed |

All emitted timestamps passed `0 <= start <= end <= duration` and ordering checks.
The initial mixed run exposed an extra English phrase beyond the recorded speech.
Word alignment and silence-hallucination filtering removed it on rerun; an explicit
regression assertion covers that sample. Output timestamps are also bounded by
the source duration. This is not a general guarantee against hallucinations.

Validation completed: 28 unit tests and 7 real-model integration tests (34 tests
in the main run, then the added automatic-language test separately), no failures
or skips in these runs. The automatic-language test contains RU and KZ subcases.
Python socket connections were blocked in adapter and real-model tests. The model
loader uses local files only and requires a tokenizer before constructing the
backend. This socket guard is a test assertion, not an OS network sandbox.

Ruff, strict mypy (6 Python source files), bytecode compilation and import without
the optional inference dependencies passed. Tests create audio under temporary
directories with registered cleanup; production inference creates no temporary
audio files. Public transcript types/schema were not modified.

Scope limits: only one public recording per language was evaluated; Kazakh output
has recognition/spelling errors. The mixed sample is concatenated speech, not
natural switching within a sentence. No WER target, noisy-meeting benchmark,
speaker attribution or production accuracy claim is made at this stage.
See `tests/fixtures/README.md` for source attribution and reproduction details.
