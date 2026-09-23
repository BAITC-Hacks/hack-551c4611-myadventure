# Local diarization adapter and explicit DEMO mode

`diarizeAudio(preparedAudio)` returns an internal envelope:

```json
{
  "mode": "LOCAL",
  "segments": [{"speakerId": "SPEAKER_00", "start": 0.0, "end": 1.0}],
  "warning": null
}
```

Each backend cluster is assigned an anonymous ID in order of first chronological
appearance. Repeated clusters retain their ID inside the recording; IDs do not
identify people or match speakers across recordings. Results are sorted, finite,
nonnegative and bounded by audio duration. Overlapping speakers remain overlapping.
An empty real result is allowed (for example silence). No identity recognition is
performed. The public transcript contract is unchanged; this is internal metadata
for the later speaker/transcript alignment stage.

## DEMO: available now, not AI speaker separation

```python
from speech import prepareAudio, diarizeAudio

result = diarizeAudio(
    prepareAudio("meeting.wav"),
    mode="demo", demo_speakers=2, demo_turn_seconds=5.0,
)
assert result["mode"] == "DEMO"
print(result["warning"])
```

This assigns alternating fixed-length time windows without analyzing voices or
detecting silence. `demo_speakers` is supplied by the caller, not inferred.
Short audio may contain fewer assigned speakers than requested. DEMO works on
very short WAVs, and simulated fast changes are controlled by `demo_turn_seconds`.
It must always be presented as **DEMO**, together with its warning; do not discard
the envelope and label these segments as AI results. Future transcript assembly
must retain this provenance internally rather than changing the public schema.
DEMO is never selected automatically after a model or inference failure.

## Real local implementation path

The optional backend targets
[pyannote community-1 offline inference](https://huggingface.co/pyannote/speaker-diarization-community-1#offline-use)
using `pyannote.audio==4.0.7`. It runs in a separate local Python process. The
adapter does not call any cloud diarization endpoint.

1. Obtain authorized access to the official community-1 repository and provision
   its complete offline bundle separately: `config.yaml` plus all actual model
   assets/subdirectories, not Git LFS pointer files. Provisioning may require
   accepting the model's access terms; runtime does not require an API key.
2. Place it in `speech/models/community-1` (ignored by Git), or another local
   directory. Set `JINALYS_DIARIZATION_MODEL_DIR` to that directory.
3. Install `speech/requirements-diarization.txt` in a dedicated environment. Set
   `JINALYS_DIARIZATION_PYTHON` to its Python executable if different from the
   application interpreter. Pyannote's file decoder may additionally require
   compatible TorchCodec/FFmpeg shared libraries. These optional prerequisites
   are not required for DEMO or the speech contract.
4. Call `diarizeAudio(prepareAudio("meeting.wav"))`. Validate real one-/two-speaker
   recordings before treating the backend as production-ready.

The official bundle determines device settings (normally CPU unless configured
otherwise). A known participant count can optionally be passed as a local model
hint; this does not identify real people:

```python
from speech import PyannoteLocalBackend, diarizeAudio, prepareAudio

backend = PyannoteLocalBackend(num_speakers=2, timeout_seconds=600)
result = diarizeAudio(prepareAudio("meeting.wav"), backend=backend)
```

`LocalDiarizationBackend` is the small protocol for alternative local
implementations. It returns `SpeakerTurn` values. Remote implementations and
demo backends pretending to be local AI are outside that interface's intended use.

The worker forces Hugging Face offline mode, disables implicit tokens, pyannote
metrics and OpenTelemetry, and calls `Pipeline.from_pretrained` with a local
directory and `token=False`. Missing dependencies/assets are controlled failures.
Settings are isolated in the child process and do not change the parent process.
The worker timeout terminates a stuck child. No temporary audio file is created.
For an operational deployment, also deny outbound network at the worker boundary.

Errors: `MODEL_MISSING`, `MODEL_LOADING_FAILED`, `INVALID_AUDIO`,
`INFERENCE_FAILED`, `INVALID_CONFIG`. Detailed backend errors are not passed
through as transcripts. There is no fallback on error unless the caller explicitly
starts a separate DEMO invocation.

## Stage 5 verification scope

On 2026-09-23, the environment had no diarization weights or pyannote installation;
anonymous access to the official model's `config.yaml` returned HTTP 401.
Real AI speaker-separation quality was therefore **not evaluated**. No gated
model was downloaded and no access terms were accepted on the user's behalf.

Twenty tests cover the permitted adapter/DEMO implementation:

- One speaker: real RU fixture through DEMO, at least `SPEAKER_00`.
- Two speakers: combined RU/KZ fixture, explicitly configured two-speaker DEMO.
- Short input: one PCM frame, controlled output.
- Rapid switching: 100 deterministic turns/second, stable three-speaker IDs.
- Unavailable model: controlled error by default; DEMO requires explicit mode.
- Local-backend doubles: stable remapping, overlaps, empty output, invalid values
  and lazy inference failures. These are not real-model evaluations.
- Real subprocess with unusable bundle: controlled loading failure.
- Worker API bridge using a test double: local path, no token, offline settings,
  error transport, timeout and malformed response handling.

Run from the repository root:

```sh
python -m unittest discover -s speech/tests -p test_diarization.py -v
python -m ruff check --config speech/pyproject.toml speech
python -m mypy --config-file speech/pyproject.toml speech
python -m compileall -q speech
python -c "from speech import diarizeAudio, DiarizationError, PyannoteLocalBackend"
```

The complete test suite uses `python -m unittest discover -s speech/tests -v`.
The pre-existing STT integration tests need `JINALYS_STT_MODEL_DIR`; their
successful inference does not imply that a diarization model is available.

Final verification: all 55 tests passed without skips (20 diarization adapter/DEMO
tests plus the previous audio/STT suite, including real STT inference). Ruff,
strict mypy for nine Python files, bytecode compilation and dependency-free import
also passed. This is PASS for the allowed adapter/DEMO scope, not a measured
real-diarization accuracy result.
