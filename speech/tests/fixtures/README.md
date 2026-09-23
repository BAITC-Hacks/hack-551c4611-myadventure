# Audio attribution

Source: Google / FLEURS contributors, [FLEURS dataset](https://huggingface.co/datasets/google/fleurs),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No endorsement implied.

- `ru.wav`: `ru_ru`, dev split, original `10005687533826592442.wav`, 13.68 seconds.
- `kk.wav`: `kk_kz`, dev split, original `1003567887281500142.wav`, 12.30 seconds.

Source TSV metadata marks the RU recording `FEMALE` and KZ recording `MALE`:
[RU dev.tsv](https://huggingface.co/datasets/google/fleurs/blob/main/data/ru_ru/dev.tsv),
[KZ dev.tsv](https://huggingface.co/datasets/google/fleurs/blob/main/data/kk_kz/dev.tsv).
These independent voices provide the two-speaker fixture; no real identity is used.

Downloaded 2026-09-23 from `data/{language}/audio/dev.tar.gz`; matching reference
transcripts are in `data/{language}/dev.tsv` in the source dataset.
Changes: decoded with PyAV, converted to mono 16 kHz signed PCM16 WAV, renamed.
Speech content was not edited. These are public recordings, not meeting audio.

The integration test constructs mixed input from RU + one second of silence + KZ
and deletes it afterward. It tests sequential language switching, not naturally
occurring switching inside a sentence. Keyword assertions confirm retained source
languages; they are not a WER benchmark or a guarantee of transcription accuracy.

## Committed Stage 9 fixtures

| File | Content | Synthetic? |
| --- | --- | --- |
| `ru.wav` | Original public RU speech, female voice | No; format conversion only |
| `kk.wav` | Original public KZ speech, male voice | No; format conversion only |
| `mixed.wav` | RU + 1 s silence + KZ, 26.98 s | Synthetic montage of human recordings |
| `multi-speaker.wav` | KZ male + 1 s silence + RU female, 26.98 s | Synthetic two-speaker montage |
| `invalid.wav` | Plain text masquerading as WAV | Synthetic negative fixture |

Both montages inherit the above CC BY 4.0 attribution. No voice cloning or cloud
TTS was used. Neither montage represents natural conversational code-switching.
All five audio/negative fixtures together occupy less than 3 MiB. No meeting data,
secrets or model weights are included.

Rebuild only derived fixtures, using the two committed source WAVs:

```sh
python -m speech.fixture_suite --build
```

Run every fixture through the public entry point with local STT weights configured
in `JINALYS_STT_MODEL_DIR` and contract test dependencies installed:

```sh
python -m speech.fixture_suite --mode demo
```

The command writes `results.json` and exits nonzero if any case fails. It verifies
source-language keywords, existing JSON Schema, timestamp/ID invariants and the
expected controlled rejection of invalid input. The multi-speaker case requires
at least two assigned IDs. The report includes file hashes, full public-fixture
transcripts and explicit DEMO provenance. In this mode the count/labels are
simulated, not evidence of AI identifying two voices. A PASS applies to pipeline
execution, not diarization quality. Use `--mode local` with an authorized complete
diarization bundle to evaluate the actual local-model pipeline.

For both montages, the runner sets the deterministic DEMO turn duration to the
first source's duration plus the one-second gap, using the known construction
boundary. The initial default five-second demo grid assigned both voices to one
ID in the reversed montage; explicit fixture timing fixes that simulation issue.
No model prediction or speaker count is overwritten to make a check pass.
