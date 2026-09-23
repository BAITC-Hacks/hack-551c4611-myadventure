# Audio attribution

Source: Google / FLEURS contributors, [FLEURS dataset](https://huggingface.co/datasets/google/fleurs),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No endorsement implied.

- `ru.wav`: `ru_ru`, dev split, original `10005687533826592442.wav`, 13.68 seconds.
- `kk.wav`: `kk_kz`, dev split, original `1003567887281500142.wav`, 12.30 seconds.

Downloaded 2026-09-23 from `data/{language}/audio/dev.tar.gz`; matching reference
transcripts are in `data/{language}/dev.tsv` in the source dataset.
Changes: decoded with PyAV, converted to mono 16 kHz signed PCM16 WAV, renamed.
Speech content was not edited. These are public recordings, not meeting audio.

The integration test constructs mixed input from RU + one second of silence + KZ
and deletes it afterward. It tests sequential language switching, not naturally
occurring switching inside a sentence. Keyword assertions confirm retained source
languages; they are not a WER benchmark or a guarantee of transcription accuracy.
