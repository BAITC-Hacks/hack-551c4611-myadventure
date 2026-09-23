# Статус JINALYS AI — 23 сентября 2026

## Объединение для main

По запросу участника №3 объединены `codex/job-pipeline-integration` (исходный head `e39b752f225f4bffae8cd54691979523d4e3691a`) и обновлённая `codex/protocol-ai` (`88a51dc51809efe61c03942b8eb0ef536503c49a`). Новый Protocol AI сохраняет контракт и добавляет проверки имён, сроков и формулировок задач; подробности в [PROTOCOL_QUALITY_UPDATE.md](PROTOCOL_QUALITY_UPDATE.md).

Текстовых конфликтов в файлах Protocol AI нет: изменяемые файлы в интеграционной ветке совпадали с общей исходной версией. Сохранены UI, очередь заданий, DOCX и обновление FFmpeg участника №1. Проверки объединения: 31 тест Protocol AI, 4 интеграционных и 5 frontend/DOCX тестов, production-сборка Next.js. Сквозная проверка с настоящей диаризацией остаётся незавершённой; публикация кода в main не означает её прохождения.

Далее сохранён исторический снимок GitHub до этого объединения; статусы PR и веток в нём относятся к предыдущей проверке.

## Что изменилось

В ветке `codex/speech-pipeline` появился коммит `c95a9c5451cdc32239d81717cfca12ed811e417b`:

- `speech/_diarization_worker.py`: поддержка `JINALYS_FFMPEG_DIR`. Каталог добавляется в PATH и через `os.add_dll_directory` до импорта pyannote; handle сохраняется до завершения inference.
- `speech/tests/test_diarization.py`: тест настройки каталога Windows DLL.
- `docs/handoff/speech.md`: установка отдельного Python environment, FFmpeg shared, доступ к community-1 и порядок запуска реального MP3.

Все три файла перенесены из указанного коммита без переписывания в интеграционную ветку. README, `.env.example` и `docs/LOCAL_RUN.md` дополнены настройкой FFmpeg и актуальным статусом проверок.

## Ветки и pull requests

- `main`: `c9e89a217cb0f5538e37eebaf8dcb0ab3d5858fb`; с предыдущей проверки не изменился. В него уже включены исходные UI, Protocol AI, long-meeting UI и Speech через PR №1–5.
- [PR №6](https://github.com/BAITC-Hacks/hack-551c4611-myadventure/pull/6): `codex/job-pipeline-integration` → `main`, draft. Содержит очередь заданий, реальный адаптер speech/protocol, UI polling, metadata, DOCX и инструкции. До текущего дополнения head был `b30fa3c0cd6169d7dac2a02e3579868e84643a9b`.
- [PR №7 «to main»](https://github.com/BAITC-Hacks/hack-551c4611-myadventure/pull/7): фактически `main` → `codex/speech-pipeline`. Это обратное направление относительно названия: merge этого PR обновляет speech-ветку, а не main. PR не изменён в рамках этой проверки.
- Новый speech-коммит ещё не был в main на момент проверки; учтён в PR №6.

## Что проверено локально

- До дополнения: Next build, 5 frontend/DOCX тестов, 25 Protocol AI тестов и 4 integration теста.
- Whisper large-v3-turbo: настоящая публичная RU-запись `speech/tests/fixtures/ru.wav` распознана, получен текст с timestamps. Это отдельная STT-проверка, без реального разделения спикеров.
- Ollama/qwen3:4b: `python -m protocol.smoke` завершён с кодом 0, `failures: []`, 75.2 секунды. Вход — синтетический текст с RU/KZ/mixed-поручениями, не аудио.
- Библиотека pyannote импортируется; веса community-1 и реальное разделение спикеров на этой машине ещё не проверены.

## Что остаётся

Подготовить community-1 и FFmpeg shared, проверить реальную цепочку MP3/WAV → speech → protocol → UI → DOCX, RU/KZ/mixed аудио, повторный запуск и около 30 минут с поручением в конце. Установку моделей пользователь временно отложил.

Ограничения AI сохраняются: контекст между частями — до двух реплик; глобальная проверка поздних отмен и далёких уточнений отсутствует, смысловые дубликаты возможны. Прототип job server хранит задания в памяти и не поддерживает backend-отмену.

