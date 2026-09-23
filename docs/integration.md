Подробный гайд для участника №2: [TEAM_HANDOFF.md](TEAM_HANDOFF.md).

# Интеграция участников №1/№2/№3

Единый контракт: lib/contracts.ts. Перед изменением типов согласуйте их всей командой.

## HTTP
POST /api/meetings, multipart/form-data, поле audio: File.
Успех: HTTP 200, тело — объект MeetingProtocol напрямую (без обёртки result).
Ошибка: ненулевой HTTP error status, {error:{code:string,message:string}}.
До подключения route возвращает 503 PIPELINE_NOT_CONFIGURED.

## №1 → №2
TranscriptSegment[]: id, speakerId, speakerName?, start/end в секундах, text, language?: ru|kk|mixed.
Diarization определяет спикеров, а не автоматически устанавливает реальные личности.

## №2 → №3
MeetingProtocol: title, date? (YYYY-MM-DD), transcript, summary, topics:[{title,summary}], actionItems.
ActionItem: id, task, assignee:string|null, deadlineText:string|null, deadline:string|null (ISO date), assignedBy?:string|null, sourceSegmentIds:string[], confidence:0..1, needsReview:boolean.
sourceSegmentIds должны ссылаться на transcript. Неизвестное остаётся null; не выдумывать даты и имена. Для относительных сроков AI должен получать дату и timezone жиналыса из согласованных metadata — UI ввода metadata ещё предстоит добавить при интеграции.

## Порядок подключения
1. №1 и №2 разрабатывают свои модули независимо на fixtures.
2. №2 как основной интегратор заменяет заглушку route адаптером: локальный speech → локальный protocol → protocolSchema.parse → JSON.
3. Backend проверяет размер, реальный формат, длительность аудио; timeout, очистку временных файлов, ошибки модели.
4. Проверить RU, KZ и mixed записи, отсутствие срока/ответственного, повторную обработку и экспорт.
5. Добавить фактические команды моделей и env в README/.env.example.

UI сейчас ждёт один HTTP ответ; поэтапные статусы не симулируются. Для реальных статусов согласовать отдельный job/SSE контракт. Отмена на клиенте не гарантирует остановку локального AI: backend должен поддержать отмену самостоятельно.

