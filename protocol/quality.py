"""Conservative safeguards independent of the model's own verdict."""
import re


def meaningful_task(text):
    words = re.findall(r"\w+", text.casefold())
    return bool(words) and ' '.join(words) not in {
        'сделать', 'сделаю', 'сделаем', 'проверить', 'проверьте', 'принято',
        'хорошо сделаю', 'понял сделаю', 'поняла сделаем', 'принято сделаем',
        'жду смету', 'не затягивайте сроки',
    }


def clean_person(value, transcript):
    if value is None:
        return None
    value = value.strip()
    technical = {s['speakerId'].casefold() for s in transcript}
    if value.casefold() in technical | {'null', 'none', 'unknown', 'неизвестно', 'неизвестный', 'n/a'}:
        return None
    if re.fullmatch(r'(?:speaker|спикер)[_\s-]*\d+', value, re.I):
        return None
    return value or None


def clean_deadline(value):
    if value is None:
        return None
    if value.strip().casefold().rstrip('.!') in {
        'не надо откладывать', 'не откладывать', 'согласуем бюджет быстро',
        'сразу', 'быстро', 'срочно', 'null', 'unknown',
    }:
        return None
    # Extract an explicit date from a mistakenly copied full instruction.
    date = re.search(r'\b(?:до|к)\s+\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4}(?:\s+года)?)?', value, re.I)
    return date.group(0) if date else value
