"""Conservative normalization: unsupported/ambiguous wording stays unresolved."""
import re
from datetime import date, timedelta

MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()


def normalize_deadline(text, meeting_date=None):
    if not text:
        return None
    value = text.lower().strip()
    try:
        explicit = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", value)
        if explicit:
            return date.fromisoformat(explicit[1]).isoformat()
        explicit = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", value)
        if explicit:
            return date(int(explicit[3]), int(explicit[2]), int(explicit[1])).isoformat()
        explicit = re.search(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})\b", value)
        if explicit:
            return date(int(explicit[3]), MONTHS.index(explicit[2]) + 1, int(explicit[1])).isoformat()
        if not meeting_date:
            return None
        base = date.fromisoformat(meeting_date)
        if value in ("сегодня", "бүгін"):
            return base.isoformat()
        if value in ("завтра", "ертең"):
            return (base + timedelta(days=1)).isoformat()
        relative = re.fullmatch(r"через (\d+|одну|две|три|один|два) (день|дня|дней|неделю|недели|недель)", value)
        if relative:
            words = {"одну": 1, "две": 2, "три": 3, "один": 1, "два": 2}
            count = words.get(relative[1]) or int(relative[1])
            return (base + timedelta(days=count * (7 if relative[2].startswith("нед") else 1))).isoformat()
    except (ValueError, OverflowError):
        return None
    # Weekdays, end-of-week and dates without years require a team policy.
    return None
