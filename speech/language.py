"""Small optional RU/KZ text heuristic, not a statistical language classifier."""

import re
import unicodedata
from typing import Literal

SegmentLanguage = Literal["ru", "kk", "mixed"]
_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
_KAZAKH_LETTERS = frozenset("әғқңөұүһі")
# Avoid shared business loanwords (e.g. проект, график) as language evidence.
_RU = frozenset("""
    это этот эта эти мы вы они сегодня завтра вчера нужно необходимо пожалуйста
    подготовьте подготовить итоговый отчёт отчет пятницы понедельника до для чтобы
    будет будем обсудим спасибо добрый день коллеги новый новая новое новые
    срок сроки согласовать отправьте отправить сделайте сделаем должны должен
    решение решили следующий следующей недели неделю работу работы сдать готов
""".split())
_KK = frozenset("""
    мен сен ол біз сіз олар осы сол бұл және немесе апта аптада жасап жасаймыз
    жасау есеп есепті керек қажет бар жоқ емес деп туралы бойынша рахмет сәлем
    бүгін ертең дейін беріңіз беремін жиналыс жиналысты бастаймыз сөйлесіп
""".split())


def detectSegmentLanguage(text: str) -> SegmentLanguage | None:
    """Return optional language evidence; never translate or modify source text.

    Two distinct Kazakh-evidence tokens avoid treating a lone name as code-switching.
    Mixed additionally needs a Russian marker; Russian alone needs two markers.
    Sparse/unknown text is left unclassified instead of assuming all Cyrillic is RU.
    """
    if not isinstance(text, str):
        return None
    words = set(_WORDS.findall(unicodedata.normalize("NFC", text).casefold()))
    if len(words) < 2:
        return None
    kk = sum(word in _KK or bool(_KAZAKH_LETTERS.intersection(word)) for word in words)
    ru = len(words & _RU)
    if kk >= 2:
        return "mixed" if ru else "kk"
    if ru >= 2:
        return "ru"
    return None
