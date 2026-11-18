"""Токенизация для BM25: нижний регистр, ё→е, стемминг Snowball для русского (без внешних зависимостей)."""

import re
from functools import lru_cache

VOWELS = "аеиоуыэюя"

PERFECTIVE_GERUND_1 = ("вшись", "вши", "в")          # после а/я
PERFECTIVE_GERUND_2 = ("ывшись", "ившись", "ывши", "ивши", "ив", "ыв")
ADJECTIVE = ("ими", "ыми", "его", "ого", "ему", "ому", "ее", "ие", "ые", "ое", "ей", "ий", "ый", "ой",
             "ем", "им", "ым", "ом", "их", "ых", "ую", "юю", "ая", "яя", "ою", "ею")
PARTICIPLE_1 = ("ем", "нн", "вш", "ющ", "щ")          # после а/я
PARTICIPLE_2 = ("ивш", "ывш", "ующ")
REFLEXIVE = ("ся", "сь")
VERB_1 = ("ете", "йте", "ешь", "нно", "ла", "на", "ли", "ем", "ло", "но", "ет", "ют", "ны", "ть", "й", "л", "н")
VERB_2 = ("ейте", "уйте", "ила", "ыла", "ена", "ите", "или", "ыли", "ило", "ыло", "ено", "ует", "уют", "ены",
          "ить", "ыть", "ишь", "ей", "уй", "ил", "ыл", "им", "ым", "ен", "ят", "ит", "ыт", "ую", "ю")
NOUN = ("иями", "ями", "ами", "ией", "иям", "ием", "иях", "ев", "ов", "ие", "ье", "еи", "ии", "ей", "ой", "ий",
        "ям", "ем", "ам", "ом", "ах", "ях", "ию", "ью", "ия", "ья", "а", "е", "и", "й", "о", "у", "ы", "ь", "ю", "я")
SUPERLATIVE = ("ейше", "ейш")
DERIVATIONAL = ("ость", "ост")


def _sorted(suffixes):
    return tuple(sorted(suffixes, key=len, reverse=True))


PERFECTIVE_GERUND_1, PERFECTIVE_GERUND_2 = _sorted(PERFECTIVE_GERUND_1), _sorted(PERFECTIVE_GERUND_2)
ADJECTIVE, PARTICIPLE_1, PARTICIPLE_2 = _sorted(ADJECTIVE), _sorted(PARTICIPLE_1), _sorted(PARTICIPLE_2)
VERB_1, VERB_2, NOUN = _sorted(VERB_1), _sorted(VERB_2), _sorted(NOUN)


def _regions(word):
    rv = r2 = len(word)
    for i, ch in enumerate(word):
        if ch in VOWELS:
            rv = i + 1
            break
    r1 = len(word)
    for i in range(1, len(word)):
        if word[i] not in VOWELS and word[i - 1] in VOWELS:
            r1 = i + 1
            break
    for i in range(r1 + 1, len(word)):
        if word[i] not in VOWELS and word[i - 1] in VOWELS:
            r2 = i + 1
            break
    return rv, r2


def _strip(word, start, group_1, group_2=()):
    """Убрать самый длинный суффикс из group_2 или (после а/я) из group_1, если он в регионе с start."""
    region = word[start:]
    for suffix in group_2:
        if region.endswith(suffix):
            return word[: -len(suffix)], True
    for suffix in group_1:
        if region.endswith(suffix) and len(region) > len(suffix) and region[-len(suffix) - 1] in "ая":
            return word[: -len(suffix)], True
    return word, False


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    if len(word) < 3 or not re.fullmatch(r"[а-я]+", word):
        return word
    rv, r2 = _regions(word)
    word, done = _strip(word, rv, PERFECTIVE_GERUND_1, PERFECTIVE_GERUND_2)
    if not done:
        word, _ = _strip(word, rv, (), REFLEXIVE)
        word, done = _strip(word, rv, (), ADJECTIVE)
        if done:
            word, _ = _strip(word, rv, PARTICIPLE_1, PARTICIPLE_2)
        else:
            word, done = _strip(word, rv, VERB_1, VERB_2)
            if not done:
                word, _ = _strip(word, rv, (), NOUN)
    if word[rv:].endswith("и"):
        word = word[:-1]
    for suffix in DERIVATIONAL:
        if len(word) > r2 and word[r2:].endswith(suffix):
            word = word[: -len(suffix)]
            break
    if word[rv:].endswith("ь"):
        word = word[:-1]
    else:
        for suffix in SUPERLATIVE:
            if word[rv:].endswith(suffix):
                word = word[: -len(suffix)]
                break
        if word.endswith("нн"):
            word = word[:-1]
    return word


STOPWORDS = set("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до его ее
ей если есть еще же за и из или им их к как ко когда кто ли либо между мы на над нас не него нее нет ни них но
о об от по под при про с со так также то того тоже той только том ту у уже чем что чтобы эта эти это этого этой
этом этот какова каков каковы какой какая какие каким какую сколько который которые которая которых составляет
составил составила составили составило согласно данным данные
""".split())


def tokenize(text: str) -> list:
    text = text.lower().replace("ё", "е")
    return [stem(t) for t in re.findall(r"[a-zа-я0-9]+", text) if t not in STOPWORDS]
