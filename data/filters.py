import re


def is_good_sentence(sentence: str) -> bool:
    sentence = sentence.strip()

    if len(sentence) < 35:
        return False

    words = sentence.split()

    if len(words) < 6:
        return False

    if len(words) > 35:
        return False

    start = sentence.lstrip('„"“')
    if not start or not re.match(r"^[А-Я]", start):
        return False

    bad_endings = (
        " г.",
        " в.",
        " сл.",
        " пр.",
        " ок.",
        " т.",
        " с.",
        " д.",
        " проф.",
        " доц.",
        " д-р",
    )

    if sentence.endswith(bad_endings):
        return False

    bracket_pairs = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
    ]

    for left, right in bracket_pairs:
        if sentence.count(left) != sentence.count(right):
            return False

    bad_patterns = [
        r"†",
        r"\bсл\.",
        r"\bпр\.",
        r"\bвж\.",
        r"\bт\.\s*нар\.",
        r"\bдн\.",
        r"\bок\.",
        r"\bкв\.",
        r"\bс\.",
        r"\bсп\.",
        r"\bмаг\.",
        r"\bпроф\.",
        r"\bдоц\.",
        r"\bд-р",
        r"ISBN",
        r"Файл:",
        r"Категория:",
        r"Шаблон:",
        r"Други:",
        r"Виж още",
        r"може да се отнася за",
        r"^\w+\s+\w+\s+\(\)",
        r"\(\s*\)",
        r"\(;",
        r";\s*\)",
        r"\b[a-zA-Z]{3,}\b",
        r"https?",
        r"www\.",
        r"\|",
        r"\[[^\]]*\]",
        r"\{[^}]*\}",
    ]

    for pattern in bad_patterns:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return False

    digit_count = len(re.findall(r"\d", sentence))
    if digit_count > 8:
        return False

    cyrillic_letters = re.findall(r"[А-Яа-я]", sentence)
    if len(cyrillic_letters) < 25:
        return False

    normal_words = 0

    for word in words:
        cleaned = word.strip(".,!?;:„“\"'()—–-")
        if re.search(r"[А-Яа-я]", cleaned):
            normal_words += 1

    if normal_words / len(words) < 0.85:
        return False

    if len(words) >= 4 and words[0:2] == words[2:4]:
        return False

    uppercase_words_after_first = 0

    for word in words[1:]:
        cleaned = word.strip(".,!?;:„“\"'()—–-")
        if re.match(r"^[А-Я]", cleaned):
            uppercase_words_after_first += 1

    if uppercase_words_after_first / max(1, len(words) - 1) > 0.35:
        return False

    return True


def clean_and_extract_sentences(text: str):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.findall(r"[^.!?]*[.!?]", text)

    for sentence in sentences:
        sentence = sentence.strip()

        if not is_good_sentence(sentence):
            continue

        yield sentence