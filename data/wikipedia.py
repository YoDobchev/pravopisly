from pathlib import Path
import re
import json

from commas import sentence_to_word_labels

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

    if ":" in sentence:
        return False

    if ";" in sentence:
        return False

    if sentence.count("(") > 1 or sentence.count(")") > 1:
        return False

    if sentence.count('"') > 2:
        return False

    if sentence.count("„") > 2 or sentence.count("“") > 2:
        return False

    digit_count = len(re.findall(r"\d", sentence))
    if digit_count > 8:
        return False

    cyrillic_letters = re.findall(r"[А-Яа-я]", sentence)
    if len(cyrillic_letters) < 25:
        return False

    latin_letters = re.findall(r"[A-Za-z]", sentence)
    if len(latin_letters) > 5:
        return False

    symbol_count = len(
        re.findall(r"[^А-Яа-яA-Za-z0-9\s,.!?;:„“\"'()—–-]", sentence)
    )
    if symbol_count > 1:
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



def append_wikipedia_sentences(path: str):
    input_folder = Path(path)
    files = [file for file in input_folder.rglob("*") if file.is_file()]

    with open("dataset.jsonl", "w", encoding="utf-8") as df:
        counter = 5

        for file in files:
            counter += 1
            if counter >= 10:
                break

            text = file.read_text(encoding="utf-8")

            for sentence in clean_and_extract_sentences(text):
                clean_sentence, comma_labels = sentence_to_word_labels(sentence)

                words = clean_sentence.split()

                if len(words) != len(comma_labels):
                    raise ValueError(
                        f"Comma mismatch: {len(comma_labels)} labels vs "
                        f"{len(words)} words\n{clean_sentence}"
                    )

                item = {
                    "s": clean_sentence,
                    "c": comma_labels,
                }

                df.write(json.dumps(item, ensure_ascii=False) + "\n")