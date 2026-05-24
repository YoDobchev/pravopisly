from pathlib import Path
import re
import json
import random
import pandas as pd

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


def build_replacements(lemmas_file):
    word_col = lemmas_file.columns[0]
    lemma_col = lemmas_file.columns[1]

    replacements = {}

    for _, group in lemmas_file.groupby(lemma_col):
        forms = (
            group[word_col]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        forms = [form for form in forms if re.fullmatch(r"[А-Яа-я]+", form)]

        if len(forms) < 2:
            continue

        forms = list(dict.fromkeys(forms))

        for form in forms:
            key = form.lower()

            alternatives = [
                other for other in forms
                if other.lower() != key
            ]

            if alternatives:
                replacements.setdefault(key, []).extend(alternatives)

    for key in replacements:
        replacements[key] = list(dict.fromkeys(replacements[key]))

    return replacements


def split_word(word: str):
    match = re.match(r"^([^А-Яа-я]*)([А-Яа-я]+)([^А-Яа-я]*)$", word)

    if not match:
        return "", word, ""

    return match.group(1), match.group(2), match.group(3)


def keep_case(original: str, replacement: str):
    if original.isupper():
        return replacement.upper()

    if original[0].isupper():
        return replacement[0].upper() + replacement[1:].lower()

    return replacement.lower()


def make_grammar_mistake(sentence: str, replacements):
    words = sentence.split()
    grammar_labels = [0] * len(words)

    possible = []

    for i, word in enumerate(words):
        prefix, core, suffix = split_word(word)

        if core.lower() in replacements:
            possible.append(i)

    if not possible:
        return sentence, grammar_labels

    mistake_count = 1

    if len(possible) >= 2 and random.random() < 0.25:
        mistake_count = 2

    selected = random.sample(possible, min(mistake_count, len(possible)))
    new_words = words[:]

    for i in selected:
        prefix, core, suffix = split_word(words[i])
        alternatives = replacements[core.lower()]

        replacement = random.choice(alternatives)
        replacement = keep_case(core, replacement)

        new_words[i] = prefix + replacement + suffix
        grammar_labels[i] = 1

    return " ".join(new_words), grammar_labels


def append_text_data(path: str, lemmasPath):
    input_folder = Path(path)
    files = [file for file in input_folder.rglob("*") if file.is_file()]

    lemmas_file = pd.read_csv(lemmasPath, sep="\t")
    replacements = build_replacements(lemmas_file)

    print(lemmas_file.head())
    print(f"Loaded {len(replacements)} replaceable words")

    with open("dataset.jsonl", "w", encoding="utf-8") as df:
        for file in files:
            text = file.read_text(encoding="utf-8")

            for sentence in clean_and_extract_sentences(text):
                clean_sentence, comma_labels = sentence_to_word_labels(
                    sentence)

                words = clean_sentence.split()

                if len(words) != len(comma_labels):
                    raise ValueError(
                        f"Comma mismatch: {len(comma_labels)} labels vs "
                        f"{len(words)} words\n{clean_sentence}"
                    )

                bad_sentence, grammar_labels = make_grammar_mistake(
                    clean_sentence,
                    replacements,
                )

                if sum(grammar_labels) > 0:
                    bad_words = bad_sentence.split()

                    if len(bad_words) != len(words):
                        raise ValueError(
                            f"Grammar mismatch: {len(grammar_labels)} labels vs "
                            f"{len(bad_words)} words\n{bad_sentence}"
                        )

                    bad_item = {
                        "s": bad_sentence,
                        "c": comma_labels,
                        "g": grammar_labels,
                    }

                    df.write(json.dumps(bad_item, ensure_ascii=False) + "\n")

                good_item = {
                    "s": clean_sentence,
                    "c": comma_labels,
                    "g": [0] * len(words),
                }

                df.write(json.dumps(good_item, ensure_ascii=False) + "\n")
