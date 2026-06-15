import re
import random
import pandas as pd


def load_spelling_words(path: str, min_frequency: int = 1):
    words_file = pd.read_csv(
        path,
        header=None,
        names=["word", "frequency"],
    )

    words_file = words_file.dropna(subset=["word"])

    words_file["word"] = (
        words_file["word"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    words_file["frequency"] = pd.to_numeric(
        words_file["frequency"],
        errors="coerce",
    )

    words_file = words_file.dropna(subset=["frequency"])

    words_file = words_file[
        words_file["word"].str.fullmatch(r"[а-я]+")
    ]

    words_file = words_file[
        words_file["frequency"] >= min_frequency
    ]

    spelling_words = set(words_file["word"])

    return spelling_words


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

    if len(possible) >= 2 and random.random() < 0.5:
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


def make_spelling_error_word(word: str, spelling_words):
    lower = word.lower()

    replacements = {
        "а": ["ъ", "я"],
        "ъ": ["а"],
        "е": ["и"],
        "и": ["е"],
        "о": ["у", "а"],
        "у": ["о"],
        "я": ["а"],
        "ю": ["у"],
        "с": ["з"],
        "з": ["с"],
        "т": ["д"],
        "д": ["т"],
        "г": ["к"],
        "к": ["г"],
    }

    candidates = []

    if len(lower) >= 5:
        for index in range(1, len(lower)):
            result = lower[:index] + lower[index + 1:]
            candidates.append(result)

    if len(lower) >= 5:
        for index in range(1, len(lower)):
            result = lower[:index] + lower[index] + lower[index:]
            candidates.append(result)

    for index in range(len(lower) - 1):
        if lower[index] != lower[index + 1]:
            result = (
                lower[:index]
                + lower[index + 1]
                + lower[index]
                + lower[index + 2:]
            )
            candidates.append(result)

    for index, char in enumerate(lower):
        if char in replacements:
            for replacement in replacements[char]:
                result = lower[:index] + replacement + lower[index + 1:]
                candidates.append(result)

    candidates = list(dict.fromkeys(candidates))

    candidates = [
        candidate for candidate in candidates
        if candidate != lower
    ]

    candidates = [
        candidate for candidate in candidates
        if candidate not in spelling_words
    ]

    if not candidates:
        return word

    result = random.choice(candidates)

    return keep_case(word, result)


def make_spelling_mistake(sentence: str, spelling_words):
    words = sentence.split()
    spelling_labels = [0] * len(words)

    possible = []

    for i, word in enumerate(words):
        prefix, core, suffix = split_word(word)

        if not re.fullmatch(r"[А-Яа-я]+", core):
            continue

        if len(core) < 4:
            continue

        if core.lower() not in spelling_words:
            continue

        possible.append(i)

    if not possible:
        return sentence, spelling_labels

    mistake_count = 1

    if len(possible) >= 2 and random.random() < 0.5:
        mistake_count = 2

    selected = random.sample(possible, min(mistake_count, len(possible)))
    new_words = words[:]

    for i in selected:
        prefix, core, suffix = split_word(words[i])

        replacement = make_spelling_error_word(core, spelling_words)

        if replacement != core:
            new_words[i] = prefix + replacement + suffix
            spelling_labels[i] = 1

    if sum(spelling_labels) == 0:
        return sentence, spelling_labels

    return " ".join(new_words), spelling_labels
