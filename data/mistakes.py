import re
import random


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


def make_spelling_error_word(word: str):
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

    operations = []

    if len(lower) >= 5:
        operations.append("delete")
        operations.append("duplicate")

    swap_positions = []

    for i in range(len(lower) - 1):
        if lower[i] != lower[i + 1]:
            swap_positions.append(i)

    if swap_positions:
        operations.append("swap")

    replace_positions = []

    for i, char in enumerate(lower):
        if char in replacements:
            replace_positions.append(i)

    if replace_positions:
        operations.append("replace")

    if not operations:
        return word

    operation = random.choice(operations)

    if operation == "delete":
        index = random.randrange(1, len(lower))
        result = lower[:index] + lower[index + 1:]

    elif operation == "duplicate":
        index = random.randrange(1, len(lower))
        result = lower[:index] + lower[index] + lower[index:]

    elif operation == "swap":
        index = random.choice(swap_positions)
        result = (
            lower[:index]
            + lower[index + 1]
            + lower[index]
            + lower[index + 2:]
        )

    else:
        index = random.choice(replace_positions)
        char = lower[index]
        replacement = random.choice(replacements[char])
        result = lower[:index] + replacement + lower[index + 1:]

    return keep_case(word, result)


def make_spelling_mistake(sentence: str):
    words = sentence.split()
    spelling_labels = [0] * len(words)

    possible = []

    for i, word in enumerate(words):
        prefix, core, suffix = split_word(word)

        if re.fullmatch(r"[А-Яа-я]+", core) and len(core) >= 4:
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

        replacement = make_spelling_error_word(core)

        if replacement != core:
            new_words[i] = prefix + replacement + suffix
            spelling_labels[i] = 1

    if sum(spelling_labels) == 0:
        return sentence, spelling_labels

    return " ".join(new_words), spelling_labels