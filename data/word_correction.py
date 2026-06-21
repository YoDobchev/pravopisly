import pandas as pd


def is_only_word(value: str) -> bool:
    value = str(value).strip()
    return value != "" and value.isalpha()


def load_word_corrections(path: str) -> dict[str, str]:
    try:
        df = pd.read_csv(
            path,
            header=None,
            names=["wrong", "correct"],
            usecols=[0, 1],
            dtype=str,
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
    except pd.errors.ParserError:
        return {}

    corrections: dict[str, str] = {}

    for _, row in df.iterrows():
        wrong = str(row["wrong"]).strip()
        correct = str(row["correct"]).strip()

        if not is_only_word(wrong):
            continue

        if not is_only_word(correct):
            continue

        corrections[wrong] = correct

    return corrections


def correct_words(sentence: str, corrections: dict[str, str]) -> str:
    result = []
    current_word = []

    def flush_word():
        if not current_word:
            return

        word = "".join(current_word)
        result.append(corrections.get(word, word))
        current_word.clear()

    for ch in sentence:
        if ch.isalpha():
            current_word.append(ch)
        else:
            flush_word()
            result.append(ch)

    flush_word()

    return "".join(result)