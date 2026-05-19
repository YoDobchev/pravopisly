import pandas as pd
import json
import difflib

from commas import sentence_to_word_labels


def find_and_label_differences(erroneous: str, correct: str):
    erroneous_clean, erroneous_commas = sentence_to_word_labels(erroneous)
    correct_clean, correct_commas = sentence_to_word_labels(correct)

    erroneous_words = erroneous_clean.split()
    correct_words = correct_clean.split()

    spelling_labels = [0] * len(erroneous_words)

    matcher = difflib.SequenceMatcher(
        None,
        erroneous_words,
        correct_words,
    )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        elif tag in {"replace", "delete"}:
            for i in range(i1, i2):
                spelling_labels[i] = 1

        elif tag == "insert":
            if not spelling_labels:
                continue

            if i1 > 0:
                spelling_labels[i1 - 1] = 1
            elif i1 < len(spelling_labels):
                spelling_labels[i1] = 1

    return {
        "erroneous_clean": erroneous_clean,
        "correct_clean": correct_clean,
        "erroneous_commas": erroneous_commas,
        "correct_commas": correct_commas,
        "spelling_labels": spelling_labels,
    }


def append_spelling_data(path: str, output_path="dataset.jsonl", mode="a"):
    data = pd.read_csv(path)

    with open(output_path, mode, encoding="utf-8") as out_file:
        for row in data.itertuples(index=False):
            result = find_and_label_differences(
                row.erroneous,
                row.correct,
            )

            erroneous_clean = result["erroneous_clean"]
            correct_clean = result["correct_clean"]

            erroneous_words = erroneous_clean.split()
            correct_words = correct_clean.split()

            spelling_labels = result["spelling_labels"]
            erroneous_commas = result["erroneous_commas"]
            correct_commas = result["correct_commas"]

            assert len(spelling_labels) == len(erroneous_words), (
                f"spelling mismatch: {len(spelling_labels)} labels vs "
                f"{len(erroneous_words)} words\n{erroneous_clean}"
            )

            assert len(erroneous_commas) == len(erroneous_words), (
                f"comma mismatch in erroneous sentence: "
                f"{len(erroneous_commas)} labels vs "
                f"{len(erroneous_words)} words\n{erroneous_clean}"
            )

            assert len(correct_commas) == len(correct_words), (
                f"comma mismatch in correct sentence: "
                f"{len(correct_commas)} labels vs "
                f"{len(correct_words)} words\n{correct_clean}"
            )

            erroneous_item = {
                "s": erroneous_clean,
                "sp": spelling_labels,
                "c": erroneous_commas,
            }

            out_file.write(
                json.dumps(erroneous_item, ensure_ascii=False) + "\n"
            )

            correct_item = {
                "s": correct_clean,
                "sp": [0] * len(correct_words),
                "c": correct_commas,
            }

            out_file.write(
                json.dumps(correct_item, ensure_ascii=False) + "\n"
            )