from pathlib import Path
import csv
import random
import pandas as pd

from filters import clean_and_extract_sentences
from lemmas import build_replacements
from mistakes import make_grammar_mistake, make_spelling_mistake, load_spelling_words
from word_correction import correct_words, load_word_corrections


def normalize(sentence):
    return " ".join(sentence.strip().split())


def make_comma_mistake(sentence):
    positions = [i for i, ch in enumerate(sentence) if ch == ","]

    if positions:
        i = random.choice(positions)
        return sentence[:i] + sentence[i + 1:]

    words = sentence.split()

    if len(words) < 4:
        return None

    possible = []

    for i in range(len(words) - 1):
        word = words[i]

        if "," in word:
            continue

        if word.endswith((".", "!", "?", ";", ":")):
            continue

        possible.append(i)

    if not possible:
        return None

    i = random.choice(possible)
    words[i] = words[i] + ","

    return " ".join(words)


def add_pair(rows, seen, erroneous, correct):
    if erroneous is None:
        return False

    erroneous = normalize(erroneous)
    correct = normalize(correct)

    if erroneous == correct:
        return False

    key = (erroneous, correct)

    if key in seen:
        return False

    seen.add(key)

    rows.append({
        "erroneous": erroneous,
        "correct": correct,
    })

    return True


def make_eval_csv(
    corpus_path,
    lemmas_path,
    spelling_words_path,
    word_correction_path=None,
    output_path="./eval_sen.csv",
    max_rows=30000,
    seed=42,
):
    random.seed(seed)

    files = sorted(
        file for file in Path(corpus_path).rglob("*")
        if file.is_file()
    )

    lemmas_file = pd.read_csv(lemmas_path, sep="\t")
    replacements = build_replacements(lemmas_file)
    spelling_words = load_spelling_words(spelling_words_path)

    word_corrections = {}

    if word_correction_path is not None:
        word_corrections = load_word_corrections(word_correction_path)

    max_comma = max_rows // 3
    max_spelling = max_rows // 3
    max_grammar = max_rows - max_comma - max_spelling

    comma_count = 0
    spelling_count = 0
    grammar_count = 0

    rows = []
    seen = set()

    for file in files:
        if len(rows) >= max_rows:
            break

        text = file.read_text(encoding="utf-8", errors="ignore")

        for sentence in clean_and_extract_sentences(text):
            if len(rows) >= max_rows:
                break

            correct = normalize(sentence)

            if word_correction_path is not None:
                correct = normalize(correct_words(correct, word_corrections))

            if comma_count < max_comma:
                erroneous = make_comma_mistake(correct)

                if add_pair(rows, seen, erroneous, correct):
                    comma_count += 1

            if spelling_count < max_spelling:
                erroneous, labels = make_spelling_mistake(
                    correct,
                    spelling_words,
                )

                if sum(labels) > 0:
                    if add_pair(rows, seen, erroneous, correct):
                        spelling_count += 1

            if grammar_count < max_grammar:
                erroneous, labels = make_grammar_mistake(
                    correct,
                    replacements,
                )

                if sum(labels) > 0:
                    if add_pair(rows, seen, erroneous, correct):
                        grammar_count += 1

    random.shuffle(rows)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["erroneous", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")
    print(f"Comma: {comma_count}")
    print(f"Spelling: {spelling_count}")
    print(f"Grammar: {grammar_count}")
