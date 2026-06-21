from pathlib import Path
import json
import pandas as pd

from commas import sentence_to_word_labels
from filters import clean_and_extract_sentences
from lemmas import build_replacements
from word_correction import correct_words, load_word_corrections
from mistakes import make_grammar_mistake, make_spelling_mistake, load_spelling_words


def append_corpus_data(path: str, lemmasPath, spellingWordsPath, wordCorrectionPath):
    input_folder = Path(path)
    files = [file for file in input_folder.rglob("*") if file.is_file()]

    lemmas_file = pd.read_csv(lemmasPath, sep="\t")
    replacements = build_replacements(lemmas_file)

    spelling_words = load_spelling_words(spellingWordsPath)

    if wordCorrectionPath != None:
        word_corrections = load_word_corrections(wordCorrectionPath)

    print(lemmas_file.head())
    print(f"Loaded {len(replacements)} replaceable words")
    print(f"Loaded {len(spelling_words)} spelling words")

    with open("dataset.jsonl", "w", encoding="utf-8") as df:
        for file in files:
            text = file.read_text(encoding="utf-8")

            for sentence in clean_and_extract_sentences(text):
                if wordCorrectionPath != None:
                    sentence = correct_words(sentence, word_corrections)

                clean_sentence, comma_labels = sentence_to_word_labels(
                    sentence
                )

                words = clean_sentence.split()

                if len(words) != len(comma_labels):
                    raise ValueError(
                        f"Comma mismatch: {len(comma_labels)} labels vs "
                        f"{len(words)} words\n{clean_sentence}"
                    )

                bad_sentence_grammar, grammar_labels = make_grammar_mistake(
                    clean_sentence,
                    replacements,
                )

                if sum(grammar_labels) > 0:
                    bad_words = bad_sentence_grammar.split()

                    if len(bad_words) != len(words):
                        raise ValueError(
                            f"Grammar mismatch: {len(grammar_labels)} labels vs "
                            f"{len(bad_words)} words\n{bad_sentence_grammar}"
                        )

                    bad_item = {
                        "s": bad_sentence_grammar,
                        "c": comma_labels,
                        "g": grammar_labels,
                        "sp": [0] * len(words),
                    }

                    df.write(json.dumps(bad_item, ensure_ascii=False) + "\n")

                bad_sentence_spelling, spelling_labels = make_spelling_mistake(
                    clean_sentence,
                    spelling_words
                )

                if sum(spelling_labels) > 0:
                    bad_words = bad_sentence_spelling.split()

                    if len(bad_words) != len(words):
                        raise ValueError(
                            f"Spelling mismatch: {len(spelling_labels)} labels vs "
                            f"{len(bad_words)} words\n{bad_sentence_spelling}"
                        )

                    bad_item = {
                        "s": bad_sentence_spelling,
                        "c": comma_labels,
                        "g": [0] * len(words),
                        "sp": spelling_labels,
                    }

                    df.write(json.dumps(bad_item, ensure_ascii=False) + "\n")

                good_item = {
                    "s": clean_sentence,
                    "c": comma_labels,
                    "g": [0] * len(words),
                    "sp": [0] * len(words),
                }

                df.write(json.dumps(good_item, ensure_ascii=False) + "\n")
