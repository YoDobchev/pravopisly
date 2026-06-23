from pathlib import Path
import orjson
import pandas as pd

from commas import sentence_to_word_labels
from filters import clean_and_extract_sentences
from lemmas import build_replacements
from word_correction import correct_words, load_word_corrections
from mistakes import make_grammar_mistake, make_spelling_mistake, load_spelling_words


def write_item(out_file, item):
    out_file.write(orjson.dumps(item).decode("utf-8") + "\n")


def append_corpus_data(
    path: str,
    lemmasPath,
    spellingWordsPath,
    wordCorrectionPath,
    output_path="dataset.jsonl",
    max_comma_rows=50_000,
    max_spelling_rows=50_000,
    max_grammar_rows=20_000,
):
    input_folder = Path(path)
    files = [file for file in input_folder.rglob("*") if file.is_file()]

    lemmas_file = pd.read_csv(lemmasPath, sep="\t")
    replacements = build_replacements(lemmas_file)

    spelling_words = load_spelling_words(spellingWordsPath)

    word_corrections = {}

    if wordCorrectionPath is not None:
        word_corrections = load_word_corrections(wordCorrectionPath)

    print(lemmas_file.head())
    print(f"Loaded {len(replacements)} replaceable words")
    print(f"Loaded {len(spelling_words)} spelling words")

    comma_count = 0
    spelling_count = 0
    grammar_count = 0

    with open(output_path, "w", encoding="utf-8") as df:
        for file in files:
            if (
                comma_count >= max_comma_rows
                and spelling_count >= max_spelling_rows
                and grammar_count >= max_grammar_rows
            ):
                break

            text = file.read_text(encoding="utf-8")

            for sentence in clean_and_extract_sentences(text):
                if (
                    comma_count >= max_comma_rows
                    and spelling_count >= max_spelling_rows
                    and grammar_count >= max_grammar_rows
                ):
                    break

                if wordCorrectionPath is not None:
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

                if comma_count < max_comma_rows:
                    comma_item = {
                        "s": clean_sentence,
                        "c": comma_labels,
                    }

                    write_item(df, comma_item)
                    comma_count += 1

                if spelling_count < max_spelling_rows:
                    clean_spelling_item = {
                        "s": clean_sentence,
                        "sp": [0] * len(words),
                    }

                    write_item(df, clean_spelling_item)
                    spelling_count += 1

                if spelling_count < max_spelling_rows:
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

                        spelling_item = {
                            "s": bad_sentence_spelling,
                            "sp": spelling_labels,
                        }

                        write_item(df, spelling_item)
                        spelling_count += 1

                if grammar_count < max_grammar_rows:
                    clean_grammar_item = {
                        "s": clean_sentence,
                        "g": [0] * len(words),
                    }

                    write_item(df, clean_grammar_item)
                    grammar_count += 1

                if grammar_count < max_grammar_rows:
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

                        grammar_item = {
                            "s": bad_sentence_grammar,
                            "g": grammar_labels,
                        }

                        write_item(df, grammar_item)
                        grammar_count += 1

    print("Corpus rows added:")
    print(f"  comma:    {comma_count}")
    print(f"  spelling: {spelling_count}")
    print(f"  grammar:  {grammar_count}")
