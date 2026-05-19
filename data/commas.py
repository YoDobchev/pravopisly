import json
import os


def sentence_to_word_labels(sentence: str):
    words = []
    labels = []

    current_word = []
    current_label = 0

    def flush_word():
        nonlocal current_word, current_label

        if current_word:
            words.append("".join(current_word))
            labels.append(current_label)

            current_word = []
            current_label = 0

    for char in sentence:
        if char == ",":
            if current_word:
                current_label = 1
            elif labels:
                labels[-1] = 1

            continue

        if char.isspace():
            flush_word()
        else:
            current_word.append(char)

    flush_word()

    clean_sentence = " ".join(words)

    return clean_sentence, labels


def label_commas():
    dataset_path = "dataset.jsonl"
    temp_path = dataset_path + ".tmp"

    with open(dataset_path, "r", encoding="utf-8") as infile, \
            open(temp_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            obj = json.loads(line)
            clean_sentence, labels = sentence_to_word_labels(obj["s"])
            obj["c"] = labels
            obj["s"] = clean_sentence
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(temp_path, dataset_path)
