import json
import torch
from torch.utils.data import Dataset


class PravopislyDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=64):
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                text = row["s"]
                words = text.split()

                comma_labels = row.get("c")
                spelling_labels = row.get("sp")

                if comma_labels is not None and len(comma_labels) != len(words):
                    raise ValueError(
                        f"comma label length mismatch: "
                        f"{len(comma_labels)} labels vs {len(words)} words\n{text}"
                    )

                if spelling_labels is not None and len(spelling_labels) != len(words):
                    raise ValueError(
                        f"spelling label length mismatch: "
                        f"{len(spelling_labels)} labels vs {len(words)} words\n{text}"
                    )

                encoded = tokenizer(
                    words,
                    is_split_into_words=True,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                word_ids = encoded.word_ids(batch_index=0)

                aligned_comma_labels = self.align_labels(
                    word_ids,
                    comma_labels,
                )

                aligned_spelling_labels = self.align_labels(
                    word_ids,
                    spelling_labels,
                )

                self.examples.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "comma_labels": torch.tensor(aligned_comma_labels),
                    "spelling_labels": torch.tensor(aligned_spelling_labels),
                    "text": text,
                    "words": words,
                })

    def align_labels(self, word_ids, word_labels):
        aligned = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)

            elif word_labels is None:
                aligned.append(-100)

            elif word_id != previous_word_id:
                aligned.append(word_labels[word_id])

            else:
                aligned.append(-100)

            previous_word_id = word_id

        return aligned

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "comma_labels": item["comma_labels"],
            "spelling_labels": item["spelling_labels"],
        }