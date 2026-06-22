import orjson
from torch.utils.data import Dataset
import torch


class PravopislyDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = []

        with open(jsonl_path, "rb", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                row = orjson.loads(line)

                text = row["s"]
                words = text.split()

                c = row.get("c")
                sp = row.get("sp")
                g = row.get("g")

                if c is not None and len(c) != len(words):
                    raise ValueError(
                        f"comma labels mismatch on line {line_num}")

                if sp is not None and len(sp) != len(words):
                    raise ValueError(
                        f"spelling labels mismatch on line {line_num}")

                if g is not None and len(g) != len(words):
                    raise ValueError(
                        f"grammar labels mismatch on line {line_num}")

                self.rows.append({
                    "words": words,
                    "c": c,
                    "sp": sp,
                    "g": g,
                })

        print(f"Loaded {len(self.rows)} raw examples", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class PravopislyCollator:
    def __init__(self, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch_words = [item["words"] for item in batch]

        encoded = self.tokenizer(
            batch_words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        comma_labels = self.align_batch(encoded, [item["c"] for item in batch])
        spelling_labels = self.align_batch(
            encoded, [item["sp"] for item in batch])
        grammar_labels = self.align_batch(
            encoded, [item["g"] for item in batch])

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "comma_labels": comma_labels,
            "spelling_labels": spelling_labels,
            "grammar_labels": grammar_labels,
        }

    def align_batch(self, encoded, batch_word_labels):
        batch_size = len(batch_word_labels)
        seq_len = encoded["input_ids"].size(1)

        labels = torch.full(
            (batch_size, seq_len),
            -100,
            dtype=torch.long,
        )

        for i, word_labels in enumerate(batch_word_labels):
            if word_labels is None:
                continue

            previous_word_id = None
            word_ids = encoded.word_ids(batch_index=i)

            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    previous_word_id = word_id
                    continue

                if word_id != previous_word_id:
                    labels[i, token_idx] = word_labels[word_id]

                previous_word_id = word_id

        return labels
