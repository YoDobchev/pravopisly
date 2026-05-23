import json
import os
import torch
from torch.utils.data import Dataset


class PravopislyDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=64,
        cache_path=None,
        rebuild_cache=False,
        batch_size=2048,
    ):
        if cache_path is None:
            cache_path = f"{jsonl_path}.maxlen{max_length}.cache.pt"

        if os.path.exists(cache_path) and not rebuild_cache:
            print(f"Loading dataset cache: {cache_path}", flush=True)
            self.examples = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=False,
            )
            print(
                f"Loaded {len(self.examples)} examples from cache", flush=True)
            return

        print(f"Building dataset from JSONL: {jsonl_path}", flush=True)
        self.examples = []

        batch_texts = []
        batch_words = []
        batch_comma_labels = []
        batch_spelling_labels = []
        batch_grammar_labels = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...", flush=True)

                row = json.loads(line)

                text = row["s"]
                words = text.split()

                comma_labels = row.get("c")
                spelling_labels = row.get("sp")
                grammar_labels = row.get("g")

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

                if grammar_labels is not None and len(grammar_labels) != len(words):
                    raise ValueError(
                        f"grammar label length mismatch: "
                        f"{len(grammar_labels)} labels vs {len(words)} words\n{text}"
                    )

                batch_texts.append(text)
                batch_words.append(words)
                batch_comma_labels.append(comma_labels)
                batch_spelling_labels.append(spelling_labels)
                batch_grammar_labels.append(grammar_labels)

                if len(batch_words) >= batch_size:
                    self.process_batch(
                        tokenizer=tokenizer,
                        batch_texts=batch_texts,
                        batch_words=batch_words,
                        batch_comma_labels=batch_comma_labels,
                        batch_spelling_labels=batch_spelling_labels,
                        batch_grammar_labels=batch_grammar_labels,
                        max_length=max_length,
                    )

                    batch_texts = []
                    batch_words = []
                    batch_comma_labels = []
                    batch_spelling_labels = []
                    batch_grammar_labels = []

        if len(batch_words) > 0:
            self.process_batch(
                tokenizer=tokenizer,
                batch_texts=batch_texts,
                batch_words=batch_words,
                batch_comma_labels=batch_comma_labels,
                batch_spelling_labels=batch_spelling_labels,
                batch_grammar_labels=batch_grammar_labels,
                max_length=max_length,
            )

        print(f"Saving dataset cache: {cache_path}", flush=True)
        tmp_cache_path = f"{cache_path}.tmp"
        torch.save(self.examples, tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)

        print(f"Saved {len(self.examples)} examples to cache", flush=True)

    def process_batch(
        self,
        tokenizer,
        batch_texts,
        batch_words,
        batch_comma_labels,
        batch_spelling_labels,
        batch_grammar_labels,
        max_length,
    ):
        encoded = tokenizer(
            batch_words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        for i in range(len(batch_words)):
            word_ids = encoded.word_ids(batch_index=i)

            aligned_comma_labels = self.align_labels(
                word_ids,
                batch_comma_labels[i],
            )

            aligned_spelling_labels = self.align_labels(
                word_ids,
                batch_spelling_labels[i],
            )

            aligned_grammar_labels = self.align_labels(
                word_ids,
                batch_grammar_labels[i],
            )

            self.examples.append({
                "input_ids": encoded["input_ids"][i],
                "attention_mask": encoded["attention_mask"][i],
                "comma_labels": torch.tensor(aligned_comma_labels),
                "spelling_labels": torch.tensor(aligned_spelling_labels),
                "grammar_labels": torch.tensor(aligned_grammar_labels),
                "text": batch_texts[i],
                "words": batch_words[i],
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
            "grammar_labels": item["grammar_labels"],
        }
