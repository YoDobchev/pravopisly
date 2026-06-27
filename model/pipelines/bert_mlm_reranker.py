import math
import torch
import torch.nn.functional as F


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)

            current.append(min(insert, delete, replace))

        previous = current

    return previous[-1]


class BertMlmReranker:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @torch.inference_mode()
    def score(self, original_text, start_index, end_index, candidate):
        text = original_text[:start_index] + \
            candidate + original_text[end_index:]

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=64,
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        candidate_start = start_index
        candidate_end = start_index + len(candidate)

        positions = []

        for token_idx, (char_start, char_end) in enumerate(offsets):
            if char_start == char_end:
                continue

            if char_start < candidate_end and char_end > candidate_start:
                positions.append(token_idx)

        if not positions:
            return float("-inf")

        scores = []

        for token_idx in positions:
            masked_input_ids = input_ids.clone()
            true_token_id = input_ids[0, token_idx].item()

            masked_input_ids[0, token_idx] = self.tokenizer.mask_token_id

            outputs = self.model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits[0, token_idx]
            log_probs = F.log_softmax(logits, dim=-1)

            scores.append(log_probs[true_token_id].item())

        return sum(scores) / len(scores)

    def rerank(
        self,
        original_text,
        start_index,
        end_index,
        candidates,
        original_word=None,
        candidate_counts=None,
    ):
        scored = []

        if original_word is None:
            original_word = original_text[start_index:end_index].lower()

        if candidate_counts is None:
            candidate_counts = {}

        for candidate in candidates:
            bert_score = self.score(
                original_text,
                start_index,
                end_index,
                candidate,
            )

            candidate_lower = candidate.lower()

            distance = edit_distance(original_word, candidate_lower)
            count = candidate_counts.get(candidate_lower, 1)

            frequency_bonus = math.log(count + 1)
            distance_penalty = distance * 1.5

            final_score = bert_score + 0.2 * frequency_bonus - distance_penalty

            scored.append((
                candidate,
                final_score,
            ))

        return sorted(scored, key=lambda x: x[1], reverse=True)
