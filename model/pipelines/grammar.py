import re
import difflib

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import pb.pravopisly_pb2 as pravopisly_pb2


WORD_RE = re.compile(r"\S+")


class Mt5GrammarCorrector:
    def __init__(self, model_path: str, device):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def correct(self, text: str) -> str:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            num_beams=2,
        )

        corrected = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return corrected.strip()


def has_confident_grammar_error(grammar_probs, min_confidence: float):
    return any(float(prob) >= min_confidence for prob in grammar_probs)


def append_grammar_suggestions(
    corrector,
    suggestions: list,
    original_text: str,
    grammar_probs,
    min_confidence: float = 0.80,
):
    if not has_confident_grammar_error(grammar_probs, min_confidence):
        return

    corrected_text = corrector.correct(original_text)

    if not corrected_text:
        return

    if corrected_text == original_text.strip():
        return

    print(f"correct: {corrected_text}")

    original_matches = list(WORD_RE.finditer(original_text))
    original_words = [match.group() for match in original_matches]
    corrected_words = corrected_text.split()

    if not original_words or not corrected_words:
        return

    matcher = difflib.SequenceMatcher(
        None,
        original_words,
        corrected_words,
    )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        replacement = " ".join(corrected_words[j1:j2])

        if tag in {"replace", "delete"}:
            start_index = original_matches[i1].start()
            end_index = original_matches[i2 - 1].end()

        elif tag == "insert":
            if i1 == 0:
                start_index = 0
                end_index = 0
                replacement = replacement + " "
            else:
                start_index = original_matches[i1 - 1].end()
                end_index = start_index
                replacement = " " + replacement

        else:
            continue

        suggestions.append(
            pravopisly_pb2.TextSuggestion(
                type=pravopisly_pb2.GRAMMAR,
                start_index=start_index,
                end_index=end_index,
                replacements=[replacement],
            )
        )
