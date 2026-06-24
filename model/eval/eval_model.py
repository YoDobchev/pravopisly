import pravopisly_pb2_grpc
import pravopisly_pb2
from pathlib import Path
import sys
import csv
import difflib

import grpc
import ollama
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_DIR = Path(__file__).resolve().parent.parent
PB_DIR = MODEL_DIR / "pb"

sys.path.append(str(MODEL_DIR))
sys.path.append(str(PB_DIR))


def apply_first_suggestions(text, suggestions, suggestion_type):
    chosen = []

    for suggestion in suggestions:
        if suggestion.type != suggestion_type:
            continue

        if not suggestion.replacements:
            continue

        chosen.append(suggestion)

    chosen.sort(
        key=lambda s: (s.start_index, s.end_index),
        reverse=True,
    )

    for suggestion in chosen:
        text = (
            text[:suggestion.start_index]
            + suggestion.replacements[0]
            + text[suggestion.end_index:]
        )

    return text


def send_to_pravopisly(text, host="localhost:50051"):
    with grpc.insecure_channel(host) as channel:
        stub = pravopisly_pb2_grpc.PravopislyCommsStub(channel)
        reply = stub.SendText(pravopisly_pb2.SendToModel(text=text))

    return reply.suggestions


def correct_with_pravopisly(text, host="localhost:50051"):
    suggestions = send_to_pravopisly(text, host)
    text = apply_first_suggestions(
        text,
        suggestions,
        pravopisly_pb2.SPELLING,
    )

    suggestions = send_to_pravopisly(text, host)
    text = apply_first_suggestions(
        text,
        suggestions,
        pravopisly_pb2.COMMA,
    )

    suggestions = send_to_pravopisly(text, host)
    text = apply_first_suggestions(
        text,
        suggestions,
        pravopisly_pb2.GRAMMAR,
    )

    return text


def clean_bggpt_output(text):
    text = str(text).strip()

    prefixes = [
        "Поправено изречение:",
        "Поправеното изречение е:",
        "Коригирано изречение:",
        "Отговор:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = text.strip()
    text = text.strip("`")
    text = text.strip()
    text = text.strip('"')
    text = text.strip("'")
    text = text.strip("„")
    text = text.strip("“")

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if lines:
        text = lines[0]

    return text.strip()


def correct_with_bggpt(
    text,
    model="todorov/bggpt:Gemma-3-4B-IT-Q4_K_M",
):
    prompt = (
        "Поправи правописните, пунктуационните и граматическите грешки "
        "в следното българско изречение. "
        "Върни само поправеното изречение, без обяснения.\n\n"
        f"{text}"
    )

    result = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        options={
            "temperature": 0,
            "num_predict": 128,
        },
    )

    if isinstance(result, dict):
        output = result["response"]
    else:
        output = result.response

    return clean_bggpt_output(output)


class Mt5Corrector:
    def __init__(self, model_path):
        if model_path is None:
            raise ValueError("mT5 model path is required")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def correct(self, text):
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
            num_beams=4,
        )

        corrected = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return corrected.strip()


def normalize(text):
    return " ".join(str(text).strip().split())


def similarity(a, b):
    return difflib.SequenceMatcher(
        None,
        normalize(a),
        normalize(b),
    ).ratio()


def compare_models(
    eval_csv_path="./eval_sen.csv",
    models=None,
    limit=None,
):
    if not models:
        raise ValueError("models is required")

    stats = {}

    for name, _ in models:
        stats[name] = {
            "exact": 0,
            "similarity_total": 0,
        }

    count = 0

    with open(eval_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if limit is not None and count >= limit:
                break

            erroneous = row["erroneous"]
            correct = row["correct"]
            correct_norm = normalize(correct)

            outputs = {}

            for name, corrector in models:
                output = corrector(erroneous)
                output_norm = normalize(output)

                is_exact = output_norm == correct_norm
                score = similarity(output, correct)

                if is_exact:
                    stats[name]["exact"] += 1

                stats[name]["similarity_total"] += score

                outputs[name] = {
                    "text": output,
                    "exact": is_exact,
                    "similarity": score,
                }

            count += 1

            print("=" * 80)
            print(f"Example {count}")
            print()
            print("Erroneous:")
            print(erroneous)
            print()
            print("Correct:")
            print(correct)
            print()

            for name, data in outputs.items():
                print(f"{name}:")
                print(data["text"])
                print(f"Exact: {data['exact']}")
                print(f"Similarity: {data['similarity']:.4f}")
                print()

    print("=" * 80)
    print("Summary")
    print(f"Rows: {count}")

    if count == 0:
        return

    print()
    print("Exact match")

    for name in stats:
        print(f"{name}: {stats[name]['exact'] / count:.4f}")

    print()
    print("Average similarity")

    for name in stats:
        print(f"{name}: {stats[name]['similarity_total'] / count:.4f}")


if __name__ == "__main__":
    mt5 = Mt5Corrector("./mt5")

    compare_models(
        eval_csv_path="./eval_sen.csv",
        models=[
            (
                "pravopisly",
                lambda text: correct_with_pravopisly(
                    text,
                    host="localhost:50051",
                ),
            ),
            (
                "bggpt",
                correct_with_bggpt,
            ),
            (
                "mt5",
                mt5.correct,
            ),
        ],
        limit=30,
    )
