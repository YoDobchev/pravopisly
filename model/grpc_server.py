from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import grpc
from concurrent import futures
from model import PravopislyBERTModel
from pipelines.commas import append_comma_suggestions

# autopep8: off
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "pb"))
sys.path.append(str(Path(__file__).parent.parent / "data"))

from commas import sentence_to_word_labels
import pravopisly_pb2
import pravopisly_pb2_grpc
# autopep8: on


class PravopislyModel:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"using device: {self.device}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "checkpoints/pravopisly_model",
            use_fast=True,
        )

        checkpoint = torch.load(
            "checkpoints/pravopisly_model/model.pt",
            map_location="cpu",
            weights_only=False,
        )

        config = checkpoint["config"]

        self.model = PravopislyBERTModel(
            encoder_name=config["encoder_name"],
            num_comma_labels=config.get("num_comma_labels", 2),
            num_spelling_labels=config.get("num_spelling_labels", 2),
            num_grammar_labels=config.get("num_grammar_labels", 2),
            tokenizer_len=len(self.tokenizer),
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str):
        clean = sentence_to_word_labels(text)[0]
        words = clean.split()

        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

        word_ids = encoded.word_ids(batch_index=0)

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        comma_probs_by_token = F.softmax(
            outputs["comma_logits"],
            dim=-1,
        )[0]

        grammar_probs_by_token = F.softmax(
            outputs["grammar_logits"],
            dim=-1,
        )[0]

        spelling_probs_by_token = F.softmax(
            outputs["spelling_logits"],
            dim=-1,
        )[0]

        comma_probs = []
        grammar_probs = []
        spelling_probs = []

        seen_word_ids = set()

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            if word_id in seen_word_ids:
                continue

            seen_word_ids.add(word_id)

            comma_prob = comma_probs_by_token[token_idx, 1].item()
            grammar_prob = grammar_probs_by_token[token_idx, 1].item()
            spelling_prob = spelling_probs_by_token[token_idx, 1].item()

            comma_probs.append(comma_prob)
            grammar_probs.append(grammar_prob)
            spelling_probs.append(spelling_prob)

        return {
            "clean_text": clean,
            "words": words,
            "comma_probs": comma_probs,
            "grammar_probs": grammar_probs,
            "spelling_probs": spelling_probs,
        }


class PravopislyServer(pravopisly_pb2_grpc.PravopislyCommsServicer):
    def __init__(self, model):
        self.model = model

    def SendText(self, request, context):
        text = request.text

        print("rec:", text, flush=True)

        predictions = self.model.predict(text)

        suggestions = []

        append_comma_suggestions(suggestions, text, predictions["comma_probs"])

        return pravopisly_pb2.ModelReply(
            suggestions=suggestions
        )


if __name__ == "__main__":
    model = PravopislyModel()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    pravopisly_pb2_grpc.add_PravopislyCommsServicer_to_server(
        PravopislyServer(model),
        server,
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("grpc server running on port 50051")

    server.wait_for_termination()
