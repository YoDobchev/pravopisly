from transformers.utils import logging
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from dataset import PravopislyDataset
from train import PravopislyBERTModel, train
from dotenv import load_dotenv
import os

logging.set_verbosity_error()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

if __name__ == "__main__":
    load_dotenv()
    data_folder = os.getenv("DATAFOLDER")
    assert data_folder != None

    tokenizer = AutoTokenizer.from_pretrained("rmihaylov/bert-base-bg")

    dataset = PravopislyDataset(
        jsonl_path=f"{data_folder}/dataset.jsonl",
        tokenizer=tokenizer,
        max_length=32,
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=24,
        shuffle=False,
    )

    model = PravopislyBERTModel(
        encoder_name="rmihaylov/bert-base-bg",
    ).to(device)

    train(model, train_loader, test_loader, device,
          what_to_train="comma_head", epochs=3)

    train(model, train_loader, test_loader, device,
          what_to_train="grammar_head", epochs=3)

    train(model, train_loader, test_loader, device,
          what_to_train="all", epochs=5)

    SAVE_DIR = "checkpoints/pravopisly_model"
    os.makedirs(SAVE_DIR, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "encoder_name": "rmihaylov/bert-base-bg",
                "num_comma_labels": 2,
                "num_spelling_labels": 2,
                "num_grammar_labels": 2,
            },
        },
        f"{SAVE_DIR}/model.pt",
    )

    tokenizer.save_pretrained(SAVE_DIR)

    print(f"Saved model to {SAVE_DIR}")
