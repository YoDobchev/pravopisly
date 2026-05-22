import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from tqdm import tqdm


class PravopislyBERTModel(nn.Module):
    def __init__(
        self,
        encoder_name="rmihaylov/bert-base-bg",
        num_comma_labels=2,
        num_spelling_labels=2,
        dropout=0.2,
    ):
        super().__init__()

        self.bert: AutoModel = AutoModel.from_pretrained(encoder_name)

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.comma_head = nn.Linear(hidden_size, num_comma_labels)
        self.spelling_head = nn.Linear(hidden_size, num_spelling_labels)

        self.what_to_train = "all"

    def _loss_for_head(self, logits, labels, num_labels, weight=None):
        if labels is None:
            return logits.new_tensor(0.0)

        if not (labels != -100).any():
            return logits.new_tensor(0.0)

        return F.cross_entropy(
            logits.reshape(-1, num_labels),
            labels.reshape(-1),
            ignore_index=-100,
            weight=weight,
        )

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def train_only_commas_head(self):
        self.freeze_bert()

        for p in self.comma_head.parameters():
            p.requires_grad = True

        for p in self.spelling_head.parameters():
            p.requires_grad = False

        self.what_to_train = "comma_head"

    def train_only_spelling_head(self):
        self.freeze_bert()

        for p in self.comma_head.parameters():
            p.requires_grad = False

        for p in self.spelling_head.parameters():
            p.requires_grad = True

        self.what_to_train = "spelling_head"

    def train_everything(self):
        self.unfreeze_bert()

        for p in self.comma_head.parameters():
            p.requires_grad = True

        for p in self.spelling_head.parameters():
            p.requires_grad = True

        self.what_to_train = "all"

    def forward(
        self,
        input_ids,
        attention_mask,
        comma_labels=None,
        spelling_labels=None,
        comma_weight=None,
        spelling_weight=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        x = outputs.last_hidden_state
        x = self.dropout(x)

        comma_logits = self.comma_head(x)
        spelling_logits = self.spelling_head(x)

        comma_loss = (
            self._loss_for_head(
                comma_logits,
                comma_labels,
                self.comma_head.out_features,
                weight=comma_weight,
            )
            if self.what_to_train in ["comma_head", "all"]
            else comma_logits.new_tensor(0.0)
        )

        spelling_loss = (
            self._loss_for_head(
                spelling_logits,
                spelling_labels,
                self.spelling_head.out_features,
                weight=spelling_weight,
            )
            if self.what_to_train in ["spelling_head", "all"]
            else spelling_logits.new_tensor(0.0)
        )

        loss = comma_loss + spelling_loss

        return {
            "loss": loss,
            "comma_logits": comma_logits,
            "spelling_logits": spelling_logits,
            "comma_loss": comma_loss.detach(),
            "spelling_loss": spelling_loss.detach(),
        }


def train_step(model, loader, optimizer, device, comma_weight=None, spelling_weight=None):
    model.train()

    total_loss = 0
    total_comma_loss = 0
    total_spelling_loss = 0
    steps = 0

    if comma_weight is not None:
        comma_weight = comma_weight.to(device)

    if spelling_weight is not None:
        spelling_weight = spelling_weight.to(device)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        comma_labels = batch["comma_labels"].to(device)
        spelling_labels = batch["spelling_labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            comma_labels=comma_labels,
            spelling_labels=spelling_labels,
            comma_weight=comma_weight,
            spelling_weight=spelling_weight,
        )

        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)

        optimizer.step()

        total_loss += loss.item()
        total_comma_loss += outputs["comma_loss"].item()
        total_spelling_loss += outputs["spelling_loss"].item()
        steps += 1

    return {
        "loss": total_loss / steps,
        "comma_loss": total_comma_loss / steps,
        "spelling_loss": total_spelling_loss / steps,
    }


@torch.inference_mode()
def test_step(model, loader, device):
    model.eval()

    total_loss = 0
    total_comma_loss = 0
    total_spelling_loss = 0
    steps = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        comma_labels = batch["comma_labels"].to(device)
        spelling_labels = batch["spelling_labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            comma_labels=comma_labels,
            spelling_labels=spelling_labels,
        )

        total_loss += outputs["loss"].item()
        total_comma_loss += outputs["comma_loss"].item()
        total_spelling_loss += outputs["spelling_loss"].item()
        steps += 1

    return {
        "loss": total_loss / steps,
        "comma_loss": total_comma_loss / steps,
        "spelling_loss": total_spelling_loss / steps,
    }


def train(model: PravopislyBERTModel, train_loader, test_loader, device, epochs=3, what_to_train="all"):
    model.to(device)
    if what_to_train == "all":
        model.train_everything()

        optimizer = torch.optim.AdamW(
            [
                {"params": model.bert.parameters(), "lr": 2e-5},
                {"params": model.comma_head.parameters(), "lr": 1e-4},
                {"params": model.spelling_head.parameters(), "lr": 1e-4},
            ]
        )

    elif what_to_train == "comma_head":
        model.train_only_commas_head()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )

    elif what_to_train == "spelling_head":
        model.train_only_spelling_head()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )
    else:
        raise ValueError(
            "invalid 'what_to_train'")

    comma_weight = torch.tensor([1.0, 3.0])
    spelling_weight = torch.tensor([1.0, 2.0])

    print(f"training {what_to_train} for {epochs} epochs...")
    for epoch in tqdm(range(epochs)):
        train_metrics = train_step(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            comma_weight=comma_weight,
            spelling_weight=spelling_weight,
            device=device,
        )

        test_metrics = test_step(
            model=model,
            loader=test_loader,
            device=device,
        )

        print(
            f"Epoch {epoch + 1} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_comma_loss={train_metrics['comma_loss']:.4f} | "
            f"train_spelling_loss={train_metrics['spelling_loss']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_comma_loss={test_metrics['comma_loss']:.4f} | "
            f"test_spelling_loss={test_metrics['spelling_loss']:.4f}"
        )
