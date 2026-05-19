import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PravopislyBERTModel(nn.Module):
    def __init__(
        self,
        encoder_name="rmihaylov/bert-base-bg",
        num_comma_labels=2,
        num_spelling_labels=2,
        dropout=0.2,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(encoder_name)

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.comma_head = nn.Linear(hidden_size, num_comma_labels)
        self.spelling_head = nn.Linear(hidden_size, num_spelling_labels)

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

        comma_loss = self._loss_for_head(
            comma_logits,
            comma_labels,
            self.comma_head.out_features,
            weight=comma_weight,
        )

        spelling_loss = self._loss_for_head(
            spelling_logits,
            spelling_labels,
            self.spelling_head.out_features,
            weight=spelling_weight,
        )

        loss = comma_loss + spelling_loss

        return {
            "loss": loss,
            "comma_logits": comma_logits,
            "spelling_logits": spelling_logits,
            "comma_loss": comma_loss.detach(),
            "spelling_loss": spelling_loss.detach(),
        }


def train_step(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    total_comma_loss = 0
    total_spelling_loss = 0
    steps = 0

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
        )

        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
