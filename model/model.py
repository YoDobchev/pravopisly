import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PravopislyBERTModel(nn.Module):
    def __init__(
        self,
        encoder_name="rmihaylov/bert-base-bg",
        num_comma_labels=2,
        num_spelling_labels=2,
        num_grammar_labels=2,
        dropout=0.3,
        tokenizer_len=None,
    ):
        super().__init__()

        self.bert: AutoModel = AutoModel.from_pretrained(encoder_name)

        if tokenizer_len is not None:
            self.bert.resize_token_embeddings(tokenizer_len)

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.comma_head = nn.Linear(hidden_size, num_comma_labels)
        self.spelling_head = nn.Linear(hidden_size, num_spelling_labels)
        self.grammar_head = nn.Linear(hidden_size, num_grammar_labels)

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

        for p in self.grammar_head.parameters():
            p.requires_grad = False

        self.what_to_train = "comma_head"

    def train_only_spelling_head(self):
        self.freeze_bert()

        for p in self.comma_head.parameters():
            p.requires_grad = False

        for p in self.spelling_head.parameters():
            p.requires_grad = True

        for p in self.grammar_head.parameters():
            p.requires_grad = False

        self.what_to_train = "spelling_head"

    def train_only_grammar_head(self):
        self.freeze_bert()

        for p in self.comma_head.parameters():
            p.requires_grad = False

        for p in self.spelling_head.parameters():
            p.requires_grad = False

        for p in self.grammar_head.parameters():
            p.requires_grad = True

        self.what_to_train = "grammar_head"

    def train_everything(self):
        self.unfreeze_bert()

        for p in self.comma_head.parameters():
            p.requires_grad = True

        for p in self.spelling_head.parameters():
            p.requires_grad = True

        for p in self.grammar_head.parameters():
            p.requires_grad = True
        self.what_to_train = "all"

    def forward(
        self,
        input_ids,
        attention_mask,
        comma_labels=None,
        spelling_labels=None,
        grammar_labels=None,
        comma_weight=None,
        spelling_weight=None,
        grammar_weight=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        x = outputs.last_hidden_state
        x = self.dropout(x)

        comma_logits = self.comma_head(x)
        spelling_logits = self.spelling_head(x)
        grammar_logits = self.grammar_head(x)

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

        grammar_loss = (
            self._loss_for_head(
                grammar_logits,
                grammar_labels,
                self.grammar_head.out_features,
                weight=grammar_weight,
            )
            if self.what_to_train in ["grammar_head", "all"]
            else grammar_logits.new_tensor(0.0)
        )

        loss = comma_loss + spelling_loss + grammar_loss

        return {
            "loss": loss,
            "comma_logits": comma_logits,
            "spelling_logits": spelling_logits,
            "grammar_logits": grammar_logits,
            "comma_loss": comma_loss.detach(),
            "spelling_loss": spelling_loss.detach(),
            "grammar_loss": grammar_loss.detach(),
        }
