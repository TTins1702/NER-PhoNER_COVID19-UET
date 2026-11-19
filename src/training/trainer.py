from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.bilstm_dataset import BiLSTMDataset, bilstm_collate_fn
from src.data.labels import LABELS, decode_tags
from src.data.vocab import PAD_IDX, CharVocab, WordVocab
from src.evaluation.metrics import compute_ner_metrics
from src.models.bilstm_crf import BiLSTMCRF
from src.utils.logger import ExperimentLogger


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict[str, Any],
    total_steps: int,
) -> tuple[torch.optim.Optimizer, Any | None]:
    """Build optimizer (and optional linear warmup scheduler) for BiLSTM-CRF or PhoBERT."""
    model_name = config.get("model", {}).get("name", "bilstm_crf")
    train_cfg = config.get("training", {})

    if model_name == "bilstm_crf":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        return optimizer, None

    if model_name == "phobert":
        from transformers import get_linear_schedule_with_warmup

        backbone_lr = float(train_cfg.get("backbone_lr", 2e-5))
        head_lr = float(train_cfg.get("head_lr", 1e-3))
        weight_decay = float(train_cfg.get("weight_decay", 0.01))

        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})

        optimizer = torch.optim.AdamW(param_groups)
        warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
        )
        return optimizer, scheduler

    raise ValueError(f"Unsupported model.name: {model_name!r}")


def build_model_and_dataloaders(
    config: dict[str, Any],
    splits: dict[str, list[dict[str, list[str]]]],
    vocab_state: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, DataLoader], dict[str, Any] | None]:
    """Construct model, DataLoaders for provided splits, and serializable vocab state."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model_name = model_cfg.get("name", "bilstm_crf")
    batch_size = int(train_cfg.get("batch_size", 32))

    if model_name == "bilstm_crf":
        if vocab_state is not None:
            word_vocab = WordVocab.from_dict(vocab_state["word_vocab"])
            char_vocab = CharVocab.from_dict(vocab_state["char_vocab"])
        else:
            if "train" not in splits:
                raise ValueError("Cannot build vocabularies without 'train' split or saved vocab_state")
            train_words = [s["words"] for s in splits["train"]]
            word_vocab = WordVocab.build(train_words, min_freq=int(data_cfg.get("min_word_freq", 1)))
            char_vocab = CharVocab.build(train_words)
            vocab_state = {
                "word_vocab": word_vocab.to_dict(),
                "char_vocab": char_vocab.to_dict(),
            }

        max_word_len = int(data_cfg.get("max_word_len", 30))
        loaders: dict[str, DataLoader] = {}
        for split_name, records in splits.items():
            dataset = BiLSTMDataset(records, word_vocab, char_vocab, max_word_len=max_word_len)
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                collate_fn=bilstm_collate_fn,
            )

        model = BiLSTMCRF(
            vocab_size=len(word_vocab),
            char_vocab_size=len(char_vocab),
            num_tags=len(LABELS),
            word_emb_dim=int(model_cfg.get("word_emb_dim", 100)),
            use_char=bool(model_cfg.get("use_char", True)),
            char_encoder_type=str(model_cfg.get("char_encoder_type", "cnn")),
            char_emb_dim=int(model_cfg.get("char_emb_dim", 30)),
            char_out_dim=int(model_cfg.get("char_out_dim", 50)),
            char_kernel_size=int(model_cfg.get("char_kernel_size", 3)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            use_crf=bool(model_cfg.get("use_crf", True)),
            pad_idx=PAD_IDX,
        )
        return model, loaders, vocab_state

    if model_name == "phobert":
        from functools import partial
        from transformers import AutoTokenizer
        from src.data.phobert_dataset import PhoBERTDataset, phobert_collate_fn
        from src.models.phobert_ner import PhoBERTForNER

        pretrained_name = str(model_cfg.get("pretrained_name", "vinai/phobert-base-v2"))
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        max_seq_length = int(data_cfg.get("max_seq_length", 256))
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

        loaders = {}
        for split_name, records in splits.items():
            dataset = PhoBERTDataset(records, tokenizer=tokenizer, max_seq_length=max_seq_length)
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                collate_fn=partial(phobert_collate_fn, pad_token_id=pad_id),
            )

        model = PhoBERTForNER(
            model_name_or_path=pretrained_name,
            num_tags=len(LABELS),
            feature_mode=str(model_cfg.get("feature_mode", "last_layer")),
            use_crf=bool(model_cfg.get("use_crf", False)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        return model, loaders, None

    raise ValueError(f"Unsupported model.name: {model_name!r}")


class Trainer:
    """Unified training, validation, checkpointing, and evaluation loop."""

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device,
        logger: ExperimentLogger | None = None,
        vocab_state: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        self.vocab_state = vocab_state
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None

    def _forward_batch(self, batch: dict[str, Any], decode: bool = True) -> dict[str, Any]:
        if "word_ids" in batch:
            tags = batch["tags"].to(self.device) if batch.get("tags") is not None else None
            return self.model(
                word_ids=batch["word_ids"].to(self.device),
                char_ids=batch["char_ids"].to(self.device),
                mask=batch["mask"].to(self.device),
                tags=tags,
                decode=decode,
            )
        if "input_ids" in batch:
            labels = batch["labels"].to(self.device) if batch.get("labels") is not None else None
            return self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                word_mask=batch["word_mask"].to(self.device),
                labels=labels,
                decode=decode,
            )
        raise ValueError(f"Unrecognized batch keys: {list(batch.keys())}")

    def _decode_batch_predictions(
        self,
        raw_predictions: list[list[int]],
        batch: dict[str, Any],
    ) -> list[list[str]]:
        if "input_ids" in batch:
            from src.data.phobert_dataset import decode_word_level_predictions

            return decode_word_level_predictions(raw_predictions, batch["orig_lengths"])
        return [decode_tags(seq) for seq in raw_predictions]

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch and return mean batch loss."""
        if self.optimizer is None:
            total_steps = max(1, len(dataloader) * int(self.config.get("training", {}).get("epochs", 1)))
            self.optimizer, self.scheduler = build_optimizer_and_scheduler(
                self.model,
                self.config,
                total_steps=total_steps,
            )

        grad_clip = float(self.config.get("training", {}).get("grad_clip", 5.0))
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            out = self._forward_batch(batch, decode=False)
            loss = out["loss"]
            if loss is None:
                raise ValueError("Model returned loss=None during training")
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        """Evaluate model on a dataloader using entity-level strict seqeval."""
        self.model.eval()
        y_true: list[list[str]] = []
        y_pred: list[list[str]] = []

        for batch in dataloader:
            out = self._forward_batch(batch)
            preds = self._decode_batch_predictions(out["predictions"], batch)
            y_true.extend(batch["gold_tags"])
            y_pred.extend(preds)

        return compute_ner_metrics(y_true, y_pred)

    def fit(self, train_loader: DataLoader, dev_loader: DataLoader) -> dict[str, Any]:
        """Train across epochs with early stopping on dev micro-F1 and checkpointing."""
        train_cfg = self.config.get("training", {})
        epochs = int(train_cfg.get("epochs", 10))
        patience = int(train_cfg.get("patience", 5))
        ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = ckpt_dir / "best_model.pt"

        total_steps = max(1, len(train_loader) * epochs)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            self.model,
            self.config,
            total_steps=total_steps,
        )

        best_f1 = -1.0
        best_epoch = 0
        no_improve = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            dev_metrics = self.evaluate(dev_loader)
            dev_f1 = float(dev_metrics["micro_f1"])

            record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "dev_micro_f1": dev_f1,
                "dev_macro_f1": float(dev_metrics["macro_f1"]),
                "dev_bio_violations": float(dev_metrics["bio_violations"]),
            }
            history.append(record)
            if self.logger is not None:
                self.logger.log_metrics(record, step=epoch)

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": self.config,
                        "vocab_state": self.vocab_state,
                        "best_epoch": best_epoch,
                        "best_dev_f1": best_f1,
                    },
                    best_ckpt_path,
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

        return {
            "best_epoch": best_epoch,
            "best_dev_f1": best_f1,
            "checkpoint_path": str(best_ckpt_path),
            "history": history,
        }
