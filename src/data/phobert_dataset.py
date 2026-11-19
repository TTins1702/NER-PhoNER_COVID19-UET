import json
import os
import torch
from torch.utils.data import Dataset

ENTITY_TYPES = [
    "AGE",
    "DATE",
    "GENDER",
    "JOB",
    "LOCATION",
    "NAME",
    "ORGANIZATION",
    "PATIENT_ID",
    "SYMPTOM_AND_DISEASE",
    "TRANSPORTATION",
]

DEFAULT_LABELS = ["O"] + [
    f"{prefix}-{entity}" for entity in ENTITY_TYPES for prefix in ("B", "I")
]
DEFAULT_LABEL2ID = {label: idx for idx, label in enumerate(DEFAULT_LABELS)}
DEFAULT_ID2LABEL = {idx: label for idx, label in enumerate(DEFAULT_LABELS)}


def get_label_maps():
    try:
        from src.data.labels import LABEL2ID, ID2LABEL

        if LABEL2ID and ID2LABEL:
            return LABEL2ID, ID2LABEL
    except ImportError:
        pass
    return DEFAULT_LABEL2ID, DEFAULT_ID2LABEL


def inspect_bpe_sample(tokenizer, words):
    return [(word, tokenizer.tokenize(word)) for word in words]


class PhoBERTDataset(Dataset):
    def __init__(
        self,
        data_or_path,
        tokenizer,
        label2id=None,
        max_seq_length=256,
        ignore_index=-100,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id if label2id is not None else get_label_maps()[0]
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index
        self.samples = []

        if isinstance(data_or_path, (str, os.PathLike)):
            with open(data_or_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
        else:
            self.samples = list(data_or_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        words = sample["words"]
        tags = sample.get("tags")
        has_tags = tags is not None and len(tags) > 0

        input_ids = [self.tokenizer.cls_token_id]
        labels = [self.ignore_index] if has_tags else None
        word_mask = [False]
        max_content_len = self.max_seq_length - 2

        for idx_word, word in enumerate(words):
            subword_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not subword_ids:
                subword_ids = [self.tokenizer.unk_token_id]

            if len(input_ids) - 1 + len(subword_ids) > max_content_len:
                break

            input_ids.extend(subword_ids)
            if labels is not None:
                labels.append(self.label2id[tags[idx_word]])
                labels.extend([self.ignore_index] * (len(subword_ids) - 1))
            word_mask.append(True)
            word_mask.extend([False] * (len(subword_ids) - 1))

        input_ids.append(self.tokenizer.sep_token_id)
        if labels is not None:
            labels.append(self.ignore_index)
        word_mask.append(False)

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
            "word_mask": torch.tensor(word_mask, dtype=torch.bool),
            "words": list(words),
            "gold_tags": list(tags) if has_tags else [],
            "orig_length": len(words),
        }

    def get_truncation_stats(self):
        max_content_len = self.max_seq_length - 2
        truncated = 0
        for sample in self.samples:
            total_subwords = 0
            for word in sample["words"]:
                subword_ids = self.tokenizer.encode(word, add_special_tokens=False)
                total_subwords += len(subword_ids) if subword_ids else 1
            if total_subwords > max_content_len:
                truncated += 1
        return {"total": len(self.samples), "truncated": truncated}


def phobert_collate_fn(batch, pad_token_id=1, ignore_index=-100):
    max_len = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)
    has_labels = all(item.get("labels") is not None for item in batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = (
        torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
        if has_labels
        else None
    )
    word_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        if labels is not None and item["labels"] is not None:
            labels[i, :seq_len] = item["labels"]
        word_mask[i, :seq_len] = item["word_mask"]

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "word_mask": word_mask,
    }
    if "words" in batch[0]:
        collated["words"] = [item["words"] for item in batch]
    if "gold_tags" in batch[0]:
        collated["gold_tags"] = [item["gold_tags"] for item in batch]
    if "orig_length" in batch[0]:
        collated["orig_lengths"] = [item["orig_length"] for item in batch]
    return collated


def extract_word_level_labels(labels, word_mask, id2label=None):
    if id2label is None:
        id2label = get_label_maps()[1]
    return [
        [id2label[int(t)] for t in labels[i][word_mask[i]]]
        for i in range(labels.size(0))
    ]


def decode_word_level_predictions(pred_ids, word_mask=None, id2label=None, pad_tag_id=0):
    if id2label is None:
        id2label = get_label_maps()[1]
    if isinstance(pred_ids, torch.Tensor):
        return [
            [id2label[int(p)] for p in pred_ids[i][word_mask[i]]]
            for i in range(pred_ids.size(0))
        ]
    if word_mask is not None and not isinstance(word_mask, torch.Tensor):
        pad_tag = id2label[int(pad_tag_id)]
        decoded = []
        for seq, length in zip(pred_ids, word_mask):
            target_len = int(length)
            tags = [id2label[int(p)] for p in seq[:target_len]]
            if len(tags) < target_len:
                tags.extend([pad_tag] * (target_len - len(tags)))
            decoded.append(tags)
        return decoded
    return [[id2label[int(p)] for p in seq] for seq in pred_ids]
