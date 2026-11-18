from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.labels import IGNORE_INDEX, encode_tags
from src.data.vocab import PAD_IDX, CharVocab, WordVocab


class BiLSTMDataset(Dataset):
    """PyTorch Dataset preparing word and character indices for BiLSTM-CRF."""

    def __init__(
        self,
        samples: list[dict[str, list[str]]],
        word_vocab: WordVocab,
        char_vocab: CharVocab,
        max_word_len: int = 30,
    ) -> None:
        self.samples = samples
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_word_len = max_word_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        words = sample["words"]
        tags = sample.get("tags", [])
        return {
            "word_ids": self.word_vocab.encode(words),
            "char_ids": [self.char_vocab.encode_word(w, max_len=self.max_word_len) for w in words],
            "tag_ids": encode_tags(tags) if tags else None,
            "words": list(words),
            "gold_tags": list(tags),
        }


def bilstm_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Dynamically pad word sequences [B, L] and character sequences [B, L, W]."""
    batch_size = len(batch)
    max_seq_len = max((len(item["word_ids"]) for item in batch), default=1)
    max_seq_len = max(1, max_seq_len)

    max_word_len = max(
        (len(chars) for item in batch for chars in item["char_ids"]),
        default=1,
    )
    max_word_len = max(1, max_word_len)

    word_ids = torch.full((batch_size, max_seq_len), PAD_IDX, dtype=torch.long)
    char_ids = torch.full((batch_size, max_seq_len, max_word_len), PAD_IDX, dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    has_tags = all(item["tag_ids"] is not None for item in batch)
    tags = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long) if has_tags else None

    words_list: list[list[str]] = []
    gold_tags_list: list[list[str]] = []

    for b, item in enumerate(batch):
        seq_len = len(item["word_ids"])
        if seq_len > 0:
            word_ids[b, :seq_len] = torch.tensor(item["word_ids"], dtype=torch.long)
            mask[b, :seq_len] = True
            if tags is not None and item["tag_ids"] is not None:
                tags[b, :seq_len] = torch.tensor(item["tag_ids"], dtype=torch.long)
            for l_idx, c_ids in enumerate(item["char_ids"]):
                if c_ids:
                    char_ids[b, l_idx, : len(c_ids)] = torch.tensor(c_ids, dtype=torch.long)

        words_list.append(item["words"])
        gold_tags_list.append(item["gold_tags"])

    return {
        "word_ids": word_ids,
        "char_ids": char_ids,
        "mask": mask,
        "tags": tags,
        "words": words_list,
        "gold_tags": gold_tags_list,
    }
