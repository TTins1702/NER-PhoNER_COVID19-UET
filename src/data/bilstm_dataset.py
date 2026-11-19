"""Dataset và collate_fn cho BiLSTM-CRF (hợp đồng giao diện: docs/interface_contract.md, mục a).

Tại sao viết như vậy:
- Mã hóa (từ -> id, ký tự -> id, nhãn -> id) làm **một lần** trong `__init__` thay vì mỗi lần
  `__getitem__`, để DataLoader nhanh hơn trên Kaggle/Colab.
- Padding làm **theo batch** trong `collate_fn` (không pad cả tập theo câu dài nhất 161 từ), nên tensor
  nhỏ và nhanh. `L` = câu dài nhất trong batch, `W` = từ dài nhất trong batch.
- `mask` kiểu bool, True ở từ thật: CRF dùng mask để (1) không cộng điểm của vị trí pad vào s(x, y),
  (2) giữ nguyên alpha khi đi qua vị trí pad trong thuật toán Forward, (3) dừng Viterbi đúng độ dài.
- Pad nhãn bằng 0 (`O`) chứ không phải -100 vì CRF phải `gather` theo nhãn ở mọi vị trí; vị trí pad đã
  bị mask loại khỏi loss. `lengths` để dùng `pack_padded_sequence` (BiLSTM không đọc token pad,
  nên trạng thái của luồng phải->trái không bị nhiễm pad).
- Không sắp xếp lại câu trong batch: dự đoán ghép lại đúng thứ tự file gốc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.download import DEFAULT_DATA_DIR, load_split
from src.data.labels import LABEL2ID, O_ID
from src.data.vocab import PAD_ID, CharVocab, WordVocab

DEFAULT_MAX_WORD_LEN = 30  # từ dài nhất trong train mức từ ~23 ký tự; 30 là ngưỡng an toàn
TAG_PAD_ID = O_ID          # = 0


class BiLSTMNERDataset(Dataset):
    """Mỗi phần tử là dict đã mã hóa của một câu.

    Phần tử `i` (trước khi collate):
        "word_ids": list[int]          độ dài n
        "char_ids": list[list[int]]    n danh sách, mỗi danh sách độ dài <= max_word_len
        "tag_ids":  list[int]          độ dài n (nếu dữ liệu có nhãn, ngược lại toàn 0)
        "words":    list[str]          từ gốc
    """

    def __init__(self, samples: Sequence[dict], word_vocab: WordVocab, char_vocab: CharVocab,
                 label2id: dict[str, int] = LABEL2ID, max_word_len: int | None = DEFAULT_MAX_WORD_LEN):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.items: list[dict[str, Any]] = []
        for s in samples:
            words = list(s["words"])
            tags = s.get("tags")
            tag_ids = [label2id[t] for t in tags] if tags is not None else [TAG_PAD_ID] * len(words)
            if len(tag_ids) != len(words):
                raise ValueError("Số nhãn khác số từ")
            self.items.append({
                "word_ids": word_vocab.encode(words),
                "char_ids": char_vocab.encode(words, max_len=max_word_len),
                "tag_ids": tag_ids,
                "words": words,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def bilstm_collate_fn(batch: Sequence[dict[str, Any]], word_pad_id: int = PAD_ID,
                      char_pad_id: int = PAD_ID, tag_pad_id: int = TAG_PAD_ID) -> dict[str, Any]:
    """Gộp một danh sách câu thành batch tensor đã pad.

    Trả về dict:
        word_ids : torch.LongTensor [B, L]     id từ, pad = 0
        char_ids : torch.LongTensor [B, L, W]  id ký tự, pad = 0
        mask     : torch.BoolTensor [B, L]     True = từ thật, False = pad
        tags     : torch.LongTensor [B, L]     id nhãn (LABEL2ID), pad = 0 (bỏ qua bằng mask)
        lengths  : torch.LongTensor [B]        số từ thật mỗi câu (CPU, cho pack_padded_sequence)
        words    : list[list[str]]             từ gốc
    với B = số câu, L = số từ của câu dài nhất trong batch, W = số ký tự của từ dài nhất trong batch.
    """
    if len(batch) == 0:
        raise ValueError("Batch rỗng")
    lengths = [len(item["word_ids"]) for item in batch]
    B, L = len(batch), max(lengths)
    W = max(len(c) for item in batch for c in item["char_ids"])

    word_ids = torch.full((B, L), word_pad_id, dtype=torch.long)
    char_ids = torch.full((B, L, W), char_pad_id, dtype=torch.long)
    tags = torch.full((B, L), tag_pad_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.bool)

    for b, item in enumerate(batch):
        n = lengths[b]
        word_ids[b, :n] = torch.tensor(item["word_ids"], dtype=torch.long)
        tags[b, :n] = torch.tensor(item["tag_ids"], dtype=torch.long)
        mask[b, :n] = True
        for t, chars in enumerate(item["char_ids"]):
            char_ids[b, t, : len(chars)] = torch.tensor(chars, dtype=torch.long)

    return {
        "word_ids": word_ids,
        "char_ids": char_ids,
        "mask": mask,
        "tags": tags,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "words": [item["words"] for item in batch],
    }


def build_vocabs(train_samples: Sequence[dict], min_freq: int = 1, lower: bool = False,
                 normalize_digits: bool = False, extend_chars: bool = True) -> tuple[WordVocab, CharVocab]:
    """Xây WordVocab + CharVocab **chỉ từ tập train**."""
    sentences = [s["words"] for s in train_samples]
    word_vocab = WordVocab.build(sentences, min_freq=min_freq, lower=lower,
                                 normalize_digits=normalize_digits)
    char_vocab = CharVocab.build(sentences, extend=extend_chars)
    return word_vocab, char_vocab


def build_bilstm_dataloaders(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    level: str = "word",
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    min_freq: int = 1,
    lower: bool = False,
    normalize_digits: bool = False,
    max_word_len: int | None = DEFAULT_MAX_WORD_LEN,
    num_workers: int = 0,
    seed: int = 42,
    splits: Sequence[str] = ("train", "dev", "test"),
) -> tuple[dict[str, DataLoader], WordVocab, CharVocab]:
    """Hàm tiện ích cho Tín: đọc dữ liệu, xây vocab từ train, trả về DataLoader cho từng tập.

    Chỉ tập train được xáo trộn (`shuffle=True`, có `generator` cố định seed để tái lập).
    Đổi `level="syllable"` để chạy thí nghiệm mức âm tiết.
    """
    data = {split: load_split(split, level, data_dir) for split in splits}
    if "train" not in data:
        data["train"] = load_split("train", level, data_dir)
    word_vocab, char_vocab = build_vocabs(data["train"], min_freq=min_freq, lower=lower,
                                          normalize_digits=normalize_digits)
    loaders: dict[str, DataLoader] = {}
    for split in splits:
        ds = BiLSTMNERDataset(data[split], word_vocab, char_vocab, max_word_len=max_word_len)
        is_train = split == "train"
        gen = torch.Generator().manual_seed(seed) if is_train else None
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if is_train else (eval_batch_size or batch_size),
            shuffle=is_train,
            collate_fn=bilstm_collate_fn,
            num_workers=num_workers,
            generator=gen,
        )
    return loaders, word_vocab, char_vocab
