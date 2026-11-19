"""Unit test cho src/data/vocab.py và src/data/bilstm_dataset.py."""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.bilstm_dataset import BiLSTMNERDataset, bilstm_collate_fn, build_vocabs
from src.data.labels import LABEL2ID
from src.data.vocab import PAD_ID, UNK_ID, CharVocab, WordVocab


def test_word_vocab_train_only_and_min_freq(tmp_path: Path) -> None:
    train_sents = [
        ["Bệnh_nhân", "BN17", "ở", "Hà_Nội"],
        ["Bệnh_nhân", "BN91", "ở", "Đà_Nẵng"],
    ]
    dev_sents = [["Bệnh_nhân", "BN416", "ở", "Vĩnh_Phúc"]]

    v1 = WordVocab.build(train_sents, min_freq=1)
    assert v1.pad_id == PAD_ID == 0
    assert v1.unk_id == UNK_ID == 1
    assert "Bệnh_nhân" in v1
    assert "Vĩnh_Phúc" not in v1
    assert v1.token_to_id("Vĩnh_Phúc") == UNK_ID
    assert abs(v1.oov_rate(dev_sents) - 0.5) < 1e-6  # 2/4 từ ("BN416", "Vĩnh_Phúc") là OOV

    # min_freq = 2 chỉ giữ "Bệnh_nhân" và "ở"
    v2 = WordVocab.build(train_sents, min_freq=2)
    assert len(v2) == 2 + 2  # <pad>, <unk>, Bệnh_nhân, ở
    assert "BN17" not in v2

    # Lưu và nạp lại JSON
    vocab_file = tmp_path / "word_vocab.json"
    v1.save(vocab_file)
    v1_loaded = WordVocab.load(vocab_file)
    assert v1_loaded.itos == v1.itos


def test_char_vocab_vietnamese_diacritics() -> None:
    cv = CharVocab.build([["Bệnh_viện", "Đà_Nẵng", "27/7"]])
    assert cv.pad_id == 0
    assert cv.unk_id == 1
    for ch in ("ệ", "Đ", "ẵ", "_", "7", "/"):
        assert ch in cv
        assert cv.stoi[ch] != UNK_ID
    ids = cv.encode_word("Bệnh", max_len=3)
    assert len(ids) == 3
    assert cv.encode_word("") == [UNK_ID]


def test_bilstm_dataset_and_collate_fn_contract() -> None:
    samples = [
        {"words": ["Bệnh_nhân", "BN17", "sốt", "."],
         "tags": ["O", "B-PATIENT_ID", "B-SYMPTOM_AND_DISEASE", "O"]},
        {"words": ["Bộ", "Y_tế"],
         "tags": ["B-ORGANIZATION", "I-ORGANIZATION"]},
    ]
    wv, cv = build_vocabs(samples, min_freq=1)
    ds = BiLSTMNERDataset(samples, wv, cv, max_word_len=20)
    assert len(ds) == 2

    batch = bilstm_collate_fn([ds[0], ds[1]])
    B, L = 2, 4
    W = len("Bệnh_nhân")  # từ dài nhất trong batch (9 ký tự)

    assert batch["word_ids"].shape == (B, L)
    assert batch["word_ids"].dtype == torch.long
    assert batch["char_ids"].shape == (B, L, W)
    assert batch["char_ids"].dtype == torch.long
    assert batch["mask"].shape == (B, L)
    assert batch["mask"].dtype == torch.bool
    assert batch["tags"].shape == (B, L)
    assert batch["tags"].dtype == torch.long
    assert batch["lengths"].tolist() == [4, 2]
    assert batch["words"] == [s["words"] for s in samples]

    # Kiểm tra mask và padding của câu thứ 2 (độ dài 2, pad 2 vị trí cuối)
    assert batch["mask"][0].tolist() == [True, True, True, True]
    assert batch["mask"][1].tolist() == [True, True, False, False]
    assert batch["word_ids"][1, 2:].tolist() == [0, 0]
    assert batch["tags"][1, :2].tolist() == [LABEL2ID["B-ORGANIZATION"], LABEL2ID["I-ORGANIZATION"]]
    assert batch["tags"][1, 2:].tolist() == [0, 0]
    assert (batch["char_ids"][1, 2:, :] == 0).all()
