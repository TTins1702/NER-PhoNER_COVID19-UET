"""Unit test cho src/data/labels.py và src/data/download.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from seqeval.scheme import IOB2, Entities

from src.data.download import (
    load_conll,
    load_jsonl,
    normalize_text,
    save_jsonl,
    validate_sample,
)
from src.data.labels import (
    DISPLAY_NAMES,
    ENTITY_TYPES,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    NUM_LABELS,
    O_ID,
    decode_ids,
    display_name,
    encode_tags,
    extract_entities,
    is_valid_label,
    split_label,
)


def test_fixed_21_bio_labels() -> None:
    assert NUM_LABELS == 21
    assert len(LABELS) == 21
    assert LABELS[0] == "O"
    assert O_ID == 0
    assert LABEL2ID["O"] == 0
    assert list(ENTITY_TYPES) == sorted(ENTITY_TYPES)
    for i, lab in enumerate(LABELS):
        assert LABEL2ID[lab] == i
        assert ID2LABEL[i] == lab
        assert is_valid_label(lab)
    assert not is_valid_label("B-PERSON")


def test_encode_decode_and_display_names() -> None:
    tags = ["O", "B-PATIENT_ID", "O", "B-LOCATION", "I-LOCATION"]
    ids = encode_tags(tags)
    assert ids[0] == 0
    assert decode_ids(ids) == tags
    assert display_name("NAME") == "PERSON_NAME"
    assert display_name("JOB") == "OCCUPATION"
    assert set(DISPLAY_NAMES.keys()) == set(ENTITY_TYPES)
    with pytest.raises(ValueError):
        encode_tags(["O", "B-INVALID"])
    with pytest.raises(ValueError):
        split_label("X-AGE")


def test_extract_entities_strict_iob2_matches_seqeval() -> None:
    # Gồm cả chuỗi chuẩn và chuỗi có I-X lạc loài (orphan I-X)
    cases = [
        ["O", "B-AGE", "O", "B-LOCATION", "I-LOCATION"],
        ["I-LOCATION", "I-LOCATION", "O", "B-NAME"],
        ["B-LOCATION", "I-ORGANIZATION", "I-ORGANIZATION", "O"],
        ["B-DATE", "B-DATE", "I-DATE"],
        ["O", "O", "O"],
    ]
    for seq in cases:
        ours = sorted(extract_entities(seq))
        seqeval_ents = Entities([seq], IOB2, suffix=False).entities[0]
        theirs = sorted((e.tag, e.start, e.end) for e in seqeval_ents)
        assert ours == theirs


def test_jsonl_and_conll_loading_and_validation(tmp_path: Path) -> None:
    samples = [
        {"words": ["Bệnh_nhân", "BN17", "ở", "Hà_Nội", "."],
         "tags": ["O", "B-PATIENT_ID", "O", "B-LOCATION", "O"]},
        {"words": ["Sốt", "và", "ho", "."],
         "tags": ["B-SYMPTOM_AND_DISEASE", "O", "B-SYMPTOM_AND_DISEASE", "O"]},
    ]
    jsonl_path = tmp_path / "sample.json"
    save_jsonl(samples, jsonl_path)
    loaded = load_jsonl(jsonl_path)
    assert loaded == samples

    # Tạo file CoNLL tương đương
    conll_path = tmp_path / "sample.conll"
    conll_lines = []
    for s in samples:
        for w, t in zip(s["words"], s["tags"]):
            conll_lines.append(f"{w} {t}")
        conll_lines.append("")
    conll_path.write_text("\n".join(conll_lines), encoding="utf-8")
    assert load_conll(conll_path) == samples

    # Kiểm tra chuẩn hóa NFC (chữ 'ế' dạng tổ hợp NFD -> NFC)
    nfd_word = "b\u00ea\u0323nh_vi\u00ea\u0323n"
    assert normalize_text(nfd_word) == "bệnh_viện"

    # Kiểm tra phát hiện lỗi lệch độ dài và nhãn lạ
    with pytest.raises(ValueError):
        validate_sample({"words": ["a", "b"], "tags": ["O"]})
    with pytest.raises(ValueError):
        validate_sample({"words": ["a"], "tags": ["B-UNKNOWN"]})
