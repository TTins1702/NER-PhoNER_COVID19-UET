"""Đọc/ghi file dự đoán theo định dạng đã chốt (docs/interface_contract.md, mục b).

Tại sao viết như vậy:
- Tín (BiLSTM-CRF) và Đức Anh (PhoBERT) chỉ cần ghi ra **cùng một định dạng JSONL**; mọi bước sau
  (metrics, error analysis, notebook 04, bảng LaTeX) đọc file này -> hai mô hình được chấm bằng đúng
  một đoạn code, không ai tự viết hàm F1 riêng.
- Mỗi dòng giữ cả `words`, `gold`, `pred` để phân tích lỗi không cần mở lại dữ liệu gốc.
- Kiểm tra độ dài ngay lúc ghi/đọc: lỗi lệch subword của PhoBERT (pred dài hơn số từ) bị bắt ngay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.data.labels import LABEL2ID

PREDICTIONS_DIR = Path("outputs/predictions")


def check_alignment(words_list: Sequence[Sequence[str]] | None, y_true: Sequence[Sequence[str]],
                    y_pred: Sequence[Sequence[str]]) -> None:
    """Báo lỗi rõ ràng nếu số câu / độ dài câu / nhãn không khớp hợp đồng."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Số câu khác nhau: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if words_list is not None and len(words_list) != len(y_true):
        raise ValueError(f"Số câu khác nhau: words={len(words_list)}, y_true={len(y_true)}")
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if len(t) != len(p):
            raise ValueError(f"Câu #{i}: gold dài {len(t)} nhưng pred dài {len(p)} "
                             f"(pred phải ở mức từ, đúng bằng len(words))")
        if words_list is not None and len(words_list[i]) != len(t):
            raise ValueError(f"Câu #{i}: {len(words_list[i])} từ nhưng {len(t)} nhãn")
        for lab in list(t) + list(p):
            if lab not in LABEL2ID:
                raise ValueError(f"Câu #{i}: nhãn không hợp lệ '{lab}' (phải là chuỗi BIO, không phải id)")


def prediction_path(experiment_name: str, split: str, out_dir: Path | str = PREDICTIONS_DIR) -> Path:
    return Path(out_dir) / f"{experiment_name}_{split}.jsonl"


def save_predictions(path: Path | str, words_list: Sequence[Sequence[str]],
                     y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]) -> Path:
    """Ghi JSONL: mỗi dòng `{"id", "words", "gold", "pred"}` theo đúng thứ tự câu gốc."""
    check_alignment(words_list, y_true, y_pred)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (w, t, p) in enumerate(zip(words_list, y_true, y_pred)):
            f.write(json.dumps({"id": i, "words": list(w), "gold": list(t), "pred": list(p)},
                               ensure_ascii=False) + "\n")
    return path


def load_predictions(path: Path | str) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Đọc file dự đoán -> (words_list, y_true, y_pred)."""
    words_list, y_true, y_pred = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            words_list.append(obj["words"])
            y_true.append(obj["gold"])
            y_pred.append(obj["pred"])
    check_alignment(words_list, y_true, y_pred)
    return words_list, y_true, y_pred
