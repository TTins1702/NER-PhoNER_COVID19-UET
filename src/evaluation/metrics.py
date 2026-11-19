"""Bộ chấm điểm chuẩn của dự án: seqeval strict IOB2, mức thực thể, mức từ.

Tại sao viết như vậy:
- **Mức thực thể, không phải mức token**: một thực thể "Bệnh_viện Bạch_Mai" chỉ được tính đúng khi
  đúng **cả ranh giới lẫn loại**. Accuracy/F1 theo token (sklearn) bị nhãn `O` (~80% token) thổi phồng
  và cho điểm cả thực thể đoán đúng một nửa -> không dùng.
- **Mức từ gốc**: PhoBERT dự đoán trên subword, BiLSTM trên từ; nếu chấm ở mức subword thì một từ bị
  tách 3 subword được đếm 3 lần -> hai mô hình không so sánh được. Mọi dự đoán phải quy về
  đúng `len(words)` nhãn trước khi vào hàm này (xem `predictions.check_alignment`).
- **strict + IOB2**: thực thể chỉ hợp lệ khi bắt đầu bằng `B-X`. `I-X` lạc loài bị bỏ (không được tính
  là thực thể), nên mô hình Softmax sinh chuỗi sai luật bị phạt đúng mức. Chế độ mặc định (lỏng) của
  seqeval/conlleval coi `I-X` lạc là thực thể mới -> che mất lợi ích của CRF.
- Đếm **vi phạm BIO** trên dự đoán thô để định lượng lợi ích của tầng CRF trong phần Analysis;
  `repair_bio` (đổi I-X lạc thành B-X) chỉ dùng như một phân tích phụ, kết quả chính chấm trên dự đoán thô.
- Schema JSON cố định (`schema_version`) để script bảng LaTeX và notebook đọc được mọi thí nghiệm.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from src.data.labels import ENTITY_TYPES, extract_entities, split_label
from src.evaluation.predictions import check_alignment, load_predictions

SCHEMA_VERSION = "1.0"
EVAL_MODE = "seqeval-strict-IOB2"
METRICS_DIR = Path("outputs/metrics")
VALID_GROUPS = ("main", "ablation_bilstm", "ablation_phobert", "other")


# ---------------------------------------------------------------------------
# Vi phạm BIO
# ---------------------------------------------------------------------------
def find_bio_violations(tags: Sequence[str]) -> list[int]:
    """Vị trí các nhãn `I-X` không hợp lệ: không đứng sau `B-X` hoặc `I-X` cùng loại.

    Ví dụ: ['O', 'I-LOCATION', 'B-NAME', 'I-AGE'] -> [1, 3]
    """
    positions = []
    prev_type: str | None = None  # loại của nhãn trước nếu là B-/I-, None nếu là O hoặc đầu câu
    for i, tag in enumerate(tags):
        prefix, etype = split_label(tag)
        if prefix == "I" and etype != prev_type:
            positions.append(i)
        prev_type = etype if prefix in ("B", "I") else None
    return positions


def count_bio_violations(sequences: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Thống kê vi phạm BIO trên cả tập.

    Trả về: total (số nhãn vi phạm), by_type (theo loại X của I-X), sentences_with_violation,
    num_tokens, per_1k_tokens.
    """
    by_type: Counter = Counter()
    sent_with = 0
    n_tokens = 0
    for tags in sequences:
        n_tokens += len(tags)
        pos = find_bio_violations(tags)
        if pos:
            sent_with += 1
        for i in pos:
            by_type[split_label(tags[i])[1]] += 1
    total = sum(by_type.values())
    return {
        "total": total,
        "by_type": {t: by_type.get(t, 0) for t in ENTITY_TYPES},
        "sentences_with_violation": sent_with,
        "num_tokens": n_tokens,
        "per_1k_tokens": round(1000 * total / n_tokens, 4) if n_tokens else 0.0,
    }


def repair_bio(tags: Sequence[str]) -> list[str]:
    """Sửa chuỗi nhãn về IOB2 hợp lệ: mỗi `I-X` không hợp lệ được đổi thành `B-X`.

    Đây là luật sửa phổ biến (giống cách conlleval hiểu chuỗi lỏng). Ví dụ:
    ['O', 'I-LOC', 'I-LOC', 'B-NAME', 'I-AGE'] -> ['O', 'B-LOC', 'I-LOC', 'B-NAME', 'B-AGE']
    """
    fixed = list(tags)
    for i in find_bio_violations(tags):
        fixed[i] = "B-" + split_label(tags[i])[1]
    return fixed


def repair_bio_batch(sequences: Sequence[Sequence[str]]) -> list[list[str]]:
    return [repair_bio(s) for s in sequences]


# ---------------------------------------------------------------------------
# Chấm điểm
# ---------------------------------------------------------------------------
def _prf(d: dict[str, Any]) -> dict[str, float | int]:
    return {"precision": float(d["precision"]), "recall": float(d["recall"]),
            "f1": float(d["f1-score"]), "support": int(d["support"])}


def _empty_prf() -> dict[str, float | int]:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def compute_metrics(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]],
                    repair: bool = False) -> dict[str, Any]:
    """Tính P/R/F1 mức thực thể (seqeval strict IOB2) cho 10 loại + micro/macro/weighted.

    Args:
        y_true, y_pred: list[list[str]] nhãn BIO mức từ, cùng độ dài từng câu.
        repair: nếu True, sửa vi phạm BIO trong y_pred **trước khi chấm** (chỉ cho phân tích phụ).
    Returns: dict gồm `overall`, `per_entity`, `bio_violations`, số câu/token/thực thể.
    """
    check_alignment(None, y_true, y_pred)
    y_true = [list(s) for s in y_true]
    raw_pred = [list(s) for s in y_pred]
    y_pred_eval = repair_bio_batch(raw_pred) if repair else raw_pred

    report = classification_report(y_true, y_pred_eval, mode="strict", scheme=IOB2,
                                   output_dict=True, zero_division=0)
    per_entity = {t: _prf(report[t]) if t in report else _empty_prf() for t in ENTITY_TYPES}
    # Nếu loại t không có trong gold lẫn pred thì seqeval không trả về -> support = 0.
    # Nếu loại t chỉ có trong pred (support = 0) seqeval vẫn trả về với P = 0.
    overall = {
        "micro": _prf(report["micro avg"]) if "micro avg" in report else _empty_prf(),
        "macro": _prf(report["macro avg"]) if "macro avg" in report else _empty_prf(),
        "weighted": _prf(report["weighted avg"]) if "weighted avg" in report else _empty_prf(),
    }
    return {
        "eval_mode": EVAL_MODE,
        "repaired_bio": repair,
        "num_sentences": len(y_true),
        "num_tokens": sum(len(s) for s in y_true),
        "num_gold_entities": sum(len(extract_entities(s)) for s in y_true),
        "num_pred_entities": sum(len(extract_entities(s)) for s in y_pred_eval),
        "overall": overall,
        "per_entity": per_entity,
        "bio_violations": {"pred": count_bio_violations(raw_pred), "gold": count_bio_violations(y_true)},
    }


def build_result(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]], experiment_name: str,
                 model: str, split: str, level: str = "word", group: str = "main",
                 description: str | None = None, is_ablation_reference: bool = False,
                 config: dict | None = None, extra: dict | None = None, repair: bool = False) -> dict[str, Any]:
    """Gói kết quả `compute_metrics` + metadata theo schema cố định (xem docs/interface_contract.md)."""
    if group not in VALID_GROUPS:
        raise ValueError(f"group phải thuộc {VALID_GROUPS}, nhận '{group}'")
    m = compute_metrics(y_true, y_pred, repair=repair)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "description": description or experiment_name,
        "model": model,
        "group": group,
        "is_ablation_reference": bool(is_ablation_reference),
        "split": split,
        "level": level,
        **m,
        "config": config or {},
        "extra": extra or {},
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def metrics_path(experiment_name: str, split: str, out_dir: Path | str = METRICS_DIR) -> Path:
    return Path(out_dir) / f"{experiment_name}_{split}.json"


def save_metrics_json(result: dict[str, Any], out_dir: Path | str = METRICS_DIR,
                      path: Path | str | None = None) -> Path:
    """Ghi kết quả ra `outputs/metrics/<experiment_name>_<split>.json`."""
    path = Path(path) if path else metrics_path(result["experiment_name"], result["split"], out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def load_metrics_json(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version={result.get('schema_version')} khác {SCHEMA_VERSION}")
    return result


def load_all_metrics(metrics_dir: Path | str = METRICS_DIR, split: str | None = None) -> list[dict[str, Any]]:
    """Đọc mọi file kết quả hợp lệ trong thư mục (bỏ qua file thống kê dữ liệu không theo schema)."""
    results = []
    for p in sorted(Path(metrics_dir).glob("*.json")):
        try:
            r = load_metrics_json(p)
        except (ValueError, json.JSONDecodeError, KeyError):
            continue
        if split is None or r["split"] == split:
            results.append(r)
    return results


def evaluate_and_save(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]], experiment_name: str,
                      model: str, split: str, level: str = "word", group: str = "main",
                      description: str | None = None, is_ablation_reference: bool = False,
                      config: dict | None = None, extra: dict | None = None, repair: bool = False,
                      out_dir: Path | str = METRICS_DIR, verbose: bool = True) -> dict[str, Any]:
    """Một lệnh cho Tín/Đức Anh: chấm điểm + ghi JSON + in tóm tắt."""
    result = build_result(y_true, y_pred, experiment_name, model, split, level, group, description,
                          is_ablation_reference, config, extra, repair)
    path = save_metrics_json(result, out_dir)
    if verbose:
        print(format_summary(result))
        print(f"[metrics] Đã lưu {path}")
    return result


def format_summary(result: dict[str, Any]) -> str:
    """Bảng chữ ngắn để in ra console/notebook."""
    lines = [f"== {result.get('experiment_name', '')} [{result.get('split', '')}] ({result['eval_mode']}) ==",
             f"{'Loại':<22}{'P':>8}{'R':>8}{'F1':>8}{'Support':>9}"]
    for t, m in result["per_entity"].items():
        lines.append(f"{t:<22}{100 * m['precision']:8.2f}{100 * m['recall']:8.2f}"
                     f"{100 * m['f1']:8.2f}{m['support']:9d}")
    for k in ("micro", "macro"):
        m = result["overall"][k]
        lines.append(f"{k + ' avg':<22}{100 * m['precision']:8.2f}{100 * m['recall']:8.2f}"
                     f"{100 * m['f1']:8.2f}{m['support']:9d}")
    v = result["bio_violations"]["pred"]
    lines.append(f"Vi phạm BIO trong dự đoán: {v['total']} nhãn / {v['sentences_with_violation']} câu")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chấm điểm file dự đoán JSONL bằng seqeval strict IOB2")
    parser.add_argument("--pred", required=True, help="outputs/predictions/<exp>_<split>.jsonl")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--level", default="word")
    parser.add_argument("--group", default="main", choices=VALID_GROUPS)
    parser.add_argument("--description", default=None)
    parser.add_argument("--reference", action="store_true", help="Đánh dấu mô hình đầy đủ của nhóm ablation")
    parser.add_argument("--repair", action="store_true", help="Sửa vi phạm BIO trước khi chấm (phân tích phụ)")
    parser.add_argument("--out-dir", default=str(METRICS_DIR))
    args = parser.parse_args()
    _, y_true, y_pred = load_predictions(args.pred)
    evaluate_and_save(y_true, y_pred, args.experiment_name, args.model, args.split, args.level, args.group,
                      args.description, args.reference, repair=args.repair, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
