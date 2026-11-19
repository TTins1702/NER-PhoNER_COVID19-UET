"""Phân tích lỗi NER mức thực thể: 4 nhóm lỗi, ma trận nhầm lẫn, so sánh hai mô hình, câu mẫu.

Tại sao viết như vậy:
- F1 chỉ cho biết "sai bao nhiêu", không cho biết "sai kiểu gì". Tách lỗi thành 4 nhóm giúp giải thích
  kết quả khi vấn đáp: mô hình yếu ở **ranh giới** (thường do tách từ, thực thể dài như LOCATION/
  ORGANIZATION) hay ở **phân loại** (nhầm LOCATION <-> ORGANIZATION), hay **bỏ sót** (từ hiếm/OOV).
- Định nghĩa 4 nhóm (so sánh tập thực thể gold G và pred P của từng câu, sau khi bỏ các cặp khớp
  hoàn toàn (loại, start, end)):
    * boundary (lỗi ranh giới): pred giao với một gold **cùng loại** nhưng khác start/end.
    * type (nhầm loại): pred giao với gold nhưng **khác loại** (ghi thêm cờ `boundary_also_wrong`
      nếu đồng thời lệch ranh giới; ta xếp vào nhóm type vì loại sai là lỗi nặng hơn).
    * false_positive (báo nhầm): pred không giao với gold nào.
    * false_negative (bỏ sót): gold không bị pred nào giao tới.
  Mỗi pred sai được xếp đúng một nhóm (boundary/type/FP), nên
  `boundary + type + false_positive = số pred - số đúng` (= FP của seqeval). Bất biến này có unit test.
- Trích thực thể bằng `extract_entities` (strict IOB2) giống hệt seqeval -> số "correct" ở đây bằng TP
  của `metrics.py`, hai module không bao giờ lệch nhau.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.labels import ENTITY_TYPES, SHORT_NAMES, extract_entities, split_label
from src.evaluation.predictions import check_alignment, load_predictions

ERROR_TYPES = ("boundary", "type", "false_positive", "false_negative")
ERROR_NAMES_VI = {
    "boundary": "Lỗi ranh giới",
    "type": "Nhầm loại",
    "false_positive": "Báo nhầm (FP)",
    "false_negative": "Bỏ sót (FN)",
}
NONE_LABEL = "O"  # dùng trong ma trận nhầm lẫn: "không có thực thể"

Span = tuple[str, int, int]  # (loại, start, end) — end không bao gồm


def _overlap_len(a: Span, b: Span) -> int:
    return max(0, min(a[2], b[2]) - max(a[1], b[1]))


def span_text(words: Sequence[str], span: Span | None) -> str:
    return "" if span is None else " ".join(words[span[1]:span[2]])


@dataclass
class EntityError:
    sent_id: int
    error_type: str
    gold: Span | None
    pred: Span | None
    gold_text: str = ""
    pred_text: str = ""
    boundary_also_wrong: bool = False

    @property
    def gold_type(self) -> str | None:
        return self.gold[0] if self.gold else None

    @property
    def pred_type(self) -> str | None:
        return self.pred[0] if self.pred else None


def classify_sentence(gold_tags: Sequence[str], pred_tags: Sequence[str], sent_id: int = 0,
                      words: Sequence[str] | None = None) -> tuple[list[Span], list[EntityError]]:
    """Phân loại lỗi trong một câu. Trả về (danh sách thực thể đúng, danh sách lỗi)."""
    gold = extract_entities(gold_tags)
    pred = extract_entities(pred_tags)
    pred_set = set(pred)
    correct = [g for g in gold if g in pred_set]
    correct_set = set(correct)
    rem_gold = [g for g in gold if g not in correct_set]
    rem_pred = [p for p in pred if p not in correct_set]
    words = words or [""] * len(gold_tags)

    errors: list[EntityError] = []
    touched: set[Span] = set()
    for p in rem_pred:
        overlaps = [g for g in rem_gold if _overlap_len(g, p) > 0]
        if not overlaps:
            errors.append(EntityError(sent_id, "false_positive", None, p, "", span_text(words, p)))
            continue
        touched.update(overlaps)  # pred gộp nhiều gold -> các gold đó không tính là "bỏ sót"
        # Ưu tiên gold cùng loại, sau đó gold giao nhiều nhất, rồi gold đứng trước.
        g = max(overlaps, key=lambda x: (x[0] == p[0], _overlap_len(x, p), -x[1]))
        if g[0] == p[0]:
            etype, also = "boundary", False
        else:
            etype, also = "type", (g[1], g[2]) != (p[1], p[2])
        errors.append(EntityError(sent_id, etype, g, p, span_text(words, g), span_text(words, p), also))
    for g in rem_gold:
        if g not in touched:
            errors.append(EntityError(sent_id, "false_negative", g, None, span_text(words, g), ""))
    return correct, errors


@dataclass
class ErrorReport:
    counts: dict[str, int]
    by_entity: dict[str, dict[str, int]]
    type_confusions: list[tuple[str, str, int]]
    num_sentences: int
    num_sentences_with_error: int
    num_gold_entities: int
    num_pred_entities: int
    errors: list[EntityError] = field(default_factory=list, repr=False)

    def summary_dict(self) -> dict[str, Any]:
        """Phần tóm tắt (không chứa câu dữ liệu) — nhẹ, commit được."""
        return {
            "counts": self.counts,
            "by_entity": self.by_entity,
            "type_confusions": [list(x) for x in self.type_confusions],
            "num_sentences": self.num_sentences,
            "num_sentences_with_error": self.num_sentences_with_error,
            "num_gold_entities": self.num_gold_entities,
            "num_pred_entities": self.num_pred_entities,
        }

    def to_json(self, path: Path | str, include_errors: bool = False) -> Path:
        d = self.summary_dict()
        if include_errors:
            d["errors"] = [asdict(e) for e in self.errors]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def analyze_errors(words_list: Sequence[Sequence[str]] | None, y_true: Sequence[Sequence[str]],
                   y_pred: Sequence[Sequence[str]]) -> ErrorReport:
    """Phân loại lỗi trên cả tập.

    `by_entity[t]` đếm theo loại **gold** cho correct/boundary/type/false_negative và theo loại **pred**
    cho false_positive (vì FP không có gold).
    """
    check_alignment(words_list, y_true, y_pred)
    counts = Counter({k: 0 for k in ("correct",) + ERROR_TYPES})
    by_entity = {t: {k: 0 for k in ("correct",) + ERROR_TYPES} for t in ENTITY_TYPES}
    confusions: Counter = Counter()
    all_errors: list[EntityError] = []
    sent_err = n_gold = n_pred = 0
    for i, (gt, pt) in enumerate(zip(y_true, y_pred)):
        words = words_list[i] if words_list is not None else None
        correct, errors = classify_sentence(gt, pt, i, words)
        n_gold += len(extract_entities(gt))
        n_pred += len(extract_entities(pt))
        counts["correct"] += len(correct)
        for c in correct:
            by_entity[c[0]]["correct"] += 1
        for e in errors:
            counts[e.error_type] += 1
            key_type = e.pred_type if e.error_type == "false_positive" else e.gold_type
            by_entity[key_type][e.error_type] += 1
            if e.error_type == "type":
                confusions[(e.gold_type, e.pred_type)] += 1
        sent_err += bool(errors)
        all_errors.extend(errors)
    type_conf = sorted(((g, p, n) for (g, p), n in confusions.items()), key=lambda x: (-x[2], x[0], x[1]))
    return ErrorReport(dict(counts), by_entity, type_conf, len(y_true), sent_err, n_gold, n_pred, all_errors)


# ---------------------------------------------------------------------------
# Ma trận nhầm lẫn
# ---------------------------------------------------------------------------
def entity_confusion_matrix(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]
                            ) -> tuple[np.ndarray, list[str]]:
    """Ma trận nhầm lẫn **mức thực thể** (hàng = loại gold, cột = loại pred), khớp theo phần giao.

    - Thực thể đúng hoàn toàn và lỗi ranh giới nằm trên đường chéo (cùng loại).
    - Nhầm loại nằm ngoài đường chéo.
    - Cột "O": gold bị bỏ sót; hàng "O": pred báo nhầm.
    """
    labels = list(ENTITY_TYPES) + [NONE_LABEL]
    idx = {t: i for i, t in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for gt, pt in zip(y_true, y_pred):
        correct, errors = classify_sentence(gt, pt)
        for c in correct:
            mat[idx[c[0]], idx[c[0]]] += 1
        for e in errors:
            g = e.gold_type or NONE_LABEL
            p = e.pred_type or NONE_LABEL
            mat[idx[g], idx[p]] += 1
    return mat, labels


def token_type_confusion_matrix(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]
                                ) -> tuple[np.ndarray, list[str]]:
    """Ma trận nhầm lẫn **mức từ** theo loại (bỏ tiền tố B/I), chỉ để trực quan hóa, không dùng tính F1."""
    labels = list(ENTITY_TYPES) + [NONE_LABEL]
    idx = {t: i for i, t in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for gt, pt in zip(y_true, y_pred):
        for a, b in zip(gt, pt):
            ga = split_label(a)[1] or NONE_LABEL
            pb = split_label(b)[1] or NONE_LABEL
            mat[idx[ga], idx[pb]] += 1
    return mat, labels


# ---------------------------------------------------------------------------
# Phân tích theo độ dài thực thể / OOV
# ---------------------------------------------------------------------------
def recall_by_entity_length(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]],
                            buckets: Sequence[int] = (1, 2, 3, 4)) -> dict[str, dict[str, float]]:
    """Recall (khớp hoàn toàn) theo độ dài thực thể gold tính bằng số từ; bucket cuối là '>= k'."""
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    last = buckets[-1]
    for gt, pt in zip(y_true, y_pred):
        pred_set = set(extract_entities(pt))
        for g in extract_entities(gt):
            n = g[2] - g[1]
            key = f">={last}" if n >= last else str(n)
            stats[key][0] += g in pred_set
            stats[key][1] += 1
    order = [str(b) for b in buckets[:-1]] + [f">={last}"]
    return {k: {"recall": stats[k][0] / stats[k][1] if stats[k][1] else 0.0, "support": stats[k][1]}
            for k in order}


def recall_by_oov(words_list: Sequence[Sequence[str]], y_true: Sequence[Sequence[str]],
                  y_pred: Sequence[Sequence[str]], in_vocab) -> dict[str, dict[str, float]]:
    """Recall của thực thể chứa ít nhất một từ OOV so với thực thể toàn từ đã biết.

    `in_vocab`: hàm `str -> bool` hoặc đối tượng hỗ trợ `in` (ví dụ `WordVocab` xây từ train).
    """
    check = in_vocab if callable(in_vocab) else (lambda w: w in in_vocab)
    stats = {"has_oov": [0, 0], "all_known": [0, 0]}
    for words, gt, pt in zip(words_list, y_true, y_pred):
        pred_set = set(extract_entities(pt))
        for g in extract_entities(gt):
            key = "has_oov" if any(not check(w) for w in words[g[1]:g[2]]) else "all_known"
            stats[key][0] += g in pred_set
            stats[key][1] += 1
    return {k: {"recall": v[0] / v[1] if v[1] else 0.0, "support": v[1]} for k, v in stats.items()}


# ---------------------------------------------------------------------------
# So sánh hai mô hình và câu mẫu
# ---------------------------------------------------------------------------
def sentence_is_correct(gold_tags: Sequence[str], pred_tags: Sequence[str]) -> bool:
    """Câu đúng = tập thực thể dự đoán trùng khớp hoàn toàn tập thực thể gold."""
    return set(extract_entities(gold_tags)) == set(extract_entities(pred_tags))


def compare_models(y_true: Sequence[Sequence[str]], pred_a: Sequence[Sequence[str]],
                   pred_b: Sequence[Sequence[str]]) -> dict[str, Any]:
    """So sánh từng câu và từng thực thể giữa hai mô hình A và B."""
    check_alignment(None, y_true, pred_a)
    check_alignment(None, y_true, pred_b)
    sent = {"both_correct": [], "only_a": [], "only_b": [], "both_wrong": []}
    ent = Counter()
    ent_by_type = {t: Counter() for t in ENTITY_TYPES}
    for i, (g, a, b) in enumerate(zip(y_true, pred_a, pred_b)):
        ca, cb = sentence_is_correct(g, a), sentence_is_correct(g, b)
        key = "both_correct" if ca and cb else "only_a" if ca else "only_b" if cb else "both_wrong"
        sent[key].append(i)
        sa, sb = set(extract_entities(a)), set(extract_entities(b))
        for e in extract_entities(g):
            k = ("both" if e in sa and e in sb else "only_a" if e in sa
                 else "only_b" if e in sb else "neither")
            ent[k] += 1
            ent_by_type[e[0]][k] += 1
    return {
        "sentence_counts": {k: len(v) for k, v in sent.items()},
        "sentence_ids": sent,
        "entity_counts": {k: ent.get(k, 0) for k in ("both", "only_a", "only_b", "neither")},
        "entity_counts_by_type": {t: {k: c.get(k, 0) for k in ("both", "only_a", "only_b", "neither")}
                                  for t, c in ent_by_type.items()},
    }


def format_tagged_sentence(words: Sequence[str], tags: Sequence[str], short: bool = True) -> str:
    """'Bệnh_nhân [BN17]PID ở [Hà_Nội]LOC' — dạng dễ đọc để đưa ví dụ vào báo cáo/slide."""
    spans = {s: (t, e) for t, s, e in extract_entities(tags)}
    out, i = [], 0
    while i < len(words):
        if i in spans:
            t, e = spans[i]
            out.append(f"[{' '.join(words[i:e])}]{SHORT_NAMES[t] if short else t}")
            i = e
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def error_samples_table(report: ErrorReport, words_list: Sequence[Sequence[str]],
                        y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]],
                        n_per_type: int = 5, seed: int = 42, max_words: int = 60):
    """Bảng câu sai mẫu: tối đa `n_per_type` lỗi cho mỗi nhóm (chọn ngẫu nhiên có seed để tái lập).

    Trả về pandas.DataFrame: error_type, sent_id, gold_type, pred_type, gold_text, pred_text,
    sentence_gold, sentence_pred. Câu quá dài (> max_words từ) bị bỏ để bảng gọn.
    """
    import pandas as pd

    rng = random.Random(seed)
    rows = []
    for etype in ERROR_TYPES:
        pool = [e for e in report.errors if e.error_type == etype and len(words_list[e.sent_id]) <= max_words]
        for e in rng.sample(pool, min(n_per_type, len(pool))):
            rows.append({
                "error_type": etype,
                "error_name": ERROR_NAMES_VI[etype],
                "sent_id": e.sent_id,
                "gold_type": e.gold_type or "",
                "pred_type": e.pred_type or "",
                "gold_text": e.gold_text,
                "pred_text": e.pred_text,
                "sentence_gold": format_tagged_sentence(words_list[e.sent_id], y_true[e.sent_id]),
                "sentence_pred": format_tagged_sentence(words_list[e.sent_id], y_pred[e.sent_id]),
            })
    return pd.DataFrame(rows, columns=["error_type", "error_name", "sent_id", "gold_type", "pred_type",
                                       "gold_text", "pred_text", "sentence_gold", "sentence_pred"])


def pick_sentence_examples(words_list: Sequence[Sequence[str]], y_true: Sequence[Sequence[str]],
                           y_pred: Sequence[Sequence[str]], correct: bool, n: int = 3,
                           min_entities: int = 2, max_words: int = 35, seed: int = 0) -> list[int]:
    """Chọn id câu ví dụ "tốt" (đúng hoàn toàn) hoặc "xấu" (có lỗi) có ít nhất `min_entities` thực thể."""
    cands = [i for i in range(len(y_true))
             if len(words_list[i]) <= max_words and len(extract_entities(y_true[i])) >= min_entities
             and sentence_is_correct(y_true[i], y_pred[i]) == correct]
    rng = random.Random(seed)
    return sorted(rng.sample(cands, min(n, len(cands))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích lỗi file dự đoán JSONL")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out-json", default=None, help="Tóm tắt số lỗi (mặc định outputs/metrics/<tên>_errors.json)")
    parser.add_argument("--out-csv", default=None, help="Bảng câu sai mẫu (mặc định outputs/metrics/<tên>_error_samples.csv)")
    parser.add_argument("--n-per-type", type=int, default=5)
    args = parser.parse_args()
    stem = Path(args.pred).stem
    words_list, y_true, y_pred = load_predictions(args.pred)
    report = analyze_errors(words_list, y_true, y_pred)
    out_json = Path(args.out_json or f"outputs/metrics/{stem}_errors.json")
    out_csv = Path(args.out_csv or f"outputs/metrics/{stem}_error_samples.csv")
    report.to_json(out_json)
    df = error_samples_table(report, words_list, y_true, y_pred, n_per_type=args.n_per_type)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(json.dumps(report.counts, ensure_ascii=False))
    print(f"[error_analysis] Đã lưu {out_json} và {out_csv}")


if __name__ == "__main__":
    main()
