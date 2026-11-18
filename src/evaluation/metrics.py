from collections.abc import Sequence
from typing import Any

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from src.data.labels import ENTITY_TYPES


def count_bio_violations(tag_sequences: Sequence[Sequence[str]]) -> int:
    """Count illegal I- transitions (at sentence start, after O, or mismatched entity type)."""
    violations = 0
    for seq in tag_sequences:
        prev_type: str | None = None
        for tag in seq:
            if tag.startswith("I-"):
                cur_type = tag[2:]
                if prev_type != cur_type:
                    violations += 1
                prev_type = cur_type
            elif tag.startswith("B-"):
                prev_type = tag[2:]
            else:
                prev_type = None
    return violations


def compute_ner_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict[str, Any]:
    """Compute entity-level strict IOB2 metrics via seqeval and count BIO violations."""
    report = classification_report(
        y_true,
        y_pred,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )

    per_entity: dict[str, dict[str, float | int]] = {}
    for entity_type in ENTITY_TYPES:
        entry = report.get(entity_type, {})
        per_entity[entity_type] = {
            "precision": float(entry.get("precision", 0.0)),
            "recall": float(entry.get("recall", 0.0)),
            "f1": float(entry.get("f1-score", 0.0)),
            "support": int(entry.get("support", 0)),
        }

    micro = report.get("micro avg", {})
    macro = report.get("macro avg", {})

    return {
        "micro_precision": float(micro.get("precision", 0.0)),
        "micro_recall": float(micro.get("recall", 0.0)),
        "micro_f1": float(micro.get("f1-score", 0.0)),
        "macro_f1": float(macro.get("f1-score", 0.0)),
        "per_entity": per_entity,
        "bio_violations": count_bio_violations(y_pred),
    }
