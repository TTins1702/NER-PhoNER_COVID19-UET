from collections.abc import Sequence
from typing import Any

from src.evaluation.metrics import count_bio_violations


def extract_entities(tags: Sequence[str], strict_iob2: bool = True) -> list[tuple[str, int, int]]:
    """Extract (entity_type, start, end) spans with half-open intervals [start, end).

    When ``strict_iob2=True``, illegal ``I-`` tags (at sentence start, after ``O``,
    or with a mismatched preceding entity type) do not start a span, matching
    ``seqeval(..., mode="strict", scheme=IOB2)``.
    """
    entities: list[tuple[str, int, int]] = []
    cur_type: str | None = None
    cur_start: int | None = None

    for idx, tag in enumerate(tags):
        if tag.startswith("B-"):
            if cur_type is not None and cur_start is not None:
                entities.append((cur_type, cur_start, idx))
            cur_type = tag[2:]
            cur_start = idx
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if cur_type != tag_type:
                if cur_type is not None and cur_start is not None:
                    entities.append((cur_type, cur_start, idx))
                if strict_iob2:
                    cur_type = None
                    cur_start = None
                else:
                    cur_type = tag_type
                    cur_start = idx
        else:
            if cur_type is not None and cur_start is not None:
                entities.append((cur_type, cur_start, idx))
            cur_type = None
            cur_start = None

    if cur_type is not None and cur_start is not None:
        entities.append((cur_type, cur_start, len(tags)))

    return entities


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def categorize_errors(
    words_list: Sequence[Sequence[str]],
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    max_examples_per_category: int = 50,
) -> dict[str, Any]:
    """Categorize NER errors into type_errors, boundary_errors, false_negatives, false_positives."""
    categories = ("type_errors", "boundary_errors", "false_negatives", "false_positives")
    counts = {cat: 0 for cat in categories}
    examples: dict[str, list[dict[str, Any]]] = {cat: [] for cat in categories}

    def _record(category: str, payload: dict[str, Any]) -> None:
        counts[category] += 1
        if len(examples[category]) < max_examples_per_category:
            examples[category].append(payload)

    for sent_idx, (words, gold_tags, pred_tags) in enumerate(zip(words_list, y_true, y_pred)):
        gold_spans = extract_entities(gold_tags)
        pred_spans = extract_entities(pred_tags)
        matched_gold: set[int] = set()
        matched_pred: set[int] = set()

        # 1. Exact span matches (correct or type error)
        for g_idx, (g_type, g_s, g_e) in enumerate(gold_spans):
            for p_idx, (p_type, p_s, p_e) in enumerate(pred_spans):
                if p_idx in matched_pred:
                    continue
                if g_s == p_s and g_e == p_e:
                    matched_gold.add(g_idx)
                    matched_pred.add(p_idx)
                    if g_type != p_type:
                        _record(
                            "type_errors",
                            {
                                "sentence_idx": sent_idx,
                                "text": " ".join(words[g_s:g_e]),
                                "gold_type": g_type,
                                "pred_type": p_type,
                                "span": (g_s, g_e),
                                "sentence": " ".join(words),
                            },
                        )
                    break

        # 2. Overlapping span matches with mismatched boundaries
        for g_idx, (g_type, g_s, g_e) in enumerate(gold_spans):
            if g_idx in matched_gold:
                continue
            for p_idx, (p_type, p_s, p_e) in enumerate(pred_spans):
                if p_idx in matched_pred:
                    continue
                if _spans_overlap(g_s, g_e, p_s, p_e):
                    matched_gold.add(g_idx)
                    matched_pred.add(p_idx)
                    _record(
                        "boundary_errors",
                        {
                            "sentence_idx": sent_idx,
                            "gold_text": " ".join(words[g_s:g_e]),
                            "pred_text": " ".join(words[p_s:p_e]),
                            "gold_type": g_type,
                            "pred_type": p_type,
                            "gold_span": (g_s, g_e),
                            "pred_span": (p_s, p_e),
                            "sentence": " ".join(words),
                        },
                    )
                    break

        # 3. Unmatched gold entities -> false_negatives
        for g_idx, (g_type, g_s, g_e) in enumerate(gold_spans):
            if g_idx not in matched_gold:
                _record(
                    "false_negatives",
                    {
                        "sentence_idx": sent_idx,
                        "text": " ".join(words[g_s:g_e]),
                        "gold_type": g_type,
                        "span": (g_s, g_e),
                        "sentence": " ".join(words),
                    },
                )

        # 4. Unmatched predicted entities -> false_positives
        for p_idx, (p_type, p_s, p_e) in enumerate(pred_spans):
            if p_idx not in matched_pred:
                _record(
                    "false_positives",
                    {
                        "sentence_idx": sent_idx,
                        "text": " ".join(words[p_s:p_e]),
                        "pred_type": p_type,
                        "span": (p_s, p_e),
                        "sentence": " ".join(words),
                    },
                )

    return {
        "counts": {**counts, "bio_violations": count_bio_violations(y_pred)},
        "examples": examples,
    }
