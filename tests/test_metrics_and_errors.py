"""Unit test cho src/evaluation/ (metrics, error_analysis, predictions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.error_analysis import (
    analyze_errors,
    compare_models,
    entity_confusion_matrix,
    error_samples_table,
    recall_by_entity_length,
    recall_by_oov,
)
from src.evaluation.metrics import (
    SCHEMA_VERSION,
    bio_violation_stats,
    compute_metrics,
    evaluate_and_save,
    find_bio_violations,
    load_metrics_json,
    repair_bio,
)
from src.evaluation.predictions import check_alignment, load_predictions, save_predictions


def test_bio_violations_and_strict_iob2_scoring() -> None:
    y_true = [["B-LOCATION", "I-LOCATION", "O", "B-AGE"]]
    # Dự đoán có 1 lỗi BIO: I-LOCATION đứng đầu câu (không có B-LOCATION) và I-DATE sau B-AGE
    y_pred = [["I-LOCATION", "I-LOCATION", "O", "I-DATE"]]

    assert find_bio_violations(y_pred[0]) == [0, 3]
    v = bio_violation_stats(y_pred)
    assert v["total"] == 2
    assert v["by_type"]["LOCATION"] == 1
    assert v["by_type"]["DATE"] == 1
    assert v["sentences_with_violation"] == 1

    # Chấm strict IOB2 trên dự đoán thô: I-LOCATION đầu câu không được tính là thực thể -> F1 = 0
    m_raw = compute_metrics(y_true, y_pred, repair=False)
    assert m_raw["overall"]["micro"]["f1"] == 0.0
    assert m_raw["bio_violations"]["pred"]["total"] == 2

    # Sau khi repair_bio: I-LOCATION đầu câu -> B-LOCATION -> khớp thực thể LOCATION
    repaired_seq = repair_bio(y_pred[0])
    assert repaired_seq == ["B-LOCATION", "I-LOCATION", "O", "B-DATE"]
    assert find_bio_violations(repaired_seq) == []
    m_rep = compute_metrics(y_true, y_pred, repair=True)
    assert abs(m_rep["per_entity"]["LOCATION"]["f1"] - 1.0) < 1e-6


def test_alignment_check_and_predictions_io(tmp_path: Path) -> None:
    words = [["Bệnh_nhân", "BN17"]]
    gold = [["O", "B-PATIENT_ID"]]
    pred = [["O", "B-PATIENT_ID"]]
    p = save_predictions(tmp_path / "pred.jsonl", words, gold, pred)
    w2, g2, p2 = load_predictions(p)
    assert (w2, g2, p2) == (words, gold, pred)

    # Lệch độ dài (ví dụ quên gộp subword về mức từ) phải bị bắt ngay
    with pytest.raises(ValueError):
        check_alignment(words, gold, [["O", "B-PATIENT_ID", "I-PATIENT_ID"]])


def test_error_analysis_four_categories_and_invariants() -> None:
    # Câu gồm 4 thực thể gold để kiểm tra đủ 4 nhóm lỗi:
    # 1) BN17: đoán đúng (correct)
    # 2) quận Hoàn_Kiếm (idx 2..4): đoán lệch ranh giới chỉ có token 2 (boundary)
    # 3) Bộ Y_tế (idx 5..7): đoán nhầm loại thành LOCATION (type)
    # 4) sốt (idx 8..9): bỏ sót (false_negative)
    # + token 9 ("hôm_nay" = O): báo nhầm thành B-DATE (false_positive)
    words = [["BN17", "ở", "quận", "Hoàn_Kiếm", ",", "Bộ", "Y_tế", "báo", "sốt", "hôm_nay"]]
    gold = [[
        "B-PATIENT_ID", "O", "B-LOCATION", "I-LOCATION", "O",
        "B-ORGANIZATION", "I-ORGANIZATION", "O", "B-SYMPTOM_AND_DISEASE", "O",
    ]]
    pred = [[
        "B-PATIENT_ID", "O", "B-LOCATION", "O", "O",
        "B-LOCATION", "I-LOCATION", "O", "O", "B-DATE",
    ]]

    rep = analyze_errors(words, gold, pred)
    assert rep.counts["correct"] == 1
    assert rep.counts["boundary"] == 1
    assert rep.counts["type"] == 1
    assert rep.counts["false_positive"] == 1
    assert rep.counts["false_negative"] == 1

    # Bất biến: boundary + type + false_positive == num_pred_entities - correct
    c = rep.counts
    assert c["boundary"] + c["type"] + c["false_positive"] == rep.num_pred_entities - c["correct"]

    cm, labels = entity_confusion_matrix(gold, pred)
    assert cm.shape == (11, 11)
    assert labels[-1] == "O"

    df_samples = error_samples_table(rep, words, gold, pred, n_per_type=2)
    assert len(df_samples) > 0

    len_rec = recall_by_entity_length(gold, pred)
    assert "1" in len_rec
    oov_rec = recall_by_oov(words, gold, pred, in_vocab=lambda w: w != "BN17")
    assert set(oov_rec.keys()) == {"has_oov", "all_known"}

    cmp_res = compare_models(gold, pred, gold)
    assert cmp_res["entity_counts"]["only_a"] == 0


def test_metrics_json_roundtrip(tmp_path: Path) -> None:
    y_true = [["B-PATIENT_ID", "O", "B-LOCATION", "I-LOCATION"]]
    y_pred = [["B-PATIENT_ID", "O", "B-LOCATION", "O"]]

    res = evaluate_and_save(
        y_true, y_pred, experiment_name="bilstm_crf_smoke", model="bilstm_crf",
        split="test", group="main", out_dir=tmp_path, verbose=False,
    )
    assert res["schema_version"] == SCHEMA_VERSION
    loaded = load_metrics_json(tmp_path / "bilstm_crf_smoke_test.json")
    assert loaded["overall"]["micro"]["f1"] == res["overall"]["micro"]["f1"]
