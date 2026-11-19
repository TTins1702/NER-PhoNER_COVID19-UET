"""Unit test cho src/evaluation/ (metrics, error_analysis, latex_tables) và app/predictor.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.predictor import DummyNERPredictor, load_model, predict_sentence, tokenize_vietnamese
from src.evaluation.dummy_predictions import make_dummy_predictions, make_synthetic_corpus
from src.evaluation.error_analysis import (
    analyze_errors,
    compare_models,
    entity_confusion_matrix,
    error_samples_table,
    recall_by_entity_length,
    recall_by_oov,
)
from src.evaluation.latex_tables import (
    export_all_tables,
    generate_main_results_table,
)
from src.evaluation.metrics import (
    SCHEMA_VERSION,
    build_result,
    compute_metrics,
    count_bio_violations,
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
    v = count_bio_violations(y_pred)
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
    # 2) Hà_Nội (idx 2..4): đoán lệch ranh giới chỉ có token 2 (boundary)
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
    corpus = make_synthetic_corpus(n_sentences=40, seed=7)
    w_list = [s["words"] for s in corpus]
    y_true = [s["tags"] for s in corpus]
    y_pred = make_dummy_predictions(y_true, error_rate=0.25, fp_rate=0.15, seed=7)

    rep_all = analyze_errors(w_list, y_true, y_pred)
    c = rep_all.counts
    assert c["boundary"] + c["type"] + c["false_positive"] == rep_all.num_pred_entities - c["correct"]

    cm, labels = entity_confusion_matrix(y_true, y_pred)
    assert cm.shape == (11, 11)
    assert labels[-1] == "O"

    df_samples = error_samples_table(rep_all, w_list, y_true, y_pred, n_per_type=2)
    assert len(df_samples) > 0

    len_rec = recall_by_entity_length(y_true, y_pred)
    assert "1" in len_rec
    oov_rec = recall_by_oov(w_list, y_true, y_pred, in_vocab=lambda w: w != "BN17")
    assert set(oov_rec.keys()) == {"has_oov", "all_known"}

    cmp_res = compare_models(y_true, y_pred, y_true)
    assert cmp_res["entity_counts"]["only_a"] == 0


def test_metrics_schema_and_latex_tables(tmp_path: Path) -> None:
    corpus = make_synthetic_corpus(n_sentences=20, seed=11)
    y_true = [s["tags"] for s in corpus]
    y_pred = make_dummy_predictions(y_true, error_rate=0.1, seed=11)

    res = evaluate_and_save(
        y_true, y_pred, experiment_name="DUMMY_bilstm_crf", model="bilstm_crf",
        split="test", group="main", out_dir=tmp_path, verbose=False,
    )
    assert res["schema_version"] == SCHEMA_VERSION
    loaded = load_metrics_json(tmp_path / "DUMMY_bilstm_crf_test.json")
    assert loaded["overall"]["micro"]["f1"] == res["overall"]["micro"]["f1"]

    # Khi allow_dummy=False, kết quả DUMMY bị lọc bỏ và bảng LaTeX hiển thị [CHỜ SỐ LIỆU: ...]
    tex_placeholder = generate_main_results_table([loaded], split="test", allow_dummy=False)
    assert "[CHỜ SỐ LIỆU:" in tex_placeholder

    # Khi result là tên thật, bảng LaTeX điền con số thật
    real_res = build_result(y_true, y_pred, experiment_name="bilstm_crf_full", model="bilstm_crf", split="test")
    tex_filled = generate_main_results_table([real_res], split="test", allow_dummy=False)
    assert f"{100 * real_res['overall']['micro']['f1']:.2f}" in tex_filled

    paths = export_all_tables(metrics_dir=tmp_path, out_dir=tmp_path / "tables", allow_dummy=False)
    assert len(paths) == 4


def test_demo_predictor_and_load_model() -> None:
    pred = load_model("dummy")
    assert isinstance(pred, DummyNERPredictor)
    text = "Bệnh nhân BN17 (nữ, 26 tuổi) đi chuyến bay VN0054 về Hà Nội ngày 2/3/2020, bị sốt."
    tokens = tokenize_vietnamese(text, mode="underthesea")
    assert len(tokens) > 5
    out = predict_sentence(text, predictor=pred)
    assert len(out["words"]) == len(out["tags"])
    detected_types = {row[3] for row in out["entities"]}
    for expected in ("PATIENT_ID", "GENDER", "AGE", "TRANSPORTATION", "DATE"):
        assert expected in detected_types
