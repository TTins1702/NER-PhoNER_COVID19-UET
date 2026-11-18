from src.evaluation.error_analysis import categorize_errors, extract_entities
from src.evaluation.metrics import compute_ner_metrics, count_bio_violations


def test_count_bio_violations_and_strict_seqeval_with_illegal_i_tags() -> None:
    y_true = [["O", "B-LOCATION", "I-LOCATION", "O", "B-AGE"]]
    y_pred = [["I-LOCATION", "B-LOCATION", "I-NAME", "O", "B-AGE"]]
    # 2 vi phạm: vị trí 0 (đầu câu là I-LOCATION) và vị trí 2 (B-LOCATION -> I-NAME)
    assert count_bio_violations(y_pred) == 2

    metrics = compute_ner_metrics(y_true, y_pred)
    assert metrics["bio_violations"] == 2
    assert "AGE" in metrics["per_entity"] and metrics["per_entity"]["AGE"]["f1"] == 1.0
    assert len(metrics["per_entity"]) == 10
    assert 0.0 <= metrics["micro_f1"] <= 1.0


def test_extract_entities_and_categorize_errors() -> None:
    words = [["BN1", "ở", "quận", "Cầu_Giấy", "sốt", "cao"]]
    y_true = [["B-PATIENT_ID", "O", "B-LOCATION", "I-LOCATION", "B-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE"]]
    # BN1 -> B-NAME (nhầm loại), chỉ bắt được Cầu_Giấy (lệch ranh giới), bỏ sót triệu chứng (FN), "ở" -> B-JOB (FP)
    y_pred = [["B-NAME", "B-JOB", "O", "B-LOCATION", "O", "O"]]

    assert extract_entities(y_true[0]) == [
        ("PATIENT_ID", 0, 1),
        ("LOCATION", 2, 4),
        ("SYMPTOM_AND_DISEASE", 4, 6),
    ]

    report = categorize_errors(words, y_true, y_pred)
    assert report["counts"]["type_errors"] == 1
    assert report["counts"]["boundary_errors"] == 1
    assert report["counts"]["false_negatives"] == 1
    assert report["counts"]["false_positives"] == 1
