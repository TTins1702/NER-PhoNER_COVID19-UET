"""Dữ liệu giả và dự đoán giả để chạy thử pipeline đánh giá khi mô hình thật chưa xong.

Tại sao viết như vậy:
- Tín và Đức Anh chưa có checkpoint, nhưng notebook 04, bảng LaTeX và demo cần đầu vào đúng định dạng
  `y_pred: list[list[str]]` để kiểm thử. Ta sinh dự đoán giả bằng cách **làm nhiễu nhãn gold** theo
  từng loại lỗi (bỏ sót, nhầm loại, lệch ranh giới, báo nhầm, vi phạm BIO) với tỉ lệ biết trước, nên
  có thể kiểm tra error_analysis đếm đúng hay không.
- Mọi kết quả sinh từ đây **chỉ để chạy thử**, không bao giờ được đưa vào báo cáo; tên thí nghiệm luôn
  có tiền tố `DUMMY_` và lưu ở thư mục riêng `outputs/metrics/dummy/`.
- Có `seed` để mỗi lần chạy cho cùng một kết quả (tái lập được).
"""

from __future__ import annotations

import random
from typing import Sequence

from src.data.labels import ENTITY_TYPES, extract_entities

# Hồ sơ lỗi giả lập cho từng "mô hình" (chỉ là tham số nhiễu, KHÔNG phải kết quả thật).
DUMMY_PROFILES: dict[str, dict[str, float]] = {
    "DUMMY_bilstm_crf": {"error_rate": 0.20, "fp_rate": 0.10, "bio_violation_rate": 0.0, "seed": 1},
    "DUMMY_bilstm_softmax": {"error_rate": 0.24, "fp_rate": 0.12, "bio_violation_rate": 0.06, "seed": 2},
    "DUMMY_phobert": {"error_rate": 0.10, "fp_rate": 0.05, "bio_violation_rate": 0.01, "seed": 3},
}

# Từ điển điền mẫu cho câu giả (mức từ, âm tiết nối bằng "_").
_FILLERS: dict[str, list[list[str]]] = {
    "PATIENT_ID": [["BN17"], ["91"], ["416"], ["BN243"], ["1347"]],
    "GENDER": [["nam"], ["nữ"], ["Nam"], ["Nữ"]],
    "AGE": [["26"], ["30"], ["55"], ["64"], ["7"]],
    "JOB": [["phi_công"], ["điều_dưỡng"], ["du_học_sinh"], ["lái_xe"], ["nhân_viên", "bán_hàng"]],
    "LOCATION": [["Hà_Nội"], ["Đà_Nẵng"], ["quận", "Hoàn_Kiếm"], ["Bệnh_viện", "Bạch_Mai"],
                 ["thành_phố", "Hồ_Chí_Minh"], ["xã", "Sơn_Lôi"]],
    "DATE": [["27/7"], ["25", "-", "7"], ["ngày", "31/7"], ["3/2020"]],
    "SYMPTOM_AND_DISEASE": [["sốt"], ["ho"], ["khó", "thở"], ["viêm", "phổi"], ["tăng", "huyết_áp"]],
    "NAME": [["Nguyễn_Văn_A"], ["Nhung"], ["T."], ["Trần_Thị_B"]],
    "ORGANIZATION": [["Bộ", "Y_tế"], ["CDC", "Hà_Nội"], ["Viện", "Pasteur"], ["Sở", "Y_tế"]],
    "TRANSPORTATION": [["VN54"], ["VN0054"], ["SU", "290"], ["29A", "-", "12345"]],
}

# Mẫu câu: chuỗi token, "{TYPE}" là chỗ điền thực thể.
_TEMPLATES: list[list[str]] = [
    "Bệnh_nhân {PATIENT_ID} ( {GENDER} , {AGE} tuổi , {JOB} ) trú tại {LOCATION} .".split(),
    "Ngày {DATE} , bệnh_nhân {PATIENT_ID} có biểu_hiện {SYMPTOM_AND_DISEASE} và được đưa tới {LOCATION} .".split(),
    "{NAME} đi chuyến bay {TRANSPORTATION} từ {LOCATION} về {LOCATION} ngày {DATE} .".split(),
    "{ORGANIZATION} cho biết bệnh_nhân {PATIENT_ID} có tiền_sử {SYMPTOM_AND_DISEASE} .".split(),
    "Hiện sức_khỏe của bệnh_nhân ổn_định , không còn {SYMPTOM_AND_DISEASE} .".split(),
    "Các ca tiếp_xúc đã được cách_ly tại nhà theo hướng_dẫn của {ORGANIZATION} .".split(),
    "Tối {DATE} , {NAME} ( {AGE} tuổi ) làm {JOB} tại {LOCATION} .".split(),
]


def make_synthetic_corpus(n_sentences: int = 300, seed: int = 0) -> list[dict]:
    """Sinh câu giả `{"words", "tags"}` mức từ có đủ 10 loại thực thể (dùng khi chưa tải dữ liệu)."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n_sentences):
        words, tags = [], []
        for tok in rng.choice(_TEMPLATES):
            if tok.startswith("{") and tok.endswith("}"):
                etype = tok[1:-1]
                ent = rng.choice(_FILLERS[etype])
                words += ent
                tags += [f"B-{etype}"] + [f"I-{etype}"] * (len(ent) - 1)
            else:
                words.append(tok)
                tags.append("O")
        samples.append({"words": words, "tags": tags})
    return samples


def _write_span(tags: list[str], etype: str, start: int, end: int) -> None:
    tags[start] = f"B-{etype}"
    for i in range(start + 1, end):
        tags[i] = f"I-{etype}"


def perturb_tags(gold: Sequence[str], rng: random.Random, error_rate: float = 0.15,
                 fp_rate: float = 0.05, bio_violation_rate: float = 0.0) -> list[str]:
    """Tạo một chuỗi dự đoán giả từ chuỗi gold. Độ dài luôn bằng gold, nhãn luôn thuộc 21 nhãn.

    Với mỗi thực thể gold, xác suất `error_rate` bị một trong ba lỗi: bỏ sót / nhầm loại / lệch ranh giới.
    Với mỗi câu, xác suất `fp_rate` thêm một thực thể báo nhầm trên một token `O`.
    Với mỗi thực thể còn lại, xác suất `bio_violation_rate` đổi `B-X` đầu thành `I-X` (lỗi kiểu Softmax).
    """
    pred = list(gold)
    n = len(pred)
    for etype, s, e in extract_entities(gold):
        if rng.random() < error_rate:
            kind = rng.choice(("drop", "type", "boundary"))
            if kind == "drop":
                for i in range(s, e):
                    pred[i] = "O"
            elif kind == "type":
                other = rng.choice([t for t in ENTITY_TYPES if t != etype])
                _write_span(pred, other, s, e)
            else:
                if e - s > 1 and rng.random() < 0.5:  # co lại: bỏ token cuối
                    pred[e - 1] = "O"
                elif e < n and pred[e] == "O":         # nở ra: nuốt thêm token sau
                    _write_span(pred, etype, s, e + 1)
                elif e - s > 1:
                    pred[e - 1] = "O"
                else:                                   # không co/nở được -> bỏ sót
                    pred[s] = "O"
        elif rng.random() < bio_violation_rate:
            pred[s] = f"I-{etype}"
    if rng.random() < fp_rate:
        free = [i for i in range(n) if gold[i] == "O" and pred[i] == "O"
                and (i == 0 or pred[i - 1] == "O") and (i + 1 == n or pred[i + 1] == "O")]
        if free:
            idx = rng.choice(free)
            _write_span(pred, rng.choice(ENTITY_TYPES), idx, idx + 1)
    return pred


def make_dummy_predictions(y_true: Sequence[Sequence[str]], error_rate: float = 0.15, fp_rate: float = 0.05,
                           bio_violation_rate: float = 0.0, seed: int = 0) -> list[list[str]]:
    """Áp `perturb_tags` cho cả tập (tái lập được nhờ `seed`)."""
    rng = random.Random(seed)
    return [perturb_tags(g, rng, error_rate, fp_rate, bio_violation_rate) for g in y_true]


def make_profile_predictions(y_true: Sequence[Sequence[str]], profile: str) -> list[list[str]]:
    """Dự đoán giả theo một hồ sơ trong `DUMMY_PROFILES`."""
    p = DUMMY_PROFILES[profile]
    return make_dummy_predictions(y_true, p["error_rate"], p["fp_rate"], p["bio_violation_rate"], int(p["seed"]))
