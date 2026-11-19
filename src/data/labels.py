"""Bộ 21 nhãn BIO cố định cho PhoNER_COVID19.

Tại sao viết như vậy:
- Nếu mỗi người tự sinh `label2id` từ dữ liệu (ví dụ `sorted(set(tags))`) thì thứ tự có thể khác nhau
  giữa các lần chạy/tập dữ liệu -> checkpoint của người này đọc sai nhãn của người kia. Vì vậy danh sách
  nhãn được viết cứng tại đây và mọi module đều import từ file này.
- `O` luôn có id = 0: tiện làm giá trị pad cho tags (CRF cần chỉ số hợp lệ ở vị trí pad).
- 10 loại thực thể xếp theo bảng chữ cái; mỗi loại có B- rồi I- đứng liền nhau
  (O, B-AGE, I-AGE, B-DATE, I-DATE, ...) để ma trận chuyển trạng thái CRF dễ đọc khi vẽ heatmap.
- Giữ tên nhãn đúng như trong dữ liệu (`NAME`, `JOB`); tên trong bài báo gốc (`PERSON_NAME`,
  `OCCUPATION`) chỉ dùng khi hiển thị/báo cáo qua `DISPLAY_NAMES`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# 10 loại thực thể theo đúng tên trong file dữ liệu, xếp theo bảng chữ cái.
ENTITY_TYPES: tuple[str, ...] = (
    "AGE",
    "DATE",
    "GENDER",
    "JOB",
    "LOCATION",
    "NAME",
    "ORGANIZATION",
    "PATIENT_ID",
    "SYMPTOM_AND_DISEASE",
    "TRANSPORTATION",
)

OUTSIDE = "O"
LABELS: tuple[str, ...] = (OUTSIDE,) + tuple(
    f"{prefix}-{etype}" for etype in ENTITY_TYPES for prefix in ("B", "I")
)
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS: int = len(LABELS)  # = 21
O_ID: int = LABEL2ID[OUTSIDE]  # = 0

# Tên hiển thị trong báo cáo (theo Truong et al., 2021).
DISPLAY_NAMES: dict[str, str] = {
    "AGE": "AGE",
    "DATE": "DATE",
    "GENDER": "GENDER",
    "JOB": "OCCUPATION",
    "LOCATION": "LOCATION",
    "NAME": "PERSON_NAME",
    "ORGANIZATION": "ORGANIZATION",
    "PATIENT_ID": "PATIENT_ID",
    "SYMPTOM_AND_DISEASE": "SYMPTOM_AND_DISEASE",
    "TRANSPORTATION": "TRANSPORTATION",
}

# Tên viết tắt cho trục biểu đồ (đỡ chồng chữ).
SHORT_NAMES: dict[str, str] = {
    "AGE": "AGE",
    "DATE": "DATE",
    "GENDER": "GEN",
    "JOB": "JOB",
    "LOCATION": "LOC",
    "NAME": "NAME",
    "ORGANIZATION": "ORG",
    "PATIENT_ID": "PID",
    "SYMPTOM_AND_DISEASE": "SYM",
    "TRANSPORTATION": "TRANS",
}

assert NUM_LABELS == 21, "PhoNER_COVID19 phải có đúng 21 nhãn BIO"


def split_label(label: str) -> tuple[str, str | None]:
    """Tách nhãn thành (tiền tố, loại). Ví dụ: 'B-LOCATION' -> ('B', 'LOCATION'), 'O' -> ('O', None)."""
    if label == OUTSIDE:
        return OUTSIDE, None
    prefix, sep, etype = label.partition("-")
    if not sep or prefix not in ("B", "I") or etype not in ENTITY_TYPES:
        raise ValueError(f"Nhãn không hợp lệ: '{label}'")
    return prefix, etype


def is_valid_label(label: str) -> bool:
    """True nếu nhãn thuộc 21 nhãn cố định."""
    return label in LABEL2ID


def encode_tags(tags: Sequence[str]) -> list[int]:
    """Chuỗi nhãn chữ -> chuỗi id. Báo lỗi ngay nếu gặp nhãn lạ."""
    try:
        return [LABEL2ID[t] for t in tags]
    except KeyError as err:
        raise ValueError(f"Nhãn không hợp lệ: {err}") from None


def decode_ids(ids: Iterable[int]) -> list[str]:
    """Chuỗi id -> chuỗi nhãn chữ (dùng sau Viterbi/argmax)."""
    return [ID2LABEL[int(i)] for i in ids]


def display_name(etype: str) -> str:
    """Tên hiển thị trong báo cáo, ví dụ 'NAME' -> 'PERSON_NAME'."""
    return DISPLAY_NAMES.get(etype, etype)


def extract_entities(tags: Sequence[str]) -> list[tuple[str, int, int]]:
    """Trích thực thể theo luật **strict IOB2** (giống `seqeval` mode='strict', scheme=IOB2).

    Trả về danh sách `(loại, start, end)` với `end` là chỉ số **không bao gồm** (kiểu slice Python).
    Luật:
      - Thực thể chỉ bắt đầu tại `B-X`, kéo dài qua các `I-X` liền sau **cùng loại X**.
      - `I-X` đứng sau `O`, sau `B-Y`/`I-Y` (Y khác X) hoặc ở đầu câu là **không hợp lệ** và bị bỏ qua
        (không tạo thực thể). Đây chính là khác biệt với chế độ "lỏng" của conlleval, nơi `I-X` lạc
        được coi như bắt đầu thực thể mới.
    Ví dụ: ['B-ORG', 'I-ORG', 'O', 'I-LOC'] -> [('ORG', 0, 2)]
    """
    entities: list[tuple[str, int, int]] = []
    cur_type: str | None = None
    cur_start = -1
    for i, tag in enumerate(tags):
        prefix, etype = split_label(tag) if tag != OUTSIDE else (OUTSIDE, None)
        if prefix == "I" and cur_type is not None and etype == cur_type:
            continue  # nối dài thực thể hiện tại
        # mọi trường hợp khác đều kết thúc thực thể đang mở (nếu có)
        if cur_type is not None:
            entities.append((cur_type, cur_start, i))
            cur_type = None
        if prefix == "B":
            cur_type, cur_start = etype, i
        # prefix == "I" lạc loài hoặc "O": không mở thực thể mới
    if cur_type is not None:
        entities.append((cur_type, cur_start, len(tags)))
    return entities


# --- Tên tương thích với code của Tín (chỉ thêm, không đổi tên cũ) ---
IGNORE_INDEX: int = -100  # nhãn "bỏ qua" khi tính loss (Tín dùng); BiLSTMNERDataset của Đạt vẫn pad nhãn bằng O_ID
decode_tags = decode_ids  # tên Tín gọi cho decode_ids