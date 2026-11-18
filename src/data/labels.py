from collections.abc import Sequence

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

LABELS: list[str] = ["O"] + [
    f"{prefix}-{entity_type}"
    for entity_type in ENTITY_TYPES
    for prefix in ("B", "I")
]

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}
IGNORE_INDEX: int = -100


def encode_tags(tags: Sequence[str]) -> list[int]:
    """Convert BIO string tags to fixed integer label IDs."""
    return [LABEL2ID[tag] for tag in tags]


def decode_tags(tag_ids: Sequence[int]) -> list[str]:
    """Convert integer label IDs back to BIO strings, skipping IGNORE_INDEX."""
    return [ID2LABEL[int(tag_id)] for tag_id in tag_ids if int(tag_id) != IGNORE_INDEX]
