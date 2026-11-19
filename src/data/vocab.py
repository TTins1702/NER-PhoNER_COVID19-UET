"""Từ điển từ (WordVocab) và từ điển ký tự (CharVocab) cho BiLSTM-CRF.

Tại sao viết như vậy:
- WordVocab **chỉ xây từ tập train**: nếu lấy cả dev/test thì mô hình "biết trước" từ của tập đánh giá
  (rò rỉ dữ liệu), tỉ lệ OOV báo cáo sẽ thấp giả tạo và kết quả không phản ánh khả năng tổng quát.
- `min_freq`: từ xuất hiện ít hơn ngưỡng bị coi là `<unk>`. Nhờ vậy embedding của `<unk>` được học
  từ chính các từ hiếm trong train -> lúc test gặp từ lạ, mô hình có một vector "từ hiếm" có nghĩa,
  thay vì một vector ngẫu nhiên chưa từng được cập nhật.
- `<pad>` = 0, `<unk>` = 1 cố định ở cả hai vocab để dùng `padding_idx=0` trong `nn.Embedding`.
- CharVocab dựng từ một bảng ký tự cố định (chữ cái tiếng Việt có dấu hoa/thường, chữ số, dấu câu)
  rồi có thể bổ sung ký tự thực tế gặp trong train. Ký tự giúp xử lý OOV: từ chưa gặp vẫn được biểu
  diễn qua ký tự (chữ hoa đầu -> tên riêng, "BN" + số -> mã bệnh nhân).
- Mọi chuỗi được chuẩn hóa NFC trước khi tra để tránh một chữ có hai cách mã hóa Unicode.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# WordVocab
# ---------------------------------------------------------------------------
class WordVocab:
    """Ánh xạ từ <-> id. Xây bằng `WordVocab.build(train_sentences, min_freq=...)`.

    Tham số:
        lower: đưa từ về chữ thường trước khi tra (thông tin hoa/thường vẫn còn ở CharEncoder).
        normalize_digits: thay mọi chữ số bằng '0' (ví dụ '19/3' -> '00/0') để gộp các ngày/số hiệu.
    """

    def __init__(self, itos: Sequence[str] | None = None, lower: bool = False,
                 normalize_digits: bool = False, min_freq: int = 1,
                 freqs: dict[str, int] | None = None):
        self.lower = lower
        self.normalize_digits = normalize_digits
        self.min_freq = min_freq
        self.itos: list[str] = list(itos) if itos is not None else [PAD_TOKEN, UNK_TOKEN]
        if self.itos[:2] != [PAD_TOKEN, UNK_TOKEN]:
            raise ValueError("Hai phần tử đầu của vocab phải là <pad>, <unk>")
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        self.freqs: dict[str, int] = dict(freqs or {})

    # -- xây dựng -----------------------------------------------------------
    def normalize(self, word: str) -> str:
        """Chuẩn hóa một từ theo đúng cấu hình của vocab (dùng chung khi build và khi encode)."""
        word = _nfc(word)
        if self.lower:
            word = word.lower()
        if self.normalize_digits:
            word = re.sub(r"\d", "0", word)
        return word

    @classmethod
    def build(cls, sentences: Iterable[Sequence[str]], min_freq: int = 1, lower: bool = False,
              normalize_digits: bool = False, max_size: int | None = None) -> "WordVocab":
        """Xây vocab từ danh sách câu (chỉ truyền câu của tập train!).

        Từ được xếp theo tần suất giảm dần, hòa thì theo bảng chữ cái -> kết quả tất định
        (cùng dữ liệu luôn cho cùng id, tái lập được).
        """
        if min_freq < 1:
            raise ValueError("min_freq phải >= 1")
        vocab = cls(lower=lower, normalize_digits=normalize_digits, min_freq=min_freq)
        counter = Counter(vocab.normalize(w) for sent in sentences for w in sent)
        kept = [(w, c) for w, c in counter.items() if c >= min_freq and w not in vocab.stoi]
        kept.sort(key=lambda wc: (-wc[1], wc[0]))
        if max_size is not None:
            kept = kept[: max(0, max_size - 2)]
        for w, _ in kept:
            vocab.stoi[w] = len(vocab.itos)
            vocab.itos.append(w)
        vocab.freqs = dict(counter)
        return vocab

    # -- tra cứu ------------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        return self.normalize(word) in self.stoi

    def token_to_id(self, word: str) -> int:
        return self.stoi.get(self.normalize(word), UNK_ID)

    def encode(self, words: Sequence[str]) -> list[int]:
        """Danh sách từ -> danh sách id (từ lạ -> 1)."""
        return [self.token_to_id(w) for w in words]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[int(i)] for i in ids]

    def oov_rate(self, sentences: Iterable[Sequence[str]]) -> float:
        """Tỉ lệ token (không phải kiểu từ) bị map thành <unk>; dùng trong EDA và phân tích lỗi."""
        total = unk = 0
        for sent in sentences:
            for w in sent:
                total += 1
                unk += self.normalize(w) not in self.stoi
        return unk / total if total else 0.0

    # -- lưu / nạp ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "WordVocab", "itos": self.itos, "lower": self.lower,
                "normalize_digits": self.normalize_digits, "min_freq": self.min_freq}

    @classmethod
    def from_dict(cls, d: dict) -> "WordVocab":
        return cls(itos=d["itos"], lower=d.get("lower", False),
                   normalize_digits=d.get("normalize_digits", False), min_freq=d.get("min_freq", 1))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "WordVocab":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# CharVocab
# ---------------------------------------------------------------------------
# 12 nguyên âm cơ sở (kể cả ă, â, ê, ô, ơ, ư) x 6 thanh điệu.
_VI_BASE_VOWELS = "aăâeêioôơuưy"
# Dấu thanh dạng tổ hợp: không dấu, sắc, huyền, hỏi, ngã, nặng.
_VI_TONE_MARKS = ("", "\u0301", "\u0300", "\u0309", "\u0303", "\u0323")
_EXTRA_PUNCT = "“”‘’–—…°«»•·"


def vietnamese_letters() -> list[str]:
    """Toàn bộ chữ cái tiếng Việt (hoa + thường, đủ dấu) và chữ Latin f, j, w, z."""
    letters: list[str] = []
    for v in _VI_BASE_VOWELS:
        for tone in _VI_TONE_MARKS:
            letters.append(_nfc(v + tone))
    letters += list("bcdđghklmnpqrstvx") + list("fjwz")
    out: list[str] = []
    for ch in letters:
        for c in (ch, ch.upper()):
            if c not in out:
                out.append(c)
    return out


def default_charset() -> list[str]:
    """Bảng ký tự mặc định: chữ tiếng Việt, chữ số, dấu câu ASCII, dấu câu thường gặp, '_' và ' '."""
    chars = vietnamese_letters() + list(string.digits) + list(string.punctuation) + list(_EXTRA_PUNCT)
    chars.append(" ")  # phòng khi token chứa khoảng trắng
    seen: list[str] = []
    for c in chars:
        if c not in seen:
            seen.append(c)
    return seen


class CharVocab:
    """Ánh xạ ký tự <-> id. `<pad>`=0, `<unk>`=1, sau đó là bảng ký tự mặc định."""

    def __init__(self, itos: Sequence[str] | None = None):
        if itos is None:
            itos = [PAD_TOKEN, UNK_TOKEN] + default_charset()
        self.itos: list[str] = list(itos)
        if self.itos[:2] != [PAD_TOKEN, UNK_TOKEN]:
            raise ValueError("Hai phần tử đầu của vocab phải là <pad>, <unk>")
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, sentences: Iterable[Sequence[str]] | None = None, min_freq: int = 1,
              extend: bool = True) -> "CharVocab":
        """Tạo vocab ký tự mặc định; nếu `extend=True` thì thêm ký tự lạ gặp trong train (>= min_freq).

        Ký tự được thêm theo thứ tự Unicode để kết quả tất định.
        """
        vocab = cls()
        if sentences is not None and extend:
            counter = Counter(c for sent in sentences for w in sent for c in _nfc(w))
            new_chars = sorted(c for c, n in counter.items() if n >= min_freq and c not in vocab.stoi)
            for c in new_chars:
                vocab.stoi[c] = len(vocab.itos)
                vocab.itos.append(c)
        return vocab

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, ch: str) -> bool:
        return ch in self.stoi

    def encode_word(self, word: str, max_len: int | None = None) -> list[int]:
        """Một từ -> danh sách id ký tự (cắt bớt nếu dài hơn `max_len`). Từ rỗng -> [<unk>]."""
        ids = [self.stoi.get(c, UNK_ID) for c in _nfc(word)]
        if max_len is not None:
            ids = ids[:max_len]
        return ids or [UNK_ID]

    def encode(self, words: Sequence[str], max_len: int | None = None) -> list[list[int]]:
        return [self.encode_word(w, max_len) for w in words]

    def to_dict(self) -> dict:
        return {"type": "CharVocab", "itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "CharVocab":
        return cls(itos=d["itos"])

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CharVocab":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- Tên tương thích với code của Tín ---
PAD_IDX = PAD_ID
UNK_IDX = UNK_ID