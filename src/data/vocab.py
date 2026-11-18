from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

_V = TypeVar("_V", bound="_BaseVocab")


class _BaseVocab:
    """Shared vocabulary storage and serialization helpers."""

    def __init__(self, itos: Sequence[str] | None = None) -> None:
        tokens = list(itos) if itos is not None else [PAD_TOKEN, UNK_TOKEN]
        if len(tokens) < 2 or tokens[PAD_IDX] != PAD_TOKEN or tokens[UNK_IDX] != UNK_TOKEN:
            raise ValueError(
                f"Vocabulary must start with {[PAD_TOKEN, UNK_TOKEN]}, got {tokens[:2]}"
            )
        self.itos: list[str] = tokens
        self.stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict[str, Any]:
        return {"itos": list(self.itos)}

    @classmethod
    def from_dict(cls: type[_V], data: dict[str, Any]) -> _V:
        return cls(itos=data["itos"])


class WordVocab(_BaseVocab):
    """Word-level vocabulary mapping tokens to integer indices."""

    @classmethod
    def build(cls: type[_V], sentences: Iterable[Sequence[str]], min_freq: int = 1) -> _V:
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts.update(sentence)

        itos = [PAD_TOKEN, UNK_TOKEN]
        for token, freq in counts.items():
            if freq >= min_freq and token not in (PAD_TOKEN, UNK_TOKEN):
                itos.append(token)
        return cls(itos=itos)

    def encode(self, words: Sequence[str]) -> list[int]:
        return [self.stoi.get(word, UNK_IDX) for word in words]


class CharVocab(_BaseVocab):
    """Character-level vocabulary mapping characters inside words to indices."""

    @classmethod
    def build(cls: type[_V], sentences: Iterable[Sequence[str]]) -> _V:
        itos = [PAD_TOKEN, UNK_TOKEN]
        seen = {PAD_TOKEN, UNK_TOKEN}
        for sentence in sentences:
            for word in sentence:
                for ch in word:
                    if ch not in seen:
                        seen.add(ch)
                        itos.append(ch)
        return cls(itos=itos)

    def encode_word(self, word: str, max_len: int | None = None) -> list[int]:
        if max_len is not None:
            word = word[:max_len]
        return [self.stoi.get(ch, UNK_IDX) for ch in word]
