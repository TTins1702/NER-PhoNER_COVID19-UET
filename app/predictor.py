"""Bộ tách từ, hàm nạp mô hình `load_model(path)` và bộ dự đoán cho Demo Gradio.

Tại sao viết như vậy:
- Tách riêng phần suy luận (`predictor.py`) khỏi giao diện (`demo.py`) để có thể kiểm thử tự động trong
  `tests/` mà không cần bật server web Gradio.
- Dữ liệu `PhoNER_COVID19` ở mức từ (`word-level`) nối các âm tiết của một từ ghép bằng dấu gạch dưới
  (ví dụ: `"Bệnh_viện"`, `"Hà_Nội"`, `"khó_thở"`). Khi người dùng nhập câu tiếng Việt tự nhiên vào Demo,
  ta dùng `underthesea.word_tokenize(..., format="text")` để tách từ về đúng quy chuẩn có dấu `"_"`
  trước khi đưa vào mô hình; nếu môi trường chưa cài `underthesea` thì tự động chuyển sang bộ tách từ
  dự phòng theo quy tắc từ ghép phổ biến.
- Hàm `load_model(path)` đóng vai trò điểm cắm (plug-in) theo đúng Hợp đồng giao diện (mục c):
  khi Tín (BiLSTM-CRF) hoặc Đức Anh (PhoBERT) có file checkpoint `.pt`, chỉ cần truyền đường dẫn vào
  `load_model(path)` mà không phải sửa code giao diện. Khi `path=None` hoặc `"dummy"`, hệ thống chạy
  `DummyNERPredictor` để thử nghiệm trọn vẹn luồng đầu vào -> đầu ra.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch

from src.data.bilstm_dataset import BiLSTMNERDataset, bilstm_collate_fn
from src.data.labels import (
    DISPLAY_NAMES,
    ENTITY_TYPES,
    ID2LABEL,
    LABEL2ID,
    decode_ids,
    extract_entities,
)
from src.data.vocab import CharVocab, WordVocab


class NERPredictorProtocol(Protocol):
    """Giao diện chung cho mọi mô hình cắm vào Demo: nhận batch câu đã tách từ, trả về nhãn BIO mức từ."""

    name: str

    def predict(self, batch_words: Sequence[Sequence[str]]) -> list[list[str]]:
        ...


# Một số từ ghép y tế/địa danh phổ biến dùng cho bộ tách từ dự phòng khi không có underthesea.
_FALLBACK_COMPOUNDS = (
    "Bệnh viện", "bệnh viện", "Bạch Mai", "Bệnh nhân", "bệnh nhân", "Hà Nội", "Đà Nẵng",
    "Hồ Chí Minh", "Việt Nam", "Bộ Y tế", "Sở Y tế", "Viện Pasteur", "khó thở", "viêm phổi",
    "huyết áp", "tiếp xúc", "cách ly", "dương tính", "âm tính", "phi công", "điều dưỡng",
    "du học sinh", "nhân viên", "chuyến bay", "triệu chứng", "bệnh nền",
)


def _fallback_word_tokenize(text: str) -> list[str]:
    """Bộ tách từ dự phòng: nối các cụm từ ghép tiếng Việt quen thuộc bằng '_' và tách dấu câu."""
    s = text.strip()
    if not s:
        return []
    for comp in _FALLBACK_COMPOUNDS:
        s = s.replace(comp, comp.replace(" ", "_"))
    # Tách dấu câu đứng sát chữ (nhưng giữ nguyên ngày tháng như 27/7, mã số 29A-12345, viết tắt N.H.N.)
    s = re.sub(r"([,;:!?()\"“”])", r" \1 ", s)
    s = re.sub(r"\.\s*$", " .", s)
    return [tok for tok in s.split() if tok]


def tokenize_vietnamese(text: str, mode: str = "underthesea") -> list[str]:
    """Tách câu tiếng Việt thành danh sách token theo quy chuẩn của PhoNER_COVID19.

    Args:
        text: câu tiếng Việt người dùng nhập.
        mode:
          - `"underthesea"`: tách từ bằng thư viện Underthesea (các âm tiết trong từ ghép nối bằng `_`),
            nếu người dùng đã gõ sẵn dấu `_` thì giữ nguyên; nếu thiếu thư viện thì dùng bộ dự phòng.
          - `"whitespace"`: tách theo khoảng trắng (dùng cho mức âm tiết hoặc câu đã tách từ sẵn).
    """
    text = text.strip()
    if not text:
        return []
    if mode == "whitespace" or "_" in text:
        return [tok for tok in re.sub(r"([,;:!?()])", r" \1 ", text).split() if tok]

    try:
        from underthesea import word_tokenize  # type: ignore

        segmented = word_tokenize(text, format="text")
        tokens = [tok for tok in segmented.split() if tok]
        if tokens:
            return tokens
    except Exception:  # noqa: BLE001
        pass
    return _fallback_word_tokenize(text)


# ---------------------------------------------------------------------------
# Mô hình giả (DummyNERPredictor) để chạy thử Demo khi chưa có checkpoint thật
# ---------------------------------------------------------------------------
class DummyNERPredictor:
    """Mô hình giả lập dựa trên quy tắc & từ điển mẫu để kiểm thử giao diện Demo."""

    name = "DUMMY (Mô hình giả chạy thử)"

    _SYMPTOMS = {
        "sốt", "ho", "khó_thở", "viêm_phổi", "đau_họng", "mệt_mỏi", "tiêu_chảy",
        "tăng_huyết_áp", "tiểu_đường", "suy_hô_hấp", "mất_vị_giác",
    }
    _JOBS = {
        "phi_công", "điều_dưỡng", "bác_sĩ", "du_học_sinh", "lái_xe", "nhân_viên",
        "công_nhân", "sinh_viên", "học_sinh", "giáo_viên", "tiếp_viên",
    }
    _ORGS = {
        "Bộ_Y_tế", "Sở_Y_tế", "Viện_Pasteur", "CDC", "WHO", "Vietnam_Airlines",
    }
    _LOCS = {
        "Hà_Nội", "Đà_Nẵng", "Việt_Nam", "Hồ_Chí_Minh", "TP.HCM", "Vĩnh_Phúc",
        "Bạch_Mai", "Bệnh_viện_Bạch_Mai", "Bệnh_viện_Đà_Nẵng", "Bệnh_viện_Chợ_Rẫy",
        "Anh", "Pháp", "Mỹ", "Nhật_Bản", "Hàn_Quốc", "Sơn_Lôi", "Hoàn_Kiếm",
    }

    def predict_one(self, words: Sequence[str]) -> list[str]:
        n = len(words)
        tags = ["O"] * n
        i = 0
        while i < n:
            w = words[i]
            wl = w.lower()

            # 1) Cụm 2 token: "Bộ Y_tế", "Bệnh_viện Bạch_Mai", "Bệnh_nhân 17"
            if i + 1 < n:
                pair = f"{w} {words[i + 1]}"
                if pair in ("Bộ Y_tế", "Sở Y_tế", "Viện Pasteur", "CDC Hà_Nội", "CDC Đà_Nẵng"):
                    tags[i], tags[i + 1] = "B-ORGANIZATION", "I-ORGANIZATION"
                    i += 2
                    continue
                if w in ("Bệnh_viện", "bệnh_viện", "quận", "huyện", "xã", "phường", "tỉnh", "thành_phố") and words[i + 1][:1].isupper():
                    tags[i], tags[i + 1] = "B-LOCATION", "I-LOCATION"
                    i += 2
                    continue
                if pair.lower() in ("khó thở", "viêm phổi", "đau họng", "tăng huyết_áp"):
                    tags[i], tags[i + 1] = "B-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE"
                    i += 2
                    continue

            # 2) Mã bệnh nhân: BN17, BN1347, hoặc số ngay sau từ "bệnh_nhân" / "BN"
            if re.fullmatch(r"BN\d+", w, flags=re.IGNORECASE):
                tags[i] = "B-PATIENT_ID"
            elif re.fullmatch(r"\d{1,5}", w) and i > 0 and words[i - 1].lower() in ("bệnh_nhân", "bn", "ca"):
                tags[i] = "B-PATIENT_ID"
            # 3) Tuổi: số đứng trước từ "tuổi"
            elif re.fullmatch(r"\d{1,3}", w) and i + 1 < n and words[i + 1].lower() == "tuổi":
                tags[i] = "B-AGE"
            # 4) Ngày tháng: dd/mm, dd/mm/yyyy, dd-mm
            elif re.fullmatch(r"\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?", w):
                tags[i] = "B-DATE"
            # 5) Giới tính
            elif wl in ("nam", "nữ"):
                tags[i] = "B-GENDER"
            # 6) Phương tiện (số hiệu chuyến bay VN54, VN0054, QH152, biển số)
            elif re.fullmatch(r"(VN|VJ|QH|SU|JL|QR|EK)\d{2,4}", w) or re.fullmatch(r"\d{2}[A-Z]-\d{4,5}", w):
                tags[i] = "B-TRANSPORTATION"
            # 7) Triệu chứng & bệnh lý
            elif wl in self._SYMPTOMS:
                tags[i] = "B-SYMPTOM_AND_DISEASE"
            # 8) Nghề nghiệp
            elif wl in self._JOBS:
                tags[i] = "B-JOB"
            # 9) Tổ chức
            elif w in self._ORGS:
                tags[i] = "B-ORGANIZATION"
            # 10) Địa điểm
            elif w in self._LOCS:
                tags[i] = "B-LOCATION"
            # 11) Tên người (từ ghép viết hoa các âm tiết hoặc viết tắt N.H.N. sau chị/anh/ông/bà)
            elif (
                ("_" in w and all(p[:1].isupper() for p in w.split("_") if p))
                or re.fullmatch(r"([A-ZĐ]\.){1,4}", w)
            ) and (i > 0 and words[i - 1].lower() in ("bệnh_nhân", "chị", "anh", "ông", "bà", "em", "ca")):
                tags[i] = "B-NAME"
            i += 1
        return tags

    def predict(self, batch_words: Sequence[Sequence[str]]) -> list[list[str]]:
        return [self.predict_one(words) for words in batch_words]


# ---------------------------------------------------------------------------
# Adapter nạp checkpoint thật cho BiLSTM-CRF (Tín) hoặc PhoBERT (Đức Anh)
# ---------------------------------------------------------------------------
def _import_from_spec(spec: str) -> Any:
    """Nạp lớp từ chuỗi `'module.path:ClassName'`."""
    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"Chuỗi định danh lớp phải có dạng 'module.path:ClassName', nhận '{spec}'")
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


class BiLSTMCheckpointPredictor:
    """Bọc mô hình BiLSTM-CRF của Tín thành bộ dự đoán theo đúng Hợp đồng giao diện."""

    def __init__(self, model: torch.nn.Module, word_vocab: WordVocab, char_vocab: CharVocab,
                 device: str = "cpu", name: str = "BiLSTM-CRF"):
        self.model = model.to(device).eval()
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.device = device
        self.name = name

    @classmethod
    def from_checkpoint(cls, ckpt: dict[str, Any], device: str = "cpu") -> "BiLSTMCheckpointPredictor":
        word_vocab = WordVocab.from_dict(ckpt["word_vocab"])
        char_vocab = CharVocab.from_dict(ckpt["char_vocab"])
        model_cls = _import_from_spec(ckpt.get("model_class", "src.models.bilstm_crf:BiLSTMCRF"))
        model = model_cls(**ckpt.get("model_kwargs", {}))
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, word_vocab, char_vocab, device=device, name=ckpt.get("name", "BiLSTM-CRF"))

    @torch.no_grad()
    def predict(self, batch_words: Sequence[Sequence[str]]) -> list[list[str]]:
        samples = [{"words": list(w)} for w in batch_words]
        ds = BiLSTMNERDataset(samples, self.word_vocab, self.char_vocab)
        batch = bilstm_collate_fn([ds[i] for i in range(len(ds))])
        word_ids = batch["word_ids"].to(self.device)
        char_ids = batch["char_ids"].to(self.device)
        mask = batch["mask"].to(self.device)

        if hasattr(self.model, "decode"):
            pred_ids = self.model.decode(word_ids, char_ids, mask)
        else:
            logits = self.model(word_ids, char_ids, mask)
            pred_ids = [
                logits[b, : int(batch["lengths"][b])].argmax(dim=-1).tolist()
                for b in range(len(batch_words))
            ]
        return [decode_ids(seq[: len(words)]) for seq, words in zip(pred_ids, batch_words)]


def load_model(path: str | Path | None = None, device: str = "cpu") -> NERPredictorProtocol:
    """Hàm nạp mô hình có thể thay thế được (plug-in) cho Demo Gradio.

    Hỗ trợ 4 chế độ (theo `docs/interface_contract.md`, mục c):
      1. `path` là `None`, `""`, hoặc `"dummy"`: trả về `DummyNERPredictor()`.
      2. File checkpoint chứa khóa `"predictor": "module.path:ClassName"`: gọi
         `ClassName.from_checkpoint(ckpt, device=device)`.
      3. File checkpoint của BiLSTM-CRF (`"model_type": "bilstm_crf"`): nạp qua `BiLSTMCheckpointPredictor`.
      4. Đối tượng đã lưu bằng `torch.save` có sẵn phương thức `.predict(batch_words)`.
    """
    if path is None or str(path).strip().lower() in ("", "dummy", "none"):
        return DummyNERPredictor()

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if hasattr(obj, "predict") and callable(obj.predict):
        if not hasattr(obj, "name"):
            obj.name = ckpt_path.stem
        return obj

    if isinstance(obj, dict):
        if "predictor" in obj:
            pred_cls = _import_from_spec(obj["predictor"])
            return pred_cls.from_checkpoint(obj, device=device)
        if obj.get("model_type") == "bilstm_crf" or ("word_vocab" in obj and "state_dict" in obj):
            return BiLSTMCheckpointPredictor.from_checkpoint(obj, device=device)

    raise ValueError(
        f"Checkpoint '{ckpt_path}' không khớp hợp đồng giao diện. "
        "Cần có khóa 'predictor' hoặc 'model_type'='bilstm_crf' hoặc phương thức .predict(batch_words)."
    )


def predict_sentence(
    text: str,
    predictor: NERPredictorProtocol | None = None,
    tokenize_mode: str = "underthesea",
) -> dict[str, Any]:
    """Xử lý trọn vẹn 1 câu tiếng Việt: tách từ -> dự đoán nhãn BIO -> trích thực thể & tô màu."""
    predictor = predictor or DummyNERPredictor()
    words = tokenize_vietnamese(text, mode=tokenize_mode)
    if not words:
        return {"words": [], "tags": [], "highlights": [], "entities": [], "model_name": getattr(predictor, "name", "")}

    tags = list(predictor.predict([words])[0])
    if len(tags) != len(words):
        raise RuntimeError(f"Mô hình trả về {len(tags)} nhãn cho {len(words)} từ (vi phạm hợp đồng giao diện)")
    for t in tags:
        if t not in LABEL2ID and isinstance(t, int) and t in ID2LABEL:
            tags = decode_ids(tags)
            break

    spans = extract_entities(tags)
    highlights: list[tuple[str, str | None]] = []
    entity_rows: list[list[Any]] = []
    cursor = 0
    for idx, (etype, start, end) in enumerate(spans, start=1):
        if start > cursor:
            plain = " ".join(w.replace("_", " ") for w in words[cursor:start])
            highlights.append((plain + " ", None))
        ent_surface = " ".join(w.replace("_", " ") for w in words[start:end])
        ent_tokens = " ".join(words[start:end])
        highlights.append((ent_surface, etype))
        highlights.append((" ", None))
        entity_rows.append([
            idx,
            ent_surface,
            ent_tokens,
            etype,
            DISPLAY_NAMES.get(etype, etype),
            f"[{start}, {end})",
        ])
        cursor = end

    if cursor < len(words):
        tail = " ".join(w.replace("_", " ") for w in words[cursor:])
        highlights.append((tail, None))

    token_rows = [[i, w, t] for i, (w, t) in enumerate(zip(words, tags))]
    return {
        "words": words,
        "tags": tags,
        "highlights": highlights,
        "entities": entity_rows,
        "token_rows": token_rows,
        "model_name": getattr(predictor, "name", "Custom Model"),
    }


__all__ = [
    "BiLSTMCheckpointPredictor",
    "DummyNERPredictor",
    "ENTITY_TYPES",
    "NERPredictorProtocol",
    "load_model",
    "predict_sentence",
    "tokenize_vietnamese",
]
