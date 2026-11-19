# Hợp đồng giao diện (Interface Contract) — gửi Tín và Đức Anh

> Người soạn: Đạt (Data Pipeline, Evaluation, Error Analysis, Demo).
> Mục đích: chốt **một** định dạng dữ liệu vào/ra giữa phần dữ liệu–đánh giá và hai mô hình,
> để ba người code song song mà không phải sửa lại khi ghép.
> Mọi thay đổi với file này phải báo cả nhóm trước khi commit.

---

## 0. Nhãn dùng chung (`src/data/labels.py`)

- 21 nhãn BIO cố định, **không** được tự tạo `label2id` riêng:
  `O`(0), `B-AGE`(1), `I-AGE`(2), `B-DATE`(3), `I-DATE`(4), `B-GENDER`(5), `I-GENDER`(6),
  `B-JOB`(7), `I-JOB`(8), `B-LOCATION`(9), `I-LOCATION`(10), `B-NAME`(11), `I-NAME`(12),
  `B-ORGANIZATION`(13), `I-ORGANIZATION`(14), `B-PATIENT_ID`(15), `I-PATIENT_ID`(16),
  `B-SYMPTOM_AND_DISEASE`(17), `I-SYMPTOM_AND_DISEASE`(18), `B-TRANSPORTATION`(19), `I-TRANSPORTATION`(20).
- Dùng: `from src.data.labels import LABELS, LABEL2ID, ID2LABEL, NUM_LABELS, ENTITY_TYPES`.
- Tên nhãn giữ đúng như trong dữ liệu (`NAME`, `JOB`). Tên hiển thị trong báo cáo
  (`PERSON_NAME`, `OCCUPATION`) lấy từ `DISPLAY_NAMES`.

## (a) Batch của `src/data/bilstm_dataset.py` (cho Tín)

```python
from src.data.bilstm_dataset import build_bilstm_dataloaders
loaders, word_vocab, char_vocab = build_bilstm_dataloaders(data_dir="data", level="word", batch_size=32)
batch = next(iter(loaders["train"]))
```

| Khóa | Kiểu | Shape | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `word_ids` | `torch.long` | `[B, L]` | id từ trong `WordVocab`; vị trí pad = `0` (`<pad>`), từ lạ = `1` (`<unk>`) |
| `char_ids` | `torch.long` | `[B, L, W]` | id ký tự trong `CharVocab`; pad (cả từ pad lẫn phần đuôi từ ngắn) = `0` |
| `mask` | `torch.bool` | `[B, L]` | `True` ở từ thật, `False` ở pad. Luôn có dạng `[True…True, False…False]` |
| `tags` | `torch.long` | `[B, L]` | id nhãn theo `LABEL2ID`; vị trí pad = `-100` (`IGNORE_INDEX`), **phải** bỏ qua bằng `mask` |
| `lengths` | `torch.long` (CPU) | `[B]` | số từ thật mỗi câu = `mask.sum(1)`, dùng cho `pack_padded_sequence(..., enforce_sorted=False)` |
| `words` | `list[list[str]]` | `B` câu | từ gốc, dùng để ghi file dự đoán |

- `B` = batch size, `L` = số từ của câu dài nhất **trong batch**, `W` = số ký tự của từ dài nhất **trong batch**
  (bị chặn bởi `max_word_len`, mặc định 30).
- Batch **không** bị sắp xếp lại: câu thứ `i` trong batch là câu thứ `i` của tập (khi `shuffle=False`),
  nên dự đoán ghép lại được với file gốc.
- Pad tag bằng `-100` (`IGNORE_INDEX`) để khớp `BiLSTMCRF`/`Trainer` của Tín; vị trí pad đã bị `mask` loại khỏi loss. Batch có thêm khóa `gold_tags` (list nhãn gốc theo câu) cho bước đánh giá.
  Khi ablation `use_crf=False`: `loss = F.cross_entropy(emissions[mask], tags[mask])`.
- Nhúng ký tự: `nn.Embedding(len(char_vocab), dim, padding_idx=char_vocab.pad_id)`;
  nhúng từ: `nn.Embedding(len(word_vocab), dim, padding_idx=word_vocab.pad_id)`.

## (b) Định dạng dự đoán `y_pred` (cho cả Tín và Đức Anh)

```python
y_true: list[list[str]]   # nhãn gốc, mức từ
y_pred: list[list[str]]   # nhãn dự đoán, mức từ
assert len(y_pred) == len(y_true)
assert all(len(p) == len(t) == len(words) for p, t, words in zip(y_pred, y_true, words_list))
```

- Mỗi phần tử là **chuỗi nhãn BIO dạng chuỗi ký tự** (`"B-LOCATION"`, không phải id), lấy từ `ID2LABEL`.
- Độ dài mỗi câu dự đoán **bằng đúng `len(words)`** của câu gốc trong file JSON.
  - BiLSTM-CRF: bỏ phần pad theo `mask` (Viterbi trả list độ dài thật là đúng).
  - PhoBERT: chỉ lấy dự đoán tại **subword đầu tiên** của mỗi từ (first-subword); câu bị cắt ở
    `max_seq_length=256` thì các từ bị cắt điền `"O"` để giữ đủ độ dài (và ghi lại số câu bị cắt vào `extra`).
- Không tự sửa nhãn BIO trước khi đưa vào: `metrics.py` sẽ đếm vi phạm BIO trên dự đoán thô.

### File dự đoán (JSONL) — `outputs/predictions/<experiment_name>_<split>.jsonl`

Mỗi dòng một câu, đúng thứ tự file gốc:

```json
{"id": 0, "words": ["Bệnh_nhân", "BN17", "..."], "gold": ["O", "B-PATIENT_ID", "..."], "pred": ["O", "B-PATIENT_ID", "..."]}
```

Ghi/đọc bằng:

```python
from src.evaluation.predictions import save_predictions, load_predictions
save_predictions("outputs/predictions/bilstm_crf_full_test.jsonl", words_list, y_true, y_pred)
words_list, y_true, y_pred = load_predictions("outputs/predictions/bilstm_crf_full_test.jsonl")
```

(Thư mục `outputs/predictions/` nằm trong `.gitignore` vì chứa câu của dữ liệu.)

### Chấm điểm và xuất JSON

```python
from src.evaluation.metrics import evaluate_and_save
result = evaluate_and_save(
    y_true, y_pred,
    experiment_name="bilstm_crf_full",      # tên file: outputs/metrics/bilstm_crf_full_test.json
    model="bilstm_crf",                      # "bilstm_crf" | "phobert" | "llm_fewshot" | ...
    group="main",                            # "main" | "ablation_bilstm" | "ablation_phobert" | "other"
    split="test", level="word",
    description="BiLSTM + CharCNN + CRF",    # nhãn dòng trong bảng LaTeX
    is_ablation_reference=True,              # mô hình đầy đủ để tính ΔF1 trong bảng ablation
    config=cfg_dict,                         # siêu tham số (dict tùy ý, JSON-serializable)
    extra={"train_time_min": 42.0, "gpu": "T4", "wandb_url": "..."},
)
```

Hoặc từ dòng lệnh với file dự đoán:

```bash
python -m src.evaluation.metrics --pred outputs/predictions/bilstm_crf_full_test.jsonl \
    --experiment-name bilstm_crf_full --model bilstm_crf --group main --split test
```

### Schema JSON cố định (`schema_version = "1.0"`)

```json
{
  "schema_version": "1.0",
  "experiment_name": "bilstm_crf_full",
  "description": "BiLSTM + CharCNN + CRF",
  "model": "bilstm_crf",
  "group": "main",
  "is_ablation_reference": true,
  "split": "test",
  "level": "word",
  "eval_mode": "seqeval-strict-IOB2",
  "repaired_bio": false,
  "num_sentences": 3000,
  "num_tokens": 85678,
  "num_gold_entities": 0,
  "num_pred_entities": 0,
  "overall": {
    "micro":    {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    "macro":    {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
  },
  "per_entity": {
    "AGE":  {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
    "...":  "đủ 10 loại, kể cả loại không xuất hiện (support = 0)"
  },
  "bio_violations": {
    "pred": {"total": 0, "by_type": {"AGE": 0, "...": 0}, "sentences_with_violation": 0, "per_1k_tokens": 0.0},
    "gold": {"total": 0, "by_type": {"AGE": 0, "...": 0}, "sentences_with_violation": 0, "per_1k_tokens": 0.0}
  },
  "config": {},
  "extra": {},
  "created_at": "2026-11-19T17:00:00+07:00"
}
```

- Mọi số P/R/F1 ở thang `[0, 1]` (bảng LaTeX nhân 100).
- `macro` = trung bình không trọng số của F1 10 loại (theo `seqeval`), `micro` = gộp TP/FP/FN của mọi thực thể.

## (c) Checkpoint để cắm vào Demo (`app/predictor.py::load_model`)

Demo nhận một trong các dạng `torch.save(...)` sau:

1. **Khuyến nghị (cả hai mô hình):** dict có khóa `"predictor": "module.path:ClassName"`. Lớp đó cài
   `@classmethod from_checkpoint(cls, ckpt: dict, device: str)` trả về đối tượng có
   `predict(batch_words: list[list[str]]) -> list[list[str]]` (đúng định dạng `y_pred` ở trên).
2. **BiLSTM-CRF (adapter có sẵn):** dict
   `{"model_type": "bilstm_crf", "model_class": "src.models.bilstm_crf:BiLSTMCRF", "model_kwargs": {...},
   "state_dict": model.state_dict(), "word_vocab": word_vocab.to_dict(), "char_vocab": char_vocab.to_dict()}`.
   Lớp mô hình cần có `decode(word_ids, char_ids, mask) -> list[list[int]]` (id nhãn, độ dài thật).
3. Một đối tượng Python bất kỳ có sẵn phương thức `predict(batch_words)` như trên.

Không có checkpoint thì demo chạy chế độ **mô hình giả** (`--model dummy`).
