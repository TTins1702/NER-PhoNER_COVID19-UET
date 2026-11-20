# NER-PhoNER_COVID19-UET

Hệ thống nhận dạng thực thể có tên (Named Entity Recognition - NER) tiếng Việt trên bộ dữ liệu **PhoNER_COVID19** (10 loại thực thể, 21 nhãn BIO), phục vụ đồ án môn Học sâu (Deep Learning) — Đề tài 7.

## Phân công Thành viên

| Thành viên | Phụ trách chính | Thành phần |
| :--- | :--- | :--- |
| **Đạt** (`dat14148`) | Data Pipeline, EDA, Evaluation & Error Analysis | `src/data/` (`download.py`, `labels.py`, `vocab.py`, `bilstm_dataset.py`), `src/evaluation/`, `notebooks/01_eda_data_stats.ipynb`, `notebooks/04_error_analysis.ipynb` |
| **Tín** (`TTins1702`) | BiLSTM-CRF từ đầu (From Scratch) & Ablation | `src/models/char_encoder.py`, `src/models/crf.py`, `src/models/bilstm_crf.py`, `src/training/trainer.py`, `configs/bilstm_crf.yaml`, `notebooks/02_train_bilstm_crf.ipynb` |
| **Đức Anh** (`AnhLe247`) | Fine-tune PhoBERT & Ablation | `src/data/phobert_dataset.py`, `src/models/phobert_ner.py`, `configs/phobert.yaml`, `notebooks/03_train_phobert.ipynb` |

## Cấu trúc Thư mục

```text
NER-UET/
├── configs/
│   ├── bilstm_crf.yaml              # Cấu hình siêu tham số và cờ ablation cho BiLSTM-CRF
│   └── phobert.yaml                 # Cấu hình siêu tham số và cờ ablation cho PhoBERT
├── notebooks/
│   ├── 01_eda_data_stats.ipynb      # [Đạt] Tải dữ liệu và thống kê EDA
│   ├── 02_train_bilstm_crf.ipynb    # [Tín] Huấn luyện BiLSTM-CRF & Ablation trên Kaggle/Colab
│   ├── 03_train_phobert.ipynb       # [Đức Anh] Huấn luyện PhoBERT & Ablation trên Kaggle/Colab
│   └── 04_error_analysis.ipynb      # [Đạt] Trực quan hóa F1, ma trận CRF & phân tích lỗi
├── scripts/
│   ├── train.py                     # CLI huấn luyện và lưu metrics JSON
│   └── evaluate.py                  # CLI đánh giá checkpoint trên tập dev/test
├── src/
│   ├── data/
│   │   ├── download.py              # Tải PhoNER_COVID19 và đọc JSONL/CoNLL theo split/level
│   │   ├── labels.py                # Thứ tự cố định 21 nhãn BIO và IGNORE_INDEX = -100
│   │   ├── vocab.py                 # WordVocab (<pad>, <unk>) và CharVocab (Unicode NFC)
│   │   ├── bilstm_dataset.py        # BiLSTMDataset và dynamic 3D char padding
│   │   └── phobert_dataset.py       # PhoBERTDataset và first-subword alignment
│   ├── models/
│   │   ├── char_encoder.py          # Bộ mã hóa ký tự (1D-CNN hoặc BiLSTM)
│   │   ├── crf.py                   # Tầng LinearChainCRF cài đặt từ đầu (Forward + Viterbi)
│   │   ├── bilstm_crf.py            # Mô hình BiLSTM-CRF kèm cờ use_char / use_crf
│   │   └── phobert_ner.py           # Mô hình PhoBERT-base-v2 kèm Softmax / CRF head
│   ├── training/
│   │   └── trainer.py               # Bộ huấn luyện chung, differential LR, early stopping
│   ├── evaluation/
│   │   ├── metrics.py               # Chấm điểm strict IOB2 qua seqeval & đếm vi phạm BIO
│   │   ├── error_analysis.py        # Phân loại 4 nhóm lỗi: Type, Boundary, FP, FN
│   │   └── predictions.py           # Lưu/đọc dự đoán JSONL và kiểm tra khớp độ dài
│   └── utils/
│       ├── config.py                # Đọc YAML & ghi đè tham số qua --override
│       ├── logger.py                # Ghi log console & Weights & Biases (tự fallback offline)
│       └── seed.py                  # Cố định random seed
├── tests/                           # Bộ unit test pytest (48 test cases)
├── outputs/
│   ├── metrics/                     # Thống kê EDA và kết quả JSON của 8 thực nghiệm
│   └── figures/                     # Biểu đồ EDA và phân tích phục vụ báo cáo
├── pytest.ini
└── requirements.txt
```

## Kết quả Thực nghiệm (Tập Test — Mức từ, Strict IOB2)

Toàn bộ kết quả đánh giá chi tiết được lưu trong `outputs/metrics/*.json` (chấm bằng `seqeval` với `mode="strict"`, `scheme=IOB2`).

### 1. Bảng tổng hợp 8 cấu hình (Chính & Ablation Study)

| Mô hình / Cấu hình | Best Epoch | Dev Micro-F1 | Test P | Test R | Test Micro-F1 | Test Macro-F1 | Vi phạm BIO |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bilstm_crf_nochar_word` (BiLSTM + CRF, tắt Char) | 26 | 91.73% | 92.56% | 89.62% | 91.07% | 85.89% | 24 |
| `bilstm_softmax_word` (BiLSTM + CharCNN + Softmax) | 30 | 92.60% | 92.62% | 90.87% | 91.74% | 88.14% | 293 |
| `bilstm_crf_word` (BiLSTM + CharCNN + CRF) | 27 | 93.43% | 92.71% | 92.35% | 92.53% | **89.38%** | 19 |
| `bilstm_charlstm_crf_word` (BiLSTM + CharBiLSTM + CRF) | 29 | 93.47% | 93.31% | 91.97% | **92.64%** | 89.25% | **16** |
| `phobert_freeze_last_softmax` (Đóng băng backbone) | 5 | 84.34% | 84.20% | 84.27% | 84.23% | 79.10% | 1097 |
| `phobert_full_last_softmax` (Full fine-tune + Softmax) | 3 | 95.76% | 94.92% | 95.30% | 95.11% | 93.28% | 112 |
| `phobert_full_concat4_softmax` (Nối 4 tầng ẩn cuối) | 5 | 95.79% | 95.20% | 95.41% | 95.30% | 93.38% | 132 |
| `phobert_full_last_crf` (Full fine-tune + LinearChainCRF) | 5 | **96.06%** | **95.35%** | **95.65%** | **95.50%** | **94.25%** | **64** |

### 2. F1 chi tiết theo từng loại trong 10 thực thể (Tập Test)

| Loại thực thể | Support | BiLSTM + CharCNN + CRF | BiLSTM + CharBiLSTM + CRF | PhoBERT + Softmax | PhoBERT + CRF |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `AGE` | 582 | 96.16% | 96.16% | 96.74% | **97.09%** |
| `DATE` | 1654 | 98.49% | 98.14% | 98.85% | **98.89%** |
| `GENDER` | 462 | 94.27% | 94.38% | **97.18%** | 96.90% |
| `JOB` | 173 | 63.50% | 61.54% | 78.05% | **83.97%** |
| `LOCATION` | 4441 | 92.33% | 92.58% | 95.10% | **95.63%** |
| `NAME` | 318 | 90.73% | 91.60% | 94.23% | **94.88%** |
| `ORGANIZATION` | 771 | 83.21% | 84.47% | 88.39% | **90.39%** |
| `PATIENT_ID` | 2005 | 97.43% | 97.69% | 97.98% | **98.20%** |
| `SYMPTOM_AND_DISEASE` | 1136 | 83.50% | 82.31% | **90.19%** | 88.88% |
| `TRANSPORTATION` | 193 | 94.21% | 93.65% | 96.10% | **97.67%** |

## Trọng số Mô hình đã Huấn luyện (Checkpoints)

Theo quy định không đưa file trọng số lớn (`.pt`) vào kho Git, toàn bộ checkpoint đã huấn luyện được lưu trên Google Drive:

* **BiLSTM-CRF (Full & Ablation):** [Google Drive — BiLSTM-CRF Checkpoints](https://drive.google.com/drive/folders/1QRDS6EDQQQjCwg0wYsoiLvsVbrlTqxgR?usp=sharing)
* **PhoBERT (Full & Ablation):** [Google Drive — PhoBERT Checkpoints](https://drive.google.com/drive/folders/1fekJLPSXtbDgyntZlHsgKfz49WCv6PMN?usp=sharing)

Để đánh giá lại hoặc chạy phân tích lỗi trên máy cá nhân, tải các thư mục checkpoint về đặt trong `checkpoints/` (thư mục này đã có trong `.gitignore`) rồi chạy lệnh ở mục **Đánh giá checkpoint đã lưu** bên dưới.

## Cài đặt Môi trường

```bash
pip install -r requirements.txt
```

## Tải Dữ liệu PhoNER_COVID19

```bash
python -m src.data.download --output-dir data
```

## Huấn luyện & Đánh giá

### 1. Huấn luyện BiLSTM-CRF đầy đủ (word-level)

```bash
python scripts/train.py --config configs/bilstm_crf.yaml
```

Nếu không đặt biến môi trường `WANDB_API_KEY`, hệ thống tự động chạy ở chế độ `WANDB_MODE=offline`.

### 2. Chạy Ablation Study cho BiLSTM-CRF bằng `--override`

```bash
# Tắt tầng CRF (BiLSTM + CharCNN + Softmax)
python scripts/train.py --config configs/bilstm_crf.yaml \
  --override experiment_name=bilstm_softmax_word model.use_crf=false training.checkpoint_dir=checkpoints/bilstm_softmax

# Tắt Character Embedding (BiLSTM + CRF)
python scripts/train.py --config configs/bilstm_crf.yaml \
  --override experiment_name=bilstm_crf_nochar_word model.use_char=false training.checkpoint_dir=checkpoints/bilstm_crf_nochar

# Thay CharCNN bằng CharBiLSTM
python scripts/train.py --config configs/bilstm_crf.yaml \
  --override experiment_name=bilstm_charlstm_crf_word model.char_encoder_type=bilstm training.checkpoint_dir=checkpoints/bilstm_charlstm_crf
```

### 3. Huấn luyện & Ablation cho PhoBERT

```bash
# PhoBERT chuẩn (full fine-tune + last_layer + Softmax)
python scripts/train.py --config configs/phobert.yaml

# Ablation 1: Đóng băng backbone
python scripts/train.py --config configs/phobert.yaml \
  --override experiment_name=phobert_freeze_last_softmax model.freeze_backbone=true training.checkpoint_dir=checkpoints/phobert/phobert_freeze_last_softmax

# Ablation 2: Nối 4 tầng ẩn cuối (concat_last_4)
python scripts/train.py --config configs/phobert.yaml \
  --override experiment_name=phobert_full_concat4_softmax model.feature_mode=concat_last_4 training.checkpoint_dir=checkpoints/phobert/phobert_full_concat4_softmax

# Ablation 3: PhoBERT + LinearChainCRF
python scripts/train.py --config configs/phobert.yaml \
  --override experiment_name=phobert_full_last_crf model.use_crf=true training.checkpoint_dir=checkpoints/phobert/phobert_full_last_crf
```

### 4. Đánh giá checkpoint đã lưu

```bash
python scripts/evaluate.py --config configs/bilstm_crf.yaml --checkpoint checkpoints/bilstm_crf/best_model.pt --split test
python scripts/evaluate.py --checkpoint checkpoints/phobert/phobert_full_last_crf/best_model.pt --split test
```

## Chạy Unit Test

```bash
pytest -v
```
