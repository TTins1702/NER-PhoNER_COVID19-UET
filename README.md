# NER-PhoNER_COVID19-UET

Hệ thống nhận dạng thực thể có tên (Named Entity Recognition - NER) tiếng Việt trên bộ dữ liệu **PhoNER_COVID19** (10 loại thực thể, 21 nhãn BIO), phục vụ đồ án môn Học sâu (Deep Learning).

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
│   │   ├── download.py              # Tải PhoNER_COVID19 và đọc JSONL theo split/level
│   │   ├── labels.py                # Thứ tự cố định 21 nhãn BIO và IGNORE_INDEX = -100
│   │   ├── vocab.py                 # WordVocab (<pad>, <unk>) và CharVocab
│   │   ├── bilstm_dataset.py        # BiLSTMDataset và dynamic 3D char padding
│   │   └── phobert_dataset.py       # PhoBERTDataset và first-subword alignment
│   ├── models/
│   │   ├── char_encoder.py          # Bộ mã hóa ký tự (1D-CNN hoặc BiLSTM)
│   │   ├── crf.py                   # Tầng LinearChainCRF cài đặt từ đầu (Forward + Viterbi)
│   │   ├── bilstm_crf.py            # Mô hình BiLSTM-CRF kèm cờ use_char / use_crf
│   │   └── phobert_ner.py           # Mô hình PhoBERT-base-v2 kèm Softmax / CRF head
│   ├── training/
│   │   └── trainer.py               # Bộ huấn luyện chung, gradient clipping, early stopping
│   ├── evaluation/
│   │   ├── metrics.py               # Chấm điểm strict IOB2 qua seqeval & đếm vi phạm BIO
│   │   └── error_analysis.py        # Phân loại 4 nhóm lỗi: Type, Boundary, FP, FN
│   └── utils/
│       ├── config.py                # Đọc YAML & ghi đè tham số qua --override
│       ├── logger.py                # Ghi log console & Weights & Biases (tự fallback offline)
│       └── seed.py                  # Cố định random seed
├── tests/                           # Bộ unit test pytest
├── outputs/
│   ├── metrics/                     # File kết quả JSON nhẹ
│   └── figures/                     # Biểu đồ phân tích phục vụ báo cáo
├── pytest.ini
└── requirements.txt
```

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

### 2. Chạy Ablation Study bằng `--override`

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
python scripts/evaluate.py --checkpoint checkpoints/phobert/phobert_full_last_softmax/best_model.pt --split test
```

## Chạy Unit Test

```bash
pytest -v
```
