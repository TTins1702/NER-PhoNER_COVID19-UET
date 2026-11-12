# Thiết kế Hệ thống & Kế hoạch Thực hiện: Nhận dạng Thực thể Có tên (NER) Tiếng Việt trên PhoNER_COVID19

- **Ngày tạo:** 2026-11-12
- **Đề tài:** Đề tài 7 — Nhận dạng thực thể có tên (NER) tiếng Việt (`docs/DL - NER.md`)
- **Phạm vi hiện tại:** Mục tiêu Yêu cầu tối thiểu (Cài đặt `BiLSTM-CRF` có Character Embedding từ đầu, fine-tune `PhoBERT`, đánh giá F1 theo từng loại thực thể trên `PhoNER_COVID19`), thiết kế sẵn kiến trúc mở rộng cho Mục tiêu Giỏi về sau.
- **Môi trường huấn luyện:** Kaggle / Google Colab (GPU T4/P100) + Quản lý mã nguồn trên GitHub + Theo dõi thực nghiệm qua Weights & Biases (`wandb`).

---

## 1. Mục đích & Bối cảnh (Intent & Why)

Dự án này là Bài tập lớn môn Học sâu (Deep Learning) thực hiện bởi nhóm 3 sinh viên trong 6 tuần, hướng tới hai đầu ra cốt lõi:
1. **Hệ thống mã nguồn thực nghiệm chuẩn mực:** Tái lập được toàn bộ bảng kết quả, có lịch sử commit đều đặn của cả 3 thành viên, có log huấn luyện trực tuyến (`wandb`) và dễ dàng giải thích từng dòng code khi bảo vệ vấn đáp cá nhân (chiếm 30% tổng điểm).
2. **Báo cáo khoa học chuẩn ACL (tối đa 8 trang):** So sánh định lượng và định tính giữa mô hình gán nhãn chuỗi truyền thống tự cài đặt từ đầu (`BiLSTM-CRF` với Character Embedding) và mô hình ngôn ngữ tiền huấn luyện (`PhoBERT`) trên bộ dữ liệu thực tế `PhoNER_COVID19`.

Vì cả 3 thành viên đều huấn luyện mô hình trên môi trường đám mây tạm thời (Kaggle / Google Colab), hệ thống cần tách biệt **logic cốt lõi (`src/`)** khỏi **môi trường chạy (`notebooks/`)** để tránh xung đột Git, đảm bảo hai mô hình dùng chung một quy chuẩn tiền xử lý và một thước đo đánh giá F1.

---

## 2. Mục tiêu, Phi mục tiêu & Điều tối kỵ (Goals, Non-Goals & Anti-Goals)

### 2.1. Mục tiêu giai đoạn hiện tại (Goals — Yêu cầu tối thiểu)
1. **Pipeline dữ liệu thống nhất (`PhoNER_COVID19`):** Hỗ trợ tải tự động và đọc dữ liệu định dạng `.json` / `.conll`. Mặc định chạy trên mức từ (`word-level`, khớp với bộ tách từ của `PhoBERT`), đồng thời hỗ trợ tham số chuyển sang mức âm tiết (`syllable-level`) chỉ bằng một cấu hình.
2. **Tự cài đặt `BiLSTM-CRF` từ đầu (From Scratch bằng PyTorch):**
   - Cài đặt tầng biểu diễn ký tự (`CharEncoder` dùng 1D-CNN hoặc BiLSTM) kết hợp với `WordEmbedding`.
   - Cài đặt bộ mã hóa ngữ cảnh hai chiều `BiLSTM`.
   - **Tự cài đặt tầng `Linear-chain CRF` từ đầu:** Tự viết hàm `log_sum_exp` ổn định số học, thuật toán **Forward** có xử lý `mask` để tính hàm mất mát Negative Log-Likelihood $-\log P(y|x)$, và thuật toán **Viterbi** để giải mã chuỗi nhãn tối ưu.
3. **Fine-tune `PhoBERT` (`vinai/phobert-base-v2`):**
   - Xử lý bài toán lệch token (subword alignment) khi bộ BPE tokenizer của PhoBERT tách một từ tiếng Việt thành nhiều subword.
   - Hỗ trợ kiến trúc `PhoBERT + Linear Softmax` (chuẩn) và tùy chọn `PhoBERT + CRF` (tái sử dụng tầng CRF tự cài) để làm thực nghiệm đối chiếu.
4. **Đánh giá và Phân tích chuyên sâu:**
   - Đánh giá theo chuẩn CoNLL entity-level strict match qua thư viện `seqeval`: báo cáo Precision, Recall, Micro-F1, Macro-F1 và **F1 cho từng loại trong 10 loại thực thể**.
   - Thực hiện **Ablation Study** (bật/tắt CRF, bật/tắt Character Embedding, chiến lược trích xuất đặc trưng PhoBERT) và **Error Analysis** (phân loại lỗi ranh giới, lỗi nhầm nhãn, lỗi bỏ sót/báo nhầm).

### 2.2. Phi mục tiêu ở giai đoạn này (Non-Goals — Để dành mở rộng sau)
- Chưa cài đặt nhánh Few-shot NER bằng LLM (nhưng thiết kế sẵn giao diện đầu ra dự đoán để sau này thêm module LLM vẫn chấm điểm chung được).
- Chưa chạy toàn bộ lưới thí nghiệm so sánh `word-level` vs `syllable-level` ở giai đoạn tối thiểu.
- Không sử dụng bộ dữ liệu VLSP 2016/2018 và không dùng mô hình XLM-R.

### 2.3. Điều tối kỵ (Anti-Goals — Nếu vi phạm coi như thất bại)
- **Không dùng thư viện CRF đóng gói sẵn** (như `pytorch-crf`, `TorchCRF`) làm mô hình chính cho `BiLSTM-CRF`. Mọi phép tính ma trận chuyển trạng thái (transition matrix), Forward score, Viterbi decoding phải được viết tường minh trong `src/models/crf.py` để phục vụ chấm điểm kỹ thuật và vấn đáp. (Chỉ được phép dùng thư viện ngoài trong unit test nếu muốn kiểm tra độ chính xác của code tự viết).
- **Không chấm điểm F1 ở mức subword hoặc token rời rạc (`sklearn`):** Phải quy đổi dự đoán về đúng mức từ gốc (`word-level`) và đánh giá mức thực thể (`seqeval` strict IOB2).
- **Không đưa dữ liệu (`data/`) hoặc trọng số mô hình (`.pt`, `.bin`, `.safetensors`) vào Git.**

---

## 3. Các ràng buộc (Constraints)

- **Thời gian:** 6 tuần theo lịch môn học (Tuần 1: Proposal 2 trang; Tuần 3: Milestone Report 4 trang; Tuần 6: Báo cáo cuối 8 trang chuẩn ACL + Code + Slide + Demo).
- **Hạ tầng:** Chạy trên Kaggle (GPU T4x2 / P100) và Google Colab (GPU T4). Vì phiên chạy sẽ xóa ổ cứng cục bộ khi ngắt kết nối, mọi log huấn luyện phải đẩy lên `wandb` (hoặc lưu về Google Drive / Kaggle Dataset output) và kết quả bảng biểu phải xuất thành file `.json`/`.csv` nhẹ để commit về GitHub.
- **Đánh giá học phần:** 70% điểm nhóm (25% kỹ thuật, 25% báo cáo & phân tích, 10% quá trình Git/log, 10% thuyết trình/demo) + 30% vấn đáp cá nhân (dưới 4/10 vấn đáp thì tổng điểm bị khống chế $\le 5/10$).

---

## 4. Lựa chọn Kiến trúc (Approaches)

Nhóm đã xem xét 3 phương án tổ chức dự án:
1. **Hướng 1 — Modular Python (`src/`) + Thin Runner Notebooks (`notebooks/`) [ĐÃ CHỌN]:**
   - Đóng gói toàn bộ logic vào `src/` theo từng file `.py` độc lập tương ứng với 3 thành viên. Các file `.ipynb` trên Kaggle/Colab chỉ chứa vài cell để clone repo, truyền cấu hình YAML và gọi hàm huấn luyện/đánh giá.
   - *Lý do chọn:* Tránh hoàn toàn xung đột Git trên file `.ipynb`, đảm bảo tính nhất quán khi tiền xử lý/đánh giá, dễ tái sử dụng khi làm Ablation Study, và giúp giảng viên dễ kiểm tra từng hàm khi vấn đáp.
2. **Hướng 2 — Self-Contained Notebooks + Shared Utils [LOẠI]:**
   - Viết toàn bộ mô hình và vòng lặp huấn luyện vào trong từng notebook. Loại vì gây khó khăn khi quản lý lịch sử commit, phải copy-paste code khi chạy ablation và khó bảo trì.
3. **Hướng 3 — Monolithic Framework (PyTorch Lightning / Hydra) [LOẠI]:**
   - Loại vì làm phức tạp hóa cấu trúc và che giấu vòng lặp huấn luyện/loss bên dưới framework, gây bất lợi khi giảng viên hỏi vấn đáp sâu về cơ chế cập nhật trọng số và masking.

---

## 5. Thiết kế Chi tiết (Detailed Design)

### 5.1. Phân công nhiệm vụ 3 thành viên (Phương án A)

| Thành viên | Vai trò chính | Các phân hệ phụ trách (`src/` & `notebooks/`) |
| :--- | :--- | :--- |
| **Đạt** | **Data Pipeline, Evaluation, Error Analysis & Demo** | `src/data/`, `src/evaluation/`, `notebooks/01_eda_data_stats.ipynb`, `notebooks/04_error_analysis.ipynb`, `app/` (Demo cuối kỳ) |
| **Tín** | **BiLSTM-CRF From Scratch & Ablation** | `src/models/char_encoder.py`, `src/models/crf.py`, `src/models/bilstm_crf.py`, `configs/bilstm_crf.yaml`, `notebooks/02_train_bilstm_crf.ipynb` |
| **Đức Anh** | **PhoBERT Fine-tuning & Ablation** | `src/data/phobert_dataset.py` (Subword alignment), `src/models/phobert_ner.py`, `configs/phobert.yaml`, `notebooks/03_train_phobert.ipynb` |

*(Phần `src/training/trainer.py` và `src/utils/` được thiết kế giao diện chung từ tuần 1 để Tín và Đức Anh cùng sử dụng).*

---

### 5.2. Cấu trúc thư mục chuẩn của Dự án

```text
NER-UET/
├── configs/
│   ├── bilstm_crf.yaml              # Siêu tham số cho BiLSTM-CRF (và các cờ ablation)
│   └── phobert.yaml                 # Siêu tham số cho PhoBERT (lr, warmup, freeze, use_crf)
├── data/                            # Dữ liệu PhoNER_COVID19 (đã có trong .gitignore)
│   ├── word/                        # train_word.json, dev_word.json, test_word.json (.conll)
│   └── syllable/                    # train_syllable.json, dev_syllable.json, test_syllable.json
├── docs/
│   ├── DL - NER.md                  # Đề bài và quy định môn học
│   └── superpowers/specs/           # Tài liệu thiết kế và kế hoạch chi tiết
├── notebooks/
│   ├── 01_eda_data_stats.ipynb      # [Đạt] Tải dữ liệu, thống kê độ dài câu, phân bố 10 nhãn
│   ├── 02_train_bilstm_crf.ipynb    # [Tín] Trạm chạy BiLSTM-CRF + Ablation trên Kaggle/Colab
│   ├── 03_train_phobert.ipynb       # [Đức Anh] Trạm chạy PhoBERT + Ablation trên Kaggle/Colab
│   └── 04_error_analysis.ipynb      # [Đạt] So sánh 2 mô hình, ma trận nhầm lẫn, phân tích lỗi
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── seed.py                  # Cố định random seed (torch, numpy, random) để tái lập kết quả
│   │   ├── config.py                # Đọc file YAML và ghi đè tham số từ CLI
│   │   └── logger.py                # Tích hợp Weights & Biases (wandb) + console logging
│   ├── data/
│   │   ├── download.py              # Script tự động tải PhoNER_COVID19 (chuyển từ notebook hiện có)
│   │   ├── labels.py                # Quản lý bộ 21 nhãn BIO cố định (label2id, id2label)
│   │   ├── vocab.py                 # WordVocab (<pad>, <unk>) và CharVocab cho BiLSTM-CRF
│   │   ├── bilstm_dataset.py        # Dataset & collate_fn (padding từ + ký tự) cho BiLSTM-CRF
│   │   └── phobert_dataset.py       # Dataset & subword alignment (first-subword labeling) cho PhoBERT
│   ├── models/
│   │   ├── char_encoder.py          # [Tín] Character-level CNN / BiLSTM tạo vector biểu diễn từ
│   │   ├── crf.py                   # [Tín] Tầng Linear-chain CRF cài đặt từ đầu (Forward + Viterbi)
│   │   ├── bilstm_crf.py            # [Tín] Kiến trúc tích hợp WordEmb + CharEncoder + BiLSTM + CRF
│   │   └── phobert_ner.py           # [Đức Anh] PhoBERT-base-v2 + Linear Classifier (hoặc CRF head)
│   ├── training/
│   │   └── trainer.py               # Vòng lặp train/eval, gradient clipping, early stopping, lưu checkpoint
│   └── evaluation/
│       ├── metrics.py               # [Đạt] Tính Precision/Recall/F1 từng thực thể bằng seqeval
│       └── error_analysis.py        # [Đạt] Phân loại lỗi (Boundary, Type, FP, FN) và xuất bảng mẫu câu sai
├── scripts/
│   ├── train.py                     # CLI chạy huấn luyện: python scripts/train.py --config ...
│   └── evaluate.py                  # CLI đánh giá checkpoint trên tập dev/test và xuất file JSON
├── outputs/
│   ├── metrics/                     # Chứa các file kết quả nhẹ (.json, .csv) được commit lên Git
│   └── figures/                     # Chứa biểu đồ phân bố, đường cong học, confusion matrix cho báo cáo ACL
├── tests/                           # Unit test kiểm tra riêng tầng CRF, subword alignment và metrics
├── .gitignore                       # Bỏ qua data/, checkpoints/, wandb/, __pycache__/
└── requirements.txt                 # Danh sách thư viện (torch, transformers, seqeval, pyyaml, wandb...)
```

---

### 5.3. Đặc tả Kỹ thuật cho từng Phân hệ

#### A. Quản lý Nhãn & Tiền xử lý Dữ liệu (`src/data/` — Đạt & Đức Anh)
- **Định dạng đầu vào:** Đọc từ file JSONL (`{split}_{level}.json`), mỗi dòng là một dict `{"words": [...], "tags": [...]}`.
- **Hệ thống 21 nhãn BIO cố định (`labels.py`):**
  - Nhãn `O` (id = `0`) và 10 loại thực thể (`AGE`, `DATE`, `GENDER`, `JOB`, `LOCATION`, `NAME`, `ORGANIZATION`, `PATIENT_ID`, `SYMPTOM_AND_DISEASE`, `TRANSPORTATION`), mỗi loại có `B-<ENTITY>` và `I-<ENTITY>` sắp xếp theo thứ tự bảng chữ cái cố định để mọi thành viên và mọi lần chạy có cùng `label2id`.
- **Cho `BiLSTM-CRF` (`bilstm_dataset.py` - Đạt):**
  - Xây dựng `WordVocab` (chỉ từ tập `train`, hỗ trợ ngưỡng tần suất tối thiểu `min_freq` để học embedding cho `<unk>`) và `CharVocab` (toàn bộ ký tự tiếng Việt, chữ số, dấu câu).
  - Mỗi batch trả về: `word_ids [B, L]`, `char_ids [B, L, W]` (với `W` là độ dài ký tự tối đa của từ trong batch), `mask [B, L]` (kiểu `bool`, `True` tại vị trí từ thật, `False` tại `<pad>`), và `tags [B, L]`.
- **Cho `PhoBERT` (`phobert_dataset.py` - Đức Anh):**
  - Trong `data/word/`, mỗi phần tử của `"words"` đã là một từ ghép sẵn dấu gạch dưới (ví dụ: `"bệnh_nhân"`, `"Hà_Nội"`).
  - Khi đi qua BPE Tokenizer của `vinai/phobert-base-v2`, một từ có thể bị tách thành nhiều subword (ví dụ: `["bệnh_nhân"]` hoặc `["Vĩnh_", " Phúc"]`).
  - **Quy tắc gán nhãn (First-Subword Alignment):**
    - Subword đầu tiên của mỗi từ nhận nhãn BIO gốc của từ đó và được đánh dấu `word_mask = True`.
    - Các subword nối tiếp của cùng từ đó (nếu bị tách) và các token đặc biệt (`<s>`, `</s>`, `<pad>`) nhận nhãn `ignore_index = -100` (hoặc bị loại qua `word_mask = False` nếu dùng tầng CRF).
    - Nếu câu dài vượt `max_seq_length` (256 đối với PhoBERT), cắt câu an toàn và ghi nhận thống kê (trong PhoNER_COVID19 hầu hết các câu đều ngắn hơn 256 subwords).
  - **Quy tắc giải mã khi đánh giá:** Chỉ trích xuất nhãn dự đoán tại các vị trí subword đầu tiên của mỗi từ để khôi phục chuỗi nhãn dự đoán có độ dài **bằng đúng số từ gốc `len(words)`**, sau đó mới đưa vào `seqeval` chấm điểm.

#### B. Mô hình 1: `BiLSTM-CRF` tự cài từ đầu (`src/models/` — Tín)
1. **`CharEncoder` (`char_encoder.py`):**
   - Nhập `char_ids [B, L, W]`, nhúng qua `nn.Embedding(num_chars, char_emb_dim)`, sau đó đi qua `Conv1d` + `ReLU` + `MaxPool1d` theo chiều độ dài từ `W` (hoặc `CharBiLSTM`), thu được vector biểu diễn ký tự kích thước `[B, L, char_out_dim]`.
2. **`BiLSTM` Encoder (`bilstm_crf.py`):**
   - Nối (`concat`) vector `WordEmbedding [B, L, word_emb_dim]` và `CharEncoder [B, L, char_out_dim]` (nếu `use_char=True`), qua `Dropout`, rồi đưa vào `nn.LSTM(..., bidirectional=True)` có dùng `pack_padded_sequence` để không tính toán trên token `<pad>`.
   - Chiếu đầu ra của BiLSTM `[B, L, 2 * hidden_dim]` qua tầng `nn.Linear` để lấy ma trận điểm phát xạ `emissions [B, L, num_tags]`.
3. **Tầng `LinearChainCRF` tự cài đặt (`crf.py`):**
   - Tham số học: Ma trận chuyển trạng thái `transitions [num_tags, num_tags]` cùng 2 vector `start_transitions [num_tags]` và `end_transitions [num_tags]`.
   - **Hàm tính Loss (`forward`):**
     - Tính tổng điểm của chuỗi nhãn thật $s(x, y) = \sum_t (\text{emission}_{t, y_t} + \text{transition}_{y_{t-1}, y_t})$ có nhân với `mask`.
     - Tính hàm phân hoạch $\log Z(x)$ bằng thuật toán quy hoạch động **Forward** sử dụng phép toán `log_sum_exp` ổn định số học qua từng bước thời gian $t = 1 \dots L$ (tại các vị trí `mask[:, t] == False`, giữ nguyên trạng thái `alpha` trước đó).
     - Trả về $\text{Loss} = \text{mean}(\log Z(x) - s(x, y))$.
   - **Hàm giải mã (`decode` - Thuật toán Viterbi):**
     - Quy hoạch động tìm đường đi có tổng điểm cao nhất bằng phép `max` và lưu con trỏ truy vết `backpointers`, trả về danh sách chuỗi nhãn dự đoán cho từng câu trong batch.
   - **Hỗ trợ Ablation:** Khi cấu hình `use_crf: False`, mô hình dùng thẳng `nn.CrossEntropyLoss` trên `emissions` và giải mã bằng `argmax` để làm đối chứng thực nghiệm cho báo cáo.

#### C. Mô hình 2: `PhoBERT` Fine-tuning (`src/models/phobert_ner.py` — Đức Anh)
- Tải backbone `AutoModel.from_pretrained("vinai/phobert-base-v2")`.
- Hỗ trợ 2 chế độ trích xuất đặc trưng (cấu hình trong `phobert.yaml`):
  - `last_layer`: Lấy `last_hidden_state`.
  - `concat_last_4`: Nối 4 tầng ẩn cuối cùng của Transformer (thường cải thiện F1 cho bài toán NER).
- Hỗ trợ 2 chế độ tầng dự đoán (Head):
  - `softmax` (mặc định): `Dropout` + `Linear(hidden_size, num_tags)` + `CrossEntropyLoss(ignore_index=-100)`.
  - `crf` (mở rộng/ablation): Rút trích vector biểu diễn tại vị trí subword đầu tiên của mỗi từ, chiếu qua `Linear` rồi đưa vào chính tầng `LinearChainCRF` của dự án.
- Tối ưu hóa: Dùng `AdamW` với tốc độ học phân tầng (Differential Learning Rate: backbone `lr = 2e-5` đến `3e-5`, tầng classifier/CRF `lr = 1e-3`) kèm `linear warmup` (10% tổng số bước).

#### D. Đánh giá, Phân tích lỗi & Quy trình Kaggle/Colab (`src/evaluation/` & `notebooks/`)
- **Bộ chấm điểm chuẩn (`src/evaluation/metrics.py`):**
  - Đầu vào: `y_true: list[list[str]]`, `y_pred: list[list[str]]` (các chuỗi nhãn BIO ở mức từ).
  - Sử dụng `seqeval.metrics.classification_report(..., mode='strict', scheme=IOB2, output_dict=True)` để trích xuất bảng điểm của cả 10 thực thể cùng `micro avg` và `macro avg`.
  - Sửa lỗi nhãn BIO không hợp lệ (nếu dùng Softmax head không có CRF sinh ra `I-LOC` đứng đầu mà không có `B-LOC`) hoặc thống kê số lượng vi phạm ràng buộc BIO để chứng minh hiệu quả của tầng CRF trong mục Analysis của báo cáo!
- **Quy trình chạy trên Kaggle / Google Colab:**
  - Mỗi notebook (`02_train_bilstm_crf.ipynb`, `03_train_phobert.ipynb`) bắt đầu bằng 1 cell thiết lập tự động:
    1. Kiểm tra nếu chưa có repo thì `git clone`, nếu có rồi thì `git pull`.
    2. Cài đặt thư viện từ `requirements.txt`.
    3. Gọi `python -m src.data.download` để tải `PhoNER_COVID19` về bộ nhớ máy ảo trong 2 giây.
    4. Đăng nhập `wandb` (lấy key từ Kaggle Secrets hoặc Colab Secrets; nếu không cấu hình key thì tự chuyển sang `WANDB_MODE=offline`).
    5. Chạy lệnh huấn luyện và tự động xuất kết quả đánh giá trên tập `dev`/`test` ra `outputs/metrics/<experiment_name>.json`.

---

## 6. Kế hoạch Thực hiện 6 Tuần (6-Week Roadmap)

Kế hoạch này bám sát các cột mốc bắt buộc trong [`docs/DL - NER.md`](file:///d:/CN12_2024_2028/Nam_3/DL/Assignments/NER-UET/docs/DL%20-%20NER.md#L21-L28):

| Tuần | Mục tiêu & Sản phẩm nộp | Công việc Đạt (Data, Eval & Analysis) | Công việc Tín (BiLSTM-CRF) | Công việc Đức Anh (PhoBERT) |
| :---: | :--- | :--- | :--- | :--- |
| **Tuần 1** | **Nộp Proposal (tối đa 2 trang)**<br>Tạo GitHub repo, mời giảng viên, chia sẻ link W&B | - Hoàn thiện `src/data/download.py`, `labels.py`.<br>- Chạy `01_eda_data_stats.ipynb` lấy bảng thống kê dữ liệu đưa vào Proposal.<br>- Viết mục Bài toán, Dữ liệu & Đánh giá trong Proposal. | - Đọc bài báo gốc BiLSTM-CRF (Lample et al., 2016; Ma & Hovy, 2016).<br>- Khởi tạo khung `src/models/crf.py`.<br>- Viết mục Phương pháp BiLSTM-CRF trong Proposal. | - Đọc bài báo PhoBERT & PhoNER_COVID19 (Nguyen & Nguyen, 2020; Truong et al., 2021).<br>- Thử nghiệm BPE tokenizer trên câu mẫu.<br>- Viết mục Related Work & PhoBERT trong Proposal. |
| **Tuần 2** | **Cài đặt & Chạy Baseline** | - Hoàn thiện `vocab.py`, `bilstm_dataset.py` và `src/evaluation/metrics.py` (`seqeval`).<br>- Viết unit test kiểm tra bộ đọc dữ liệu và hàm tính F1. | - Cài đặt xong `BiLSTM` cơ bản (chưa có Char/CRF) + hàm huấn luyện trong `trainer.py`.<br>- Chạy baseline đầu tiên trên Colab/Kaggle và đẩy log lên W&B. | - Hoàn thiện `phobert_dataset.py` (First-subword alignment) + unit test kiểm tra khớp độ dài nhãn.<br>- Chạy thử 1 epoch PhoBERT baseline. |
| **Tuần 3** | **Nộp Milestone Report (tối đa 4 trang)**<br>Hoàn thiện phương pháp chính | - Tổng hợp kết quả baseline từ `outputs/metrics/`.<br>- Kiểm tra số lượng lỗi vi phạm BIO giữa các mô hình.<br>- Phụ trách tổng hợp Milestone Report. | - Hoàn thiện `char_encoder.py` và tầng `LinearChainCRF` (`crf.py`) từ đầu.<br>- Huấn luyện hoàn chỉnh `BiLSTM-CRF` trên `data/word/` và ghi nhận F1 tập `dev`/`test`. | - Hoàn thiện `phobert_ner.py` với Differential LR & Warmup.<br>- Huấn luyện hoàn chỉnh `PhoBERT` chuẩn trên `data/word/` và xuất kết quả cho Milestone Report. |
| **Tuần 4** | **Thí nghiệm, Ablation & Tinh chỉnh siêu tham số** | - Xây dựng `src/evaluation/error_analysis.py` phân loại tự động 4 nhóm lỗi NER.<br>- Tạo script tự động xuất bảng LaTeX chuẩn ACL từ file `.json`. | - Chạy **Ablation Study cho BiLSTM-CRF**:<br>  1) BiLSTM + Softmax (không CRF)<br>  2) BiLSTM + CRF (không CharEmb)<br>  3) BiLSTM + CharCNN + CRF (đầy đủ)<br>- Tinh chỉnh `lr`, `hidden_dim`, `dropout`. | - Chạy **Ablation Study cho PhoBERT**:<br>  1) Freeze backbone vs. Full fine-tune<br>  2) Last hidden state vs. Concat 4 layers<br>  3) PhoBERT + Softmax vs. PhoBERT + CRF. |
| **Tuần 5** | **Phân tích lỗi, Trực quan hóa & Bản nháp Báo cáo** *(Tùy chọn mở rộng nếu kịp)* | - Chạy `04_error_analysis.ipynb`: Vẽ Confusion Matrix, biểu đồ F1 theo từng thực thể, trích xuất các ví dụ câu dự đoán đúng/sai điển hình.<br>- Xây dựng giao diện Demo nhẹ (Gradio). | - Phân tích sâu ma trận chuyển trạng thái (`transitions`) học được của tầng CRF (vẽ heatmap ma trận chuyển nhãn BIO).<br>- Viết mục Method & Experiments của BiLSTM-CRF. | - Phân tích các trường hợp PhoBERT vượt trội BiLSTM-CRF (thực thể dài, từ hiếm/OOV như `SYMPTOM_AND_DISEASE`, `LOCATION`).<br>- Viết mục Method & Analysis của PhoBERT. |
| **Tuần 6** | **Nộp Báo cáo cuối (8 trang ACL), Code, Slide & Demo** | - Hoàn thiện README.md, kiểm tra khả năng tái tạo kết quả bằng 1 lệnh.<br>- Chuẩn bị Demo và ôn tập câu hỏi vấn đáp. | - Rà soát code `crf.py`, `bilstm_crf.py`, chuẩn bị câu trả lời vấn đáp về masking và độ phức tạp thuật toán Forward/Viterbi. | - Rà soát code `phobert_ner.py`, `phobert_dataset.py`, chuẩn bị câu trả lời vấn đáp về BPE subword alignment và Transformer. |

---

## 7. Những phần giao quyền cho người cài đặt (Left to the Builder)

Các chi tiết dưới đây không làm thay đổi kiến trúc hệ thống và người lập trình có thể chủ động quyết định trong quá trình cài đặt:
- Các giá trị siêu tham số khởi tạo cụ thể trong file YAML (ví dụ: `char_emb_dim=30` hay `50`, `word_emb_dim=100` hay `300`, `bilstm_hidden_dim=256`, `batch_size=32`).
- Định dạng màu sắc, kích thước font của các biểu đồ xuất ra trong `outputs/figures/`.
- Giao diện cụ thể của bản Demo cuối kỳ (dùng Gradio chạy ngay trên notebook Colab hoặc ứng dụng Streamlit độc lập).

