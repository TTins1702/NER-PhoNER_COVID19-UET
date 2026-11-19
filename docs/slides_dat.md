---
marp: true
theme: default
paginate: true
size: 16:9
---

# Nhận dạng Thực thể Có tên (NER) Tiếng Việt trên `PhoNER_COVID19`
### Đối sánh Thực nghiệm giữa `BiLSTM-CRF` Tự Cài đặt và `PhoBERT`

**Nhóm thực hiện:** Đạt — Tín — Đức Anh
**Trường Đại học Công nghệ — ĐHQGHN (UET-VNU)**
*(Phần trình bày của Đạt: Bài toán, Dữ liệu & EDA, Hợp đồng Đánh giá, Phân tích lỗi & Demo)*

---

## Slide 1 — Phát biểu Bài toán & Thách thức (1 phút)

- **Đầu vào:** Câu tiếng Việt gồm $N$ token $\mathbf{x} = (x_1, \dots, x_N)$
  - *Mức từ (`word-level`, mặc định):* `["Bệnh_viện", "Bạch_Mai", "rất", "lớn"]`
  - *Mức âm tiết (`syllable-level`):* `["Bệnh", "viện", "Bạch", "Mai", "rất", "lớn"]`
- **Đầu ra:** Chuỗi $N$ nhãn theo lược đồ **IOB2** ($|\mathcal{Y}| = 21$ nhãn cố định: `O` + `B-/I-` của 10 loại) $\rightarrow$ trích xuất tập thực thể $\{(c_k, s_k, e_k)\}$.
- **3 thách thức đặc thù trên miền dịch tễ tiếng Việt:**
  1. **Ranh giới từ ghép:** Khoảng trắng tách âm tiết, không phải tách từ.
  2. **Tỉ lệ OOV cao ở thực thể mới:** Mã bệnh nhân mới (`BN1347`), viết tắt tên (`N.H.N.`), số hiệu chuyến bay (`VN0054`).
  3. **Nhập nhằng ngữ cảnh & thực thể dài:** `"Bệnh_viện Đà_Nẵng"` vừa có thể là `LOCATION` (nơi đến) vừa có thể là `ORGANIZATION` (chủ thể thông báo).

---

## Slide 2 — Bộ dữ liệu `PhoNER_COVID19` & Thống kê EDA (1.5 phút)

| Tập dữ liệu | Số câu | Số từ (`word`) | Số âm tiết (`syllable`) | Dài TB (từ) | Số thực thể | Tỉ lệ nhãn `O` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | $5.027$ | $132.511$ | $167.541$ | $26{,}36$ | $15.767$ | $79{,}05\%$ |
| `dev` | $2.000$ | $56.283$ | $71.325$ | $28{,}14$ | $7.478$ | $74{,}92\%$ |
| `test` | $3.000$ | $85.678$ | $108.354$ | $28{,}56$ | $11.735$ | $74{,}17\%$ |
| **Tổng cộng** | **$10.027$** | **$274.472$** | **$347.220$** | **$27{,}37$** | **$34.980$** | **$76{,}68\%$** |

- **10 loại thực thể:** `LOCATION` ($35{,}95\%$), `PATIENT_ID` ($18{,}64\%$), `DATE` ($15{,}17\%$), `SYMPTOM_AND_DISEASE` ($9{,}55\%$), `ORGANIZATION` ($7{,}03\%$), `AGE` ($4{,}65\%$), `GENDER` ($3{,}66\%$), `NAME` ($2{,}44\%$), `JOB` ($1{,}46\%$), `TRANSPORTATION` ($1{,}45\%$).
- **Phát hiện từ EDA (`01_eda_data_stats.ipynb`):**
  - Mất cân bằng lớp rất lớn (`LOCATION` gấp $>24$ lần `JOB` và `TRANSPORTATION`).
  - OOV chung trên `test` là $2{,}96\%$, nhưng trong token thực thể `NAME` lên tới **$38{,}4\%$**, `TRANSPORTATION` **$34{,}7\%$**, `PATIENT_ID` **$14{,}7\%$** $\rightarrow$ Lý do bắt buộc phải có **Character Embedding** / **Subword BPE**.

---

## Slide 3 — Pipeline Dữ liệu & Hợp đồng Giao diện (1 phút)

- **Từ điển (`src/data/vocab.py`):**
  - `WordVocab` **chỉ xây từ tập `train`** ($5.169$ từ với `min_freq=1`, $3.274$ từ với `min_freq=2`) để chống rò rỉ dữ liệu; từ hiếm chuyển thành `<unk>` giúp học được vector biểu diễn cho từ lạ khi test.
  - `CharVocab` phủ toàn bộ chữ cái tiếng Việt có dấu (hoa/thường, chuẩn hóa Unicode `NFC`), chữ số, dấu câu.
- **Đóng gói Batch động (`src/data/bilstm_dataset.py`):**
  - `word_ids [B, L]`, `char_ids [B, L, W]`, `mask [B, L]` (`bool`), `tags [B, L]`, `lengths [B]`.
  - Tại sao pad `tags` bằng `0` (`O`) thay vì `-100`? Vì tầng CRF tự cài cần `torch.gather` theo chỉ số nhãn hợp lệ $[0..20]$, sau đó dùng `mask` để triệt tiêu vị trí pad khỏi hàm loss.
- *(Chuyển lời cho Tín trình bày `BiLSTM-CRF` và Đức Anh trình bày `PhoBERT`)*.

---

## Slide 4 — Quy chuẩn Đánh giá Nghiêm ngặt (`src/evaluation/metrics.py`) (1 phút)

- **Quy về mức từ gốc (`word-level`):**
  - Mọi dự đoán $\hat{\mathbf{y}}$ của cả `BiLSTM-CRF` và `PhoBERT` (sau bước rút trích *first-subword*) phải có độ dài **bằng đúng `len(words)`** của câu gốc.
- **Chấm điểm mức thực thể bằng `seqeval` (`mode="strict", scheme=IOB2`):**
  - Một thực thể $(c, s, e)$ chỉ được tính là đúng (**TP**) khi đúng **cả ranh giới đầu–cuối lẫn loại thực thể**.
  - Khác với chế độ mặc định (lỏng), chế độ `strict IOB2` **không** chấp nhận thực thể bắt đầu bằng `I-X` mồ côi $\rightarrow$ phản ánh trung thực lợi ích của tầng CRF.
- **Bảng kết quả tổng thể trên tập `test_word`:**
  - `BiLSTM + CharCNN + CRF`: Micro-F1 = `[CHỜ SỐ LIỆU: Micro-F1 BiLSTM-CRF]`%, Macro-F1 = `[CHỜ SỐ LIỆU: Macro-F1 BiLSTM-CRF]`%, Vi phạm BIO = `[CHỜ SỐ LIỆU: BIO BiLSTM-CRF]`.
  - `PhoBERT-base-v2`: Micro-F1 = `[CHỜ SỐ LIỆU: Micro-F1 PhoBERT]`%, Macro-F1 = `[CHỜ SỐ LIỆU: Macro-F1 PhoBERT]`%, Vi phạm BIO = `[CHỜ SỐ LIỆU: BIO PhoBERT]`.

---

## Slide 5 — Phân tích Lỗi 4 Nhóm & Vi phạm BIO (`04_error_analysis.ipynb`) (1.5 phút)

- **Phân rã tự động mọi thực thể dự đoán sai vào 4 nhóm loại trừ lẫn nhau:**
  1. **Lỗi ranh giới (`Boundary`):** Đúng loại nhưng thừa/thiếu từ ở biên (tập trung ở `ORGANIZATION` dài TB $3{,}24$ từ và `LOCATION` dài tới $13$ từ).
  2. **Lỗi nhầm loại (`Type`):** Giao với thực thể thật nhưng sai nhãn (hay gặp nhất giữa `LOCATION` $\leftrightarrow$ `ORGANIZATION` trên các bệnh viện/địa điểm hành chính).
  3. **Báo nhầm (`False Positive - FP`):** Gán nhãn thực thể lên vùng `O`.
  4. **Bỏ sót (`False Negative - FN`):** Dự đoán toàn `O` trên thực thể thật (hay gặp ở thực thể chứa từ OOV hoặc câu phức).
- **Định lượng tác dụng của tầng CRF:**
  - Mô hình dùng Softmax độc lập sinh ra các chuỗi vi phạm quy tắc BIO (`O -> I-LOC`, `B-ORG -> I-LOC`); tầng `Linear-chain CRF` triệt tiêu lỗi này nhờ ma trận chuyển trạng thái học được.

---

## Slide 6 — Trực quan hóa & Demo Hệ thống (`app/demo.py`) (1.5 phút)

- **Biểu đồ phân tích (đọc rõ ở thang xám trong báo cáo ACL):**
  - `outputs/figures/eval_per_entity_f1.pdf`: So sánh F1 trên từng loại trong 10 loại thực thể.
  - `outputs/figures/eval_confusion_matrix_*.pdf`: Ma trận nhầm lẫn chuẩn hóa theo hàng (%).
  - `outputs/figures/eval_error_breakdown.pdf` & `eval_recall_by_length.pdf`: Phân rã 4 nhóm lỗi và độ sụt giảm Recall khi thực thể dài $\ge 4$ từ.
- **Demo trực tiếp bằng Gradio (`python -m app.demo`):**
  - Nhập câu tiếng Việt tự nhiên $\rightarrow$ tự động tách từ bằng `Underthesea` (`Bệnh_viện`, `Hà_Nội`) $\rightarrow$ nạp checkpoint qua hàm cắm `load_model(path)` $\rightarrow$ tô màu 10 loại thực thể và xuất bảng `(Thực thể, Loại, Span [start, end))`.
