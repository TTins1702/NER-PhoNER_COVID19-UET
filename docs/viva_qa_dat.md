# Cẩm nang Ôn tập Vấn đáp Cá nhân (Viva Q&A) — Đạt (UET-VNU)

> **Tỉ trọng vấn đáp:** 30% tổng điểm (hiểu code mình phụ trách: 3đ; hiểu phương pháp: 3đ; giải thích kết quả: 2đ; xử lý tình huống: 2đ). **Lưu ý:** Dưới 4/10 vấn đáp thì tổng điểm học phần bị khống chế $\le 5/10$. Giảng viên có thể hỏi bất kỳ đoạn code nào trong repo (kể cả phần của Tín và Đức Anh).

---

## PHẦN 1. 5 CÂU HỎI BẮT BUỘC TRỌNG TÂM (HIỂU SÂU BẢN CHẤT)

### Câu 1. Vì sao trong `src/data/bilstm_dataset.py` em thiết kế `mask` kiểu `bool` `[B, L]` và lại pad `tags` bằng `0` (nhãn `O`) thay vì `-100` như `CrossEntropyLoss`?
**Trả lời:**
1. **Tại sao cần `mask [B, L]` kiểu `torch.bool` (`True` ở từ thật, `False` ở vị trí `<pad>`):**
   - Trong một mini-batch, các câu có độ dài thực tế $n_1, n_2, \dots, n_B$ khác nhau và được đệm (pad) về độ dài câu dài nhất trong batch $L = \max_b n_b$.
   - Tầng `Linear-chain CRF` tính hàm mất mát $-\log P(\mathbf{y} \mid \mathbf{x}) = \log Z(\mathbf{x}) - s(\mathbf{x}, \mathbf{y})$ trên từng bước thời gian $t = 1 \dots L$:
     - Khi tính điểm chuỗi nhãn thật $s(\mathbf{x}, \mathbf{y})$, ta nhân điểm phát xạ (emission) và điểm chuyển trạng thái (transition) tại bước $t$ với `mask[:, t].float()` để các vị trí `<pad>` không đóng góp điểm nào vào $s(\mathbf{x}, \mathbf{y})$.
     - Khi chạy quy hoạch động **Forward** tính $\log Z(\mathbf{x})$, tại bước thời gian $t$ mà `mask[b, t] == False`, ta giữ nguyên vector trạng thái $\alpha_t = \alpha_{t-1}$ bằng `torch.where(mask[:, t].unsqueeze(1), alpha_next, alpha)` để câu ngắn kết thúc đúng ở từ thật cuối cùng của nó.
     - Khi giải mã **Viterbi**, ta dùng `lengths = mask.sum(dim=1)` để chỉ truy vết đúng $n_b$ nhãn đầu tiên, trả về chuỗi dự đoán có độ dài đúng bằng `len(words)`.
2. **Tại sao pad `tags` bằng `0` (`O_ID = 0`) thay vì `-100`:**
   - Trong thuật toán tính $s(\mathbf{x}, \mathbf{y})$ của tầng CRF, ở mọi bước thời gian $t = 0 \dots L-1$, ta đều gọi phép truy xuất chỉ số tensor `emissions[:, t].gather(1, tags[:, t].unsqueeze(1))` và `transitions[tags[:, t-1], tags[:, t]]` trên toàn batch **trước khi** nhân với `mask[:, t]`.
   - Nếu vị trí pad trong `tags` mang giá trị `-100`, phép `gather` hoặc indexing trên GPU sẽ báo lỗi truy cập ngoài miền chỉ số (`CUDA error: device-side assert triggered`, vì chỉ số hợp lệ phải thuộc $[0, 20]$).
   - Đặt `TAG_PAD_ID = O_ID = 0` đảm bảo chỉ số luôn hợp lệ khi `gather`, còn giá trị thu được tại vị trí pad sẽ bị nhân với `mask[:, t] = 0` nên hoàn toàn không ảnh hưởng tới loss hay gradient.
   - Khi chạy nhánh Ablation không có CRF (`use_crf = False`), ta chỉ cần viết `F.cross_entropy(emissions[mask], tags[mask])` là cũng loại bỏ hoàn toàn các vị trí pad.

---

### Câu 2. Vì sao chỉ chấm F1 ở mức từ (`word-level`) và mức thực thể (`seqeval`) mà tuyệt đối không chấm ở mức subword hay bằng `sklearn` (mức token)?
**Trả lời:**
Có hai lý do độc lập và rất quan trọng:
1. **Tại sao phải chấm ở mức thực thể (`entity-level`) bằng `seqeval` chứ không chấm ở mức token (`sklearn.metrics.f1_score`):**
   - Trong `PhoNER_COVID19` mức từ, nhãn ngoài thực thể `O` chiếm tới **$74{,}17\%$ – $79{,}05\%$** tổng số từ. Nếu chấm accuracy hay F1 ở mức từng token rời rạc, một mô hình đoán đúng nhãn `O` và chỉ đoán trúng 1 từ trong một thực thể dài 4 từ (ví dụ `"Bệnh_viện Bệnh_nhiệt_đới Trung_ương"`) vẫn được điểm token F1 rất cao, dù thực thể rút trích ra bị cắt cụt và sai hoàn toàn về mặt ứng dụng.
   - Bài toán NER yêu cầu rút trích **danh sách thực thể hoàn chỉnh**: một thực thể dự đoán $(c, s, e)$ chỉ có giá trị khi đúng **cả ranh giới bắt đầu $s$, kết thúc $e$ lẫn loại thực thể $c$**. Thư viện `seqeval` đối chiếu trực tiếp hai tập thực thể chuẩn và dự đoán theo đúng chuẩn hội nghị CoNLL.
2. **Tại sao phải quy về mức từ gốc (`word-level`, độ dài đúng bằng `len(words)`) chứ không chấm ở mức subword của PhoBERT:**
   - `BiLSTM-CRF` làm việc trực tiếp trên đơn vị từ (`words`), còn `PhoBERT` dùng bộ tách BPE có thể tách một từ tiếng Việt (như tên riêng, địa danh, mã bệnh nhân) thành $2$–$4$ subword.
   - Nếu chấm điểm của `PhoBERT` trên chuỗi subword: (a) mẫu số và đơn vị thực thể giữa hai mô hình khác nhau nên **không thể so sánh công bằng**; (b) các từ bị tách thành nhiều subword sẽ bị đếm lặp hoặc một lỗi dự đoán ở subword giữa làm gãy thực thể theo cách khác với mức từ.
   - Do đó, Hợp đồng giao diện của nhóm bắt buộc mọi mô hình phải trả về `y_pred: list[list[str]]` có độ dài từng câu bằng đúng `len(words)` của file gốc (được kiểm tra cưỡng bức bằng hàm `check_alignment` trong `src/evaluation/predictions.py`).

---

### Câu 3. Trong `src/evaluation/metrics.py`, chế độ `seqeval` `mode="strict", scheme=IOB2` khác với chế độ mặc định (lỏng) như thế nào?
**Trả lời:**
- Trong lược đồ **IOB2**, mọi thực thể hợp lệ bắt buộc phải mở đầu bằng nhãn `B-X` và các token tiếp theo (nếu có) mang nhãn `I-X` cùng loại. Nhãn `I-X` **không bao giờ** được phép đứng đầu câu, đứng ngay sau `O`, hoặc đứng ngay sau `B-Y`/`I-Y` khác loại ($Y \ne X$).
- **Chế độ mặc định (lỏng / conlleval truyền thống):** Khi gặp một chuỗi vi phạm quy tắc BIO như `["O", "I-LOCATION", "I-LOCATION"]`, bộ chấm điểm mặc định tự động "tha thứ" và ngầm hiểu nhãn `I-LOCATION` đầu tiên là bắt đầu một thực thể `LOCATION` mới. Chúng em đã kiểm chứng bằng code: với nhãn thật `["B-AGE", "I-AGE"]` và nhãn dự đoán sai luật `["I-AGE", "I-AGE"]`, chế độ mặc định của `seqeval` chấm **F1 = 1.0 (100%)**!
- **Chế độ `mode="strict", scheme=IOB2`:** Yêu cầu thực thể dự đoán phải tuân thủ đúng cú pháp IOB2 (phải mở đầu bằng `B-X`). Trong ví dụ trên, chuỗi dự đoán `["I-AGE", "I-AGE"]` bị coi là vi phạm cấu trúc, không tạo thành thực thể hợp lệ nào và bị chấm **F1 = 0.0**.
- **Ý nghĩa khoa học đối với báo cáo của nhóm:** Mô hình dùng đầu dự đoán `Softmax` phân loại độc lập từng từ nên rất hay sinh ra nhãn `I-X` mồ côi; trong khi tầng `Linear-chain CRF` học ma trận chuyển trạng thái để phạt nặng phép chuyển `O -> I-X`. Nếu dùng chế độ lỏng, bộ chấm điểm đã tự sửa hộ lỗi BIO cho Softmax, làm che khuất hoàn toàn lợi ích của tầng CRF trong Ablation Study!

---

### Câu 4. Vì sao `WordVocab` trong `src/data/vocab.py` chỉ được xây từ tập `train`? Ngưỡng `min_freq` và token `<unk>` hoạt động thế nào?
**Trả lời:**
1. **Chỉ xây `WordVocab` từ tập `train` để chống rò rỉ dữ liệu (*Data Leakage*):**
   - Trong thực tế triển khai và theo nguyên lý học máy, tập `dev` và `test` đại diện cho dữ liệu tương lai chưa từng biết khi huấn luyện.
   - Nếu đưa cả từ vựng của `dev`/`test` vào `WordVocab`: (a) ta đã gian lận thông tin từ tập kiểm thử; (b) các từ chỉ xuất hiện trong `test` nhưng không có trong `train` sẽ được gán một ID riêng trong bảng `nn.Embedding`, nhưng vì từ đó không hề xuất hiện trong vòng lặp huấn luyện trên tập `train`, vector embedding của nó **chưa bao giờ được cập nhật gradient** (vẫn là nhiễu khởi tạo ngẫu nhiên Gaussian ban đầu!), làm mô hình dự đoán sai nặng hơn cả khi quy về một vector `<unk>` đã được huấn luyện.
2. **Vai trò của `min_freq` và cách học vector `<unk>`:**
   - Khi đặt `min_freq = 2` (hoặc `min_freq > 1`), các từ xuất hiện chỉ $1$ lần trong tập `train` ($5.169 - 3.274 = 1.895$ từ hiếm) được ánh xạ về `UNK_ID = 1` (`<unk>`) ngay trong quá trình huấn luyện.
   - Nhờ vậy, vector embedding tại chỉ số `1` (`<unk>`) nhận được gradient từ các từ hiếm trong tập `train` và học được một biểu diễn ngữ nghĩa "từ hiếm/từ mới". Khi đánh giá trên tập `test` gặp từ ngoài từ điển (OOV), từ đó được tra về `UNK_ID = 1` (vector đã được học) kết hợp với vector hình thái từ `CharEncoder` (vẫn đọc được từng chữ cái của từ OOV đó).

---

### Câu 5. Vì sao khi huấn luyện và đánh giá `PhoBERT` phải dùng quy tắc `first-subword` (chỉ lấy dự đoán ở subword đầu tiên của mỗi từ)?
**Trả lời:**
- Trong `PhoNER_COVID19` mức từ (`data/word/`), mỗi phần tử trong `sample["words"]` là một từ đã ghép âm tiết bằng dấu `"_"` (ví dụ `"Vĩnh_Phúc"` hoặc `"BN1347"`), tương ứng với đúng $1$ nhãn trong `sample["tags"]`.
- Khi đưa từ đó qua bộ BPE Tokenizer của `vinai/phobert-base-v2`, một từ có thể bị bẻ thành $k \ge 1$ mảnh subword: ví dụ `"BN1347"` $\rightarrow$ `["BN@@", "13@@", "47"]` ($k=3$ subword).
- **Khi huấn luyện:** Ta cần ánh xạ 1-1 giữa mỗi từ gốc và nhãn BIO của nó. Theo chuẩn BERT/RoBERTa (Devlin et al., 2019), ta gán nhãn BIO gốc của từ cho **subword đầu tiên** (`"BN@@"` nhận `B-PATIENT_ID`, `word_mask = True`), còn các subword đuôi (`"13@@"`, `"47"`) cùng token đặc biệt `<s>`, `</s>`, `<pad>` được gán `ignore_index = -100` để hàm `CrossEntropyLoss` không tính loss trên chúng. (Nếu gán lặp lại `B-PATIENT_ID` cho cả 3 subword thì mô hình sẽ học sai rằng 3 thực thể liên tiếp bắt đầu ngay trong cùng một từ!).
- **Khi đánh giá:** Ta chỉ trích xuất nhãn dự đoán tại các vị trí `word_mask == True` (tức subword đầu tiên của mỗi từ). Nhờ mỗi từ gốc có đúng 1 subword đầu tiên, chuỗi nhãn thu được sẽ có độ dài **bằng đúng `len(words)`** của câu gốc, khôi phục hoàn hảo về mức từ để đưa vào `seqeval` chấm điểm công bằng với `BiLSTM-CRF`.

---

## PHẦN 2. GIẢI THÍCH TỪNG FILE CODE ĐẠT PHỤ TRÁCH (TIÊU CHÍ 1 — 3/10 ĐIỂM)

| File trong repo | Hàm / Lớp cốt lõi | Câu trả lời ngắn gọn khi giảng viên chỉ vào code hỏi "Hàm này làm gì, tại sao viết thế?" |
| :--- | :--- | :--- |
| `src/data/labels.py` | `LABELS`, `LABEL2ID`, `ID2LABEL`, `extract_entities` | Cố định 21 nhãn BIO (`O` id=0, 10 loại xếp alphabet) để 3 thành viên không bao giờ lệch ID nhãn. `extract_entities` trích danh sách `(etype, start, end)` theo đúng luật **strict IOB2** (chỉ mở thực thể tại `B-X`, nối qua `I-X` cùng loại, bỏ qua `I-X` mồ côi) — đã kiểm thử khớp 100% với `seqeval` trên 3.000 chuỗi ngẫu nhiên. |
| `src/data/download.py` | `download_phoner_covid19`, `normalize_text`, `load_jsonl`, `validate_sample` | Tải file zip từ repo VinAI bằng `urllib` (có retry 3 lần phòng mạng Colab chập chờn), chỉ giải nén thư mục `data/` bằng `zipfile.ZipFile(io.BytesIO(raw))` ngay trên RAM mà không tạo rác trên ổ đĩa. Khi đọc câu, chuẩn hóa Unicode `NFC` để tránh lỗi chữ tiếng Việt tổ hợp (`NFD` vs `NFC`) và kiểm tra ngay `len(words) == len(tags)`. |
| `src/data/vocab.py` | `WordVocab`, `CharVocab`, `vietnamese_letters` | `WordVocab` xây từ `train`, sắp xếp theo `(-freq, word)` để thứ tự ID hoàn toàn tất định (deterministic). `CharVocab` sinh sẵn toàn bộ 12 nguyên âm tiếng Việt $\times$ 6 thanh điệu (cả hoa lẫn thường) + phụ âm + chữ số + dấu câu, đảm bảo mọi chữ cái tiếng Việt có dấu đều có ID riêng khi đưa vào `CharEncoder`. |
| `src/data/bilstm_dataset.py` | `BiLSTMNERDataset`, `bilstm_collate_fn`, `build_bilstm_dataloaders` | Mã hóa câu sang ID ngay trong `__init__` một lần duy nhất để tăng tốc DataLoader. `bilstm_collate_fn` thực hiện **dynamic padding** theo chiều dài tối đa $L$ và độ dài từ tối đa $W$ của riêng batch đó (thay vì pad cố định tới 162 từ), tiết kiệm bộ nhớ GPU và thời gian chạy LSTM. Không sort lại thứ tự câu trong batch để dự đoán khớp thứ tự file gốc. |
| `src/evaluation/predictions.py` | `check_alignment`, `save_predictions`, `load_predictions` | Lưu dự đoán ra JSONL `{"id", "words", "gold", "pred"}`. Hàm `check_alignment` kiểm tra cưỡng bức `len(y_pred[i]) == len(y_true[i]) == len(words[i])` và mọi nhãn thuộc 21 nhãn BIO; nếu mô hình quên gộp subword sẽ báo lỗi rõ ràng ở câu thứ mấy ngay lập tức. |
| `src/evaluation/metrics.py` | `find_bio_violations`, `count_bio_violations`, `repair_bio`, `compute_metrics`, `evaluate_and_save` | Gọi `seqeval.metrics.classification_report(..., mode="strict", scheme=IOB2, output_dict=True)` lấy P/R/F1 của cả 10 loại thực thể + micro/macro/weighted avg. Đồng thời đếm số nhãn `I-X` vi phạm quy tắc BIO trên dự đoán thô và đóng gói ra file JSON theo schema cố định `schema_version = "1.0"`. |
| `src/evaluation/error_analysis.py` | `classify_sentence`, `analyze_errors`, `entity_confusion_matrix`, `recall_by_entity_length`, `recall_by_oov`, `compare_models` | Đối chiếu thực thể gold và pred của từng câu, loại bỏ các thực thể đúng hoàn toàn (TP), rồi phân loại mọi thực thể dự đoán sai vào đúng 1 trong 3 nhóm (`boundary`, `type`, `false_positive`) và các thực thể chuẩn không được chạm tới vào `false_negative`. Đảm bảo bất biến toán học: `boundary + type + FP = num_pred - correct`. |
| `src/evaluation/plots.py` | `plot_per_entity_f1`, `plot_confusion_matrix`, `plot_error_breakdown`, `plot_bio_violations`, `plot_recall_by_length` | Vẽ biểu đồ chuẩn báo cáo ACL: kết hợp sắc độ xám (`0.15`, `0.50`, `0.78`) với hoa văn gạch chéo (`hatch`: `""`, `"///"`, `"..."`) và kiểu nét đường (`-`, `--`) để in đen trắng vẫn phân biệt rõ từng mô hình; xuất vector `.pdf` nhúng font Type 42. |
| `src/evaluation/latex_tables.py` & `scripts/export_latex_tables.py` | `generate_main_results_table`, `generate_per_entity_table`, `generate_ablation_table`, `export_all_tables` | Tự động đọc các file JSON kết quả để xuất 4 bảng LaTeX chuẩn ACL (`\textbf{}` cho giá trị cao nhất cột, tự tính $\Delta\text{F1}$ cho bảng ablation). Khi chưa có kết quả thật (hoặc file có tiền tố `DUMMY_` mà không bật `--allow-dummy`), tự động điền `[CHỜ SỐ LIỆU: ...]`. |
| `app/predictor.py` & `app/demo.py` | `tokenize_vietnamese`, `load_model`, `predict_sentence`, `build_demo` | Ứng dụng Demo Gradio: tự động tách từ câu tiếng Việt bằng `Underthesea` (nối âm tiết từ ghép bằng `"_"`), gọi hàm cắm `load_model(path)` hỗ trợ cả checkpoint thật lẫn `DummyNERPredictor`, hiển thị văn bản tô màu thực thể và bảng danh sách thực thể rút trích. |

---

## PHẦN 3. CÂU HỎI CHÉO VỀ PHẦN CỦA TÍN (BiLSTM-CRF) & ĐỨC ANH (PhoBERT) (TIÊU CHÍ 2 — 3/10 ĐIỂM)

### Câu 6. (Phần của Tín) Tại sao hàm `log_sum_exp` trong tầng CRF phải trừ đi giá trị lớn nhất $m = \max_i v_i$? Công thức và độ phức tạp của thuật toán Forward / Viterbi là gì?
**Trả lời:**
- Khi tính hàm phân hoạch $\log Z(\mathbf{x})$ bằng quy hoạch động Forward, ở mỗi bước $t$ ta cần tính $\log \sum_{i=1}^{K} \exp(v_i)$ với $K = 21$ nhãn và $v_i = \alpha_{t-1}(i) + A_{i, j} + P_{t, j}$.
- Nếu tính trực tiếp $\exp(v_i)$, khi câu dài ($N \approx 100$ từ), tổng điểm tích lũy $v_i$ có thể lên tới hàng백 (ví dụ $v_i = 120 > 88{,}7$ là ngưỡng tràn của `float32`), làm $\exp(v_i) = +\infty$ (**tràn số trên — overflow**); ngược lại nếu $v_i = -120$, $\exp(v_i) = 0$ hết dẫn tới $\log(0) = -\infty$ (**tràn số dưới — underflow**).
- Thủ thuật ổn định số học (*Log-Sum-Exp trick*): đặt $m = \max_i v_i$, ta có hằng đẳng thức:
  $$\text{log\_sum\_exp}(\mathbf{v}) = m + \log \sum_{i=1}^{K} \exp(v_i - m)$$
  Vì $v_i - m \le 0$ với mọi $i$ và có ít nhất một số hạng bằng $\exp(0) = 1$, tổng bên trong $\log$ luôn nằm trong đoạn $[1, K] = [1, 21]$, hoàn toàn không bao giờ bị tràn số!
- **Độ phức tạp thời gian:** Ở mỗi bước thời gian $t \in \{1, \dots, L\}$, ta xét mọi cặp chuyển nhãn $(y_{t-1} = i, y_t = j)$ với $i, j \in \{1, \dots, K\}$, nên độ phức tạp cho một câu dài $L$ là **$O(L \cdot K^2)$** (với $K = 21 \Rightarrow K^2 = 441$, rất nhỏ). Thuật toán Forward dùng phép `log_sum_exp` (tổng xác suất mọi đường đi để tính loss khi train), còn Viterbi thay bằng phép `max` và lưu mảng `backpointers` (tìm 1 đường đi có điểm cao nhất khi inference).

### Câu 7. (Phần của Tín) Tại sao cần `CharEncoder` (1D-CNN trên ký tự) ghép với `WordEmbedding`?
**Trả lời:**
- `WordEmbedding` coi mỗi từ là một đơn vị nguyên tử (atomic ID): hai từ `"BN17"` và `"BN1347"` có hai ID hoàn toàn độc lập, và nếu `"BN1347"` chưa xuất hiện trong tập `train` thì nó bị biến thành `<unk>`.
- `CharEncoder` nhúng từng ký tự trong từ thành ma trận $[W, d_c]$, trượt bộ lọc tích chập `Conv1d` (ví dụ kernel size $k=3$) qua chuỗi ký tự rồi lấy `MaxPool1d` theo chiều dài từ $W$. Nhờ đó nó bắt được các **mẫu hình thái học (morphology)**: tiền tố `"BN"` + chữ số $\rightarrow$ `PATIENT_ID`; chữ cái viết hoa kèm dấu chấm `"N.H.N."` $\rightarrow$ `NAME`;2 chữ cái in hoa + 4 chữ số `"VN0054"` $\rightarrow$ `TRANSPORTATION`.

### Câu 8. (Phần của Đức Anh) Tại sao Transformer / PhoBERT không dùng RNN (như BiLSTM) mà vẫn biết được thứ tự của các từ trong câu?
**Trả lời:**
- Phép toán cốt lõi của Transformer là **Self-Attention**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$. Nếu chỉ có Self-Attention thuần túy thì phép toán này **bất biến với phép hoán vị** (*permutation-equivariant*): đổi chỗ hai từ trong câu chỉ làm đổi chỗ vector đầu ra tương ứng chứ không làm thay đổi giá trị tích vô hướng giữa chúng.
- Để mô hình biết thứ tự từ, trước khi đưa vào tầng Transformer đầu tiên, PhoBERT cộng trực tiếp **vector mã hóa vị trí (*Positional Embedding* học được, độ dài tối đa $258$ token)** vào vector nhúng subword: $\mathbf{z}_t^{(0)} = \mathbf{e}_{\text{subword}}(s_t) + \mathbf{p}_t$. Nhờ $\mathbf{p}_t$ khác nhau tại mỗi vị trí $t$, ma trận truy vấn $Q$ và khóa $K$ mang cả thông tin nội dung từ lẫn vị trí của từ trong câu.
- **So với BiLSTM:** BiLSTM xử lý tuần tự $t = 1 \dots L$ nên thông tin giữa hai từ cách xa nhau phải đi qua $O(L)$ bước cổng nhớ (dễ suy giảm trên câu dài), trong khi Self-Attention kết nối trực tiếp hai vị trí bất kỳ trong câu chỉ qua **$O(1)$ bước** (độ dài đường truyền bằng 1, chi phí tính toán $O(L^2)$).

### Câu 9. (Phần của Đức Anh) Tại sao khi fine-tune PhoBERT lại dùng tốc độ học phân tầng (*Differential Learning Rate*: backbone $2\times 10^{-5}$, tầng classifier/CRF $10^{-3}$)?
**Trả lời:**
- Backbone `PhoBERT-base-v2` ($135$ triệu tham số) đã được tiền huấn luyện kỹ trên $20$GB văn bản tiếng Việt nên các trọng số của nó đã nằm gần cực tiểu tốt; nếu dùng learning rate lớn ($10^{-3}$), gradient lớn ở những epoch đầu sẽ phá hỏng tri thức tiền huấn luyện (*catastrophic forgetting*).
- Ngược lại, tầng phân loại `Linear` hoặc ma trận chuyển trạng thái `CRF` gắn thêm ở đầu ra được **khởi tạo ngẫu nhiên từ đầu**, nên cần learning rate lớn hơn ($10^{-3}$) để hội tụ nhanh theo kịp đặc trưng của backbone.

---

## PHẦN 4. GIẢI THÍCH KẾT QUẢ & XỬ LÝ TÌNH HUỐNG TẠI CHỖ (TIÊU CHÍ 3 & 4 — 4/10 ĐIỂM)

### Câu 10. (Giải thích kết quả) Vì sao cả hai mô hình thường đạt F1 rất cao ($>95\%$) ở `PATIENT_ID`, `AGE`, `GENDER` nhưng lại thấp hơn rõ rệt ở `JOB` (`OCCUPATION`), `TRANSPORTATION` và `ORGANIZATION`?
**Trả lời:**
Dựa trực tiếp trên số liệu EDA nhóm đã phân tích trong `01_eda_data_stats.ipynb`:
1. **`PATIENT_ID`, `AGE`, `GENDER` rất dễ vì:**
   - **Độ dài cố định bằng 1 từ:** $99{,}8\%$ `PATIENT_ID`, $99{,}7\%$ `AGE` và $100\%$ `GENDER` ở mức từ chỉ dài đúng $1$ token $\rightarrow$ hầu như **không có lỗi ranh giới (Boundary Error)**.
   - **Khuôn mẫu từ vựng & ngữ cảnh hẹp:** `GENDER` chỉ xoay quanh `{nam, nữ}` ($0\%$ OOV); `AGE` là số đứng ngay trước chữ `"tuổi"` ($0{,}2\%$ OOV); `PATIENT_ID` luôn có tiền tố `"BN"` hoặc đứng ngay sau chữ `"bệnh_nhân"`.
2. **`JOB` và `TRANSPORTATION` khó vì:**
   - **Số mẫu huấn luyện cực ít (mất cân bằng lớp):** Chỉ có $205$ thực thể `JOB` ($1{,}46\%$) và $226$ thực thể `TRANSPORTATION` ($1{,}45\%$) trong toàn bộ tập `train`.
   - **Tỉ lệ OOV cao:** `TRANSPORTATION` có tới **$34{,}7\%$** token trên tập `test` chưa từng xuất hiện trong `train` (số hiệu chuyến bay, biển số xe mới).
3. **`ORGANIZATION` và `LOCATION` hay sai vì:**
   - **Thực thể dài:** `ORGANIZATION` có độ dài trung bình $3{,}24$ từ (dài nhất tới $14$ từ), `LOCATION` dài tới $13$ từ $\rightarrow$ rất dễ bị cắt thiếu/thừa 1 từ ở biên (trong strict IOB2, lệch 1 từ ranh giới bị phạt cả 1 FP lẫn 1 FN).
   - **Nhập nhằng ngữ nghĩa `LOCATION` $\leftrightarrow$ `ORGANIZATION`:** Các bệnh viện, trường học, ủy ban vừa là địa điểm dịch tễ (`LOCATION`) vừa là cơ quan (`ORGANIZATION`), chỉ phân biệt được nhờ động từ/giới từ điều khiển trong câu.

### Câu 11. (Tình huống 1) Nếu có thêm thời gian, em sẽ thử kỹ thuật gì để cải thiện F1 cho các lớp hiếm (`JOB`, `TRANSPORTATION`) và giảm lỗi ranh giới?
**Trả lời:**
1. **Ghép nối 4 tầng ẩn cuối của PhoBERT (`concat_last_4`) + Tầng CRF:** Đã được thiết kế sẵn trong Ablation Study của nhóm; tầng CRF giúp giữ tính liên tục `B-ORG -> I-ORG` trên các thực thể dài 5–10 từ.
2. **Đánh trọng số lớp (Class-weighted Loss) hoặc Focal Loss / Oversampling câu chứa lớp hiếm:** Tăng trọng số hàm mất mát cho các câu chứa `JOB`, `TRANSPORTATION`, `NAME` để gradient không bị áp đảo bởi `LOCATION` và nhãn `O`.
3. **Bổ sung đặc trưng tiền tố/định dạng (Regex/Gazetteer features) hoặc Data Augmentation thay thế thực thể cùng loại:** Hoán đổi các danh từ chỉ nghề nghiệp và số hiệu phương tiện trong tập `train` để giảm hiện tượng học vẹt từ vựng.

### Câu 12. (Tình huống 2 — Sửa code tại chỗ) Giảng viên yêu cầu: *"Bây giờ thầy muốn trong `src/data/vocab.py` hoặc `bilstm_dataset.py` gom tất cả các chữ số trong từ về chữ số `'0'` (ví dụ `'BN1347'` $\rightarrow$ `'BN0000'`, `'27/7'` $\rightarrow$ `'00/0'`) để giảm tỉ lệ OOV cho `PATIENT_ID` và `DATE` thì em sửa ở đâu?"*
**Trả lời:**
- Em **đã thiết kế sẵn** tham số `normalize_digits: bool = False` ngay trong `WordVocab` (`src/data/vocab.py` dòng 43–64) và `build_bilstm_dataloaders` (`src/data/bilstm_dataset.py` dòng 129)!
- Cụ thể, trong hàm `WordVocab.normalize(self, word)`:
  ```python
  if self.normalize_digits:
      word = re.sub(r"\d", "0", word)
  ```
- Khi bật `normalize_digits=True`, mọi số hiệu bệnh nhân 4 chữ số mới trong tập `test` như `"BN1347"`, `"BN1452"` đều được chuẩn hóa về cùng một token `"BN0000"` đã xuất hiện trong tập `train`, giúp tỉ lệ OOV của `PATIENT_ID` và `DATE` trong `WordVocab` giảm xuống gần $0\%$, trong khi `CharEncoder` vẫn nhận nguyên chuỗi ký tự gốc `item["words"]` chưa bị thay số.

### Câu 13. (Tình huống 3 — Sửa code tại chỗ) Giảng viên yêu cầu: *"Nếu muốn tính thêm F1 ở chế độ lỏng (chấp nhận cả thực thể trùng một phần — partial/relaxed match, không cần khớp tuyệt đối ranh giới) trong `src/evaluation/error_analysis.py` thì tính từ các biến nào đã có?"*
**Trả lời:**
- Trong `analyze_errors` (`src/evaluation/error_analysis.py`), nhóm đã phân tách mọi thực thể thành `correct` (khớp hoàn toàn cả loại và ranh giới), `boundary` (đúng loại thực thể nhưng lệch ranh giới, có phần giao $>0$), `type` (giao nhau nhưng sai loại), `false_positive` và `false_negative`.
- Do đó, để tính **Relaxed Type-Match F1** (chỉ cần đúng loại và có giao nhau với thực thể thật), số dự đoán đúng mức lỏng chính là:
  $$\text{TP}_{\text{relaxed}} = \text{\texttt{report.counts["correct"]}} + \text{\texttt{report.counts["boundary"]}}$$
  Từ đó suy ra ngay:
  $$\text{P}_{\text{relaxed}} = \frac{\text{TP}_{\text{relaxed}}}{\text{\texttt{report.num\_pred\_entities}}}, \quad \text{R}_{\text{relaxed}} = \frac{\text{TP}_{\text{relaxed}}}{\text{\texttt{report.num\_gold\_entities}}}.$$
  Độ chênh lệch $\text{F1}_{\text{relaxed}} - \text{F1}_{\text{strict}}$ chính là phần trăm F1 bị mất thuần túy do lỗi ranh giới!
