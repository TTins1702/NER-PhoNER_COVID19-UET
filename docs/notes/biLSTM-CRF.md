# Ghi chú BiLSTM-CRF 

Ký hiệu kích thước batch:
- $B$: batch size
- $L$: độ dài câu dài nhất trong batch (số từ)
- $W$: độ dài từ dài nhất trong batch (số ký tự)
- $C = 21$: số nhãn BIO cố định (`O` + 10 thực thể $\times$ `B`/`I`)

---

## 1. Luồng xử lý chung

1. `word_ids [B, L]` $\to$ `WordEmbedding` $\to$ `[B, L, d_w]`
2. `char_ids [B, L, W]` $\to$ `CharEncoder` $\to$ `[B, L, d_c]`
3. Ghép 2 vector theo chiều cuối (`cat(..., dim=-1)`), qua `Dropout` $\to$ `[B, L, d_w + d_c]`
4. Đưa qua `BiLSTM` (dùng `pack_padded_sequence` theo `mask.sum(dim=1)` để bỏ qua `<pad>`), rồi `Linear(2 * hidden, C)` $\to$ ma trận điểm phát xạ `emissions` $E \in \mathbb{R}^{B \times L \times C}$
5. `CRF`:
   - Train: nhận `(E, tags, mask)` $\to$ trả về NLL loss.
   - Eval/Infer: nhận `(E, mask)` $\to$ Viterbi trả về chuỗi nhãn dự đoán độ dài thật $L_b$ cho từng câu.

---

## 2. Character Encoder

Dùng để xử lý từ hiếm / OOV (`BN1452`, `SARS-CoV-2`, tên riêng, ngày tháng, chữ viết hoa).

### CharCNN (Ma & Hovy, 2016)
- Gom chiều batch và từ: `[B, L, W]` $\to$ `[B*L, W]`
- `Embedding(num_chars, d_char, padding_idx=0)` $\to$ `[B*L, W, d_char]`
- Đổi trục cho `Conv1d`: `permute(0, 2, 1)` $\to$ `[B*L, d_char, W]`
- `Conv1d(d_char, num_filters, kernel_size=k, padding=k//2)` + `ReLU` + `max(dim=-1)` $\to$ `[B*L, num_filters]`
- Tách lại thành câu: `view(B, L, num_filters)`
- **Lưu ý:** Cần `padding=k//2` (hoặc pad $W \ge k$). Nếu không, gặp batch toàn từ 1 ký tự (`W=1`) với `kernel_size=3` sẽ lỗi runtime.

### CharBiLSTM (Lample et al., 2016)
- `Embedding`: `[B*L, W, d_char]` $\to$ `LSTM(d_char, d_c_hidden, bidirectional=True, batch_first=True)`
- Lấy hidden cuối của chiều xuôi (tại ký tự thật cuối cùng của từ) ghép với hidden đầu của chiều ngược (index `0`) $\to$ `[B*L, 2 * d_c_hidden]` $\to$ `[B, L, 2 * d_c_hidden]`.
- Nếu dùng `pack_padded_sequence` theo độ dài từ: các vị trí từ `<pad>` có độ dài 0 phải tạm clamp lên 1 trước khi pack, sau đó nhân lại bằng `mask`.
- CharCNN chạy song song theo $W$ nên nhanh hơn; CharBiLSTM chạy tuần tự nên chậm hơn nhưng bắt thứ tự ký tự xa tốt hơn.

---

## 3. Linear-chain CRF

Softmax tính độc lập từng token nên dễ sinh nhãn BIO sai luật (`O -> I-LOC`, `B-NAME -> I-DATE`). CRF chấm điểm trên toàn chuỗi nhãn $y = (y_0, \dots, y_{L_b-1})$.

### Tham số của `LinearChainCRF`
- `transitions` $T \in \mathbb{R}^{C \times C}$: $T_{i, j}$ là điểm chuyển từ nhãn $i$ (bước $t-1$) sang nhãn $j$ (bước $t$).
- `start_transitions` $S \in \mathbb{R}^{C}$: $S_j$ là điểm bắt đầu câu bằng nhãn $j$.
- `end_transitions` $F \in \mathbb{R}^{C}$: $F_i$ là điểm kết thúc câu ở nhãn $i$.

### Điểm chuỗi (Gold Score) và Loss
Với câu có độ dài thật $L_b$ ($1 \le L_b \le L$):

$$s(x, y) = S_{y_0} + E_{0, y_0} + \sum_{t=1}^{L_b - 1} \big(T_{y_{t-1}, y_t} + E_{t, y_t}\big) + F_{y_{L_b - 1}}$$

$$P(y \mid x) = \frac{\exp(s(x, y))}{Z(x)}, \qquad Z(x) = \sum_{\tilde{y}} \exp(s(x, \tilde{y}))$$

$$\mathcal{L}_{\text{NLL}} = -\log P(y \mid x) = \log Z(x) - s(x, y) \ge 0$$

- **Lưu ý tính `gold_score` với `mask`:** `end_transitions` phải cộng tại `last_idx = mask.long().sum(dim=1) - 1` (dùng `gather`), tuyệt đối không lấy `tags[:, -1]` vì cuối batch có thể là `<pad>`.

---

## 4. `log_sum_exp` ổn định số học

Tính trực tiếp $\log \sum \exp(a_i)$ dễ bị tràn số (`inf` khi $a_i$ lớn) hoặc `log(0) = -inf` (khi mọi $a_i$ quá âm). Đặt $m = \max_i a_i$:

$$\text{log\_sum\_exp}(\mathbf{a}) = m + \log \sum_{i=1}^{C} \exp(a_i - m)$$

Khi trừ $m$: số mũ lớn nhất là $0$ ($\exp(0) = 1$), các số mũ còn lại $\le 0$ nên tổng bên trong $\log$ luôn nằm trong $[1, C]$, tránh cả overflow lẫn underflow.

---

## 5. Thuật toán Forward (tính $\log Z(x)$, độ phức tạp $O(L \cdot C^2)$)

$\alpha_t(j)$ là log-tổng-exp điểm của mọi tiền tố độ dài $t+1$ kết thúc tại nhãn $j$.

1. **Khởi tạo ($t = 0$):**
   `alpha = start_transitions + emissions[:, 0]` $\to$ `[B, C]`
2. **Lặp $t = 1 \dots L-1$:**
   - Vector hóa toàn bộ cặp chuyển $(i \to j)$:
     ```python
     # [B, C, 1] + [1, C, C] + [B, 1, C] -> [B, C, C]
     next_alpha = alpha.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
     next_alpha = log_sum_exp(next_alpha, dim=1)  # [B, C]
     ```
   - **Xử lý `mask`:** Nếu `mask[b, t] == False` (vị trí `<pad>`), phải giữ nguyên `alpha` cũ:
     ```python
     alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)
     ```
     Không được nhân `alpha * mask`, vì đặt `alpha = 0` thì $\exp(0) = 1$ vẫn đóng góp vào $Z(x)$.
3. **Kết thúc:** Vì `alpha` bị đóng băng từ vị trí $L_b$ trở đi, sau vòng lặp `alpha` giữ đúng trạng thái tại $L_b - 1$:
   `log_Z = log_sum_exp(alpha + end_transitions, dim=1)` $\to$ `[B]`

---

## 6. Thuật toán Viterbi (giải mã $\arg\max_y s(x, y)$, độ phức tạp $O(L \cdot C^2)$)

Cấu trúc giống Forward nhưng thay `log_sum_exp` bằng `max` và lưu con trỏ truy vết (`history`).

1. **Khởi tạo ($t = 0$):**
   `score = start_transitions + emissions[:, 0]` (`[B, C]`), `history = []`
2. **Lặp $t = 1 \dots L-1$:**
   ```python
   # [B, C, 1] + [1, C, C] -> [B, C, C]
   cand = score.unsqueeze(2) + transitions.unsqueeze(0)
   best_prev_score, best_prev_tag = cand.max(dim=1)  # [B, C]
   next_score = best_prev_score + emissions[:, t]
   score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
   history.append(best_prev_tag)  # history[k] map bước k+1 -> k
   ```
3. **Truy vết từng câu $b$ với độ dài thật $L_b = \text{mask}[b].\text{sum}()$:**
   - Nhãn cuối: `best_tag = (score[b] + end_transitions).argmax().item()`
   - Chỉ lần ngược qua `history[:L_b - 1]` (bỏ qua toàn bộ bước padding $t \ge L_b$):
     ```python
     best_path = [best_tag]
     for bptr in reversed(history[: L_b - 1]):
         best_tag = bptr[b, best_tag].item()
         best_path.append(best_tag)
     best_path.reverse()
     ```

---

## 7. Sanity checks khi viết unit test cho `crf.py`

1. **Loss không âm:** Với mọi input hợp lệ, $\log Z(x) - s(x, y) \ge 0$.
2. **Bất biến theo padding:** Chạy 1 câu độ dài 5 với `batch_size=1` (không pad) vs chạy chính câu đó trong batch được pad lên 20 $\to$ NLL loss và chuỗi Viterbi của câu đó phải bằng nhau tuyệt đối.
3. **Đối chiếu vét cạn trên câu nhỏ:** Thử `L=4, C=3`, liệt kê cả $3^4 = 81$ chuỗi để kiểm tra $\log Z(x)$ từ `forward` và đường đi tối ưu từ `decode`.
