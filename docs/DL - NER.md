# **Giới thiệu chung**

Qua bài tập lớn, sinh viên tự cài đặt, huấn luyện và phân tích các mô hình học sâu trên một bài toán cụ thể, sau đó viết báo cáo theo format bài báo hội nghị ACL.

Mỗi đề tài có **yêu cầu tối thiểu** (đủ để đạt mức khá) và **phần mở rộng** (cần thiết để đạt mức giỏi). Phân công gợi ý chỉ mang tính tham khảo, mỗi nhóm có thể chia lại nhưng phải ghi rõ trong báo cáo.

# **Đề tài 7\. Nhận dạng thực thể có tên (NER) tiếng Việt**

**Mô tả:** Gán nhãn từng token trong câu để nhận diện tên người, địa danh, tổ chức và các loại thực thể khác. So sánh mô hình gán nhãn chuỗi truyền thống, mô hình pretrained và LLM.

•        **Đầu vào:** Câu tiếng Việt (đã hoặc chưa tách từ).  
•        **Đầu ra:** Chuỗi nhãn theo định dạng BIO cho từng token, từ đó trích ra danh sách thực thể và loại của chúng.  
•        **Dữ liệu:** PhoNER\_COVID19.  
•        **Kiến thức liên quan:** Sequence labeling, BiLSTM-CRF, character embedding, contextual embeddings.  
•        **Yêu cầu tối thiểu:** Cài BiLSTM-CRF từ đầu (có character embedding); so sánh với PhoBERT; đánh giá F1 theo từng loại thực thể.  
•        **Mở rộng:** So sánh mức âm tiết và mức từ; thử few-shot NER bằng LLM và so sánh với mô hình fine-tune.  
•        **Phân công gợi ý:** (1) BiLSTM-CRF; (2) PhoBERT; (3) thí nghiệm LLM và phân tích.

# **PAPER: Tất cả đều viết theo template ACL.**

| Tuần | Nội dung chính | Sản phẩm nộp |
| :---: | :---- | :---- |
| 1 | Đọc 2–3 bài báo chính, chuẩn bị dữ liệu, chạy thử môi trường | **Proposal (tối đa 2 trang)** |
| 2 | Cài đặt và chạy baseline | — |
| 3 | Cài đặt phương pháp chính | **Milestone report (tối đa 4 trang)** |
| 4 | Thí nghiệm, ablation, tinh chỉnh siêu tham số | — |
| 5 | Phần mở rộng, phân tích lỗi, trực quan hóa | Bản nháp báo cáo (không chấm điểm) |
| 6 | Hoàn thiện báo cáo, slide, demo | **Báo cáo cuối, mã nguồn, slide, thuyết trình** |

# **Yêu cầu file báo cáo**

Tất cả các nhóm, kể cả nhóm làm đề tài thị giác máy tính, viết báo cáo theo template chính thức của hội nghị ACL. Báo cáo cuối tối đa 8 trang nội dung, không tính tài liệu tham khảo, mục Limitations và phụ lục.

## **Template và định dạng**

* Dùng template LaTeX chính thức của ACL (kho acl-org/acl-style-files trên GitHub, hoặc bản ACL trên Overleaf). Khuyến khích LaTeX; nhóm dùng Word phải giữ đúng bố cục của template Word ACL.  
* Dùng chế độ không ẩn danh: \\usepackage\[final\]{acl}, ghi đầy đủ họ tên và mã sinh viên của 3 thành viên.  
* Khổ A4, hai cột, giữ nguyên font, cỡ chữ và lề của template; không được chỉnh để nhồi thêm nội dung.  
* Ngôn ngữ: tiếng Việt. Cần thêm gói hỗ trợ tiếng Việt và biên dịch bằng XeLaTeX hoặc LuaLaTeX.  
* Trích dẫn theo kiểu tác giả–năm của ACL (\\citet, \\citep), tài liệu tham khảo quản lý bằng BibTeX.

## **Giới hạn độ dài theo từng mốc**

| Sản phẩm | Nội dung tối đa | Nội dung bắt buộc |
| :---- | :---- | :---- |
| **Proposal** | 2 trang | Bài toán, đầu vào/đầu ra, dữ liệu, baseline dự kiến, cách đánh giá, phân công |
| **Milestone report** | 4 trang | Kết quả baseline, tiến độ, khó khăn, kế hoạch còn lại |
| **Báo cáo cuối** | 8 trang | Toàn bộ cấu trúc ở mục dưới |

## **Cấu trúc báo cáo cuối**

1. **Abstract** (150–250 từ): bài toán, phương pháp, kết quả chính kèm con số.  
2. **Introduction:** động lực, phát biểu bài toán với đầu vào và đầu ra rõ ràng, tóm tắt đóng góp.  
3. **Related Work:** các hướng tiếp cận chính; chỉ trích dẫn tài liệu nhóm đã đọc.  
4. **Method:** mô tả mô hình, hàm loss, có hình minh họa kiến trúc; ghi rõ phần nào nhóm tự cài đặt, phần nào dùng thư viện hoặc mã có sẵn.  
5. **Experiments:** dữ liệu và cách chia train/val/test; thiết lập huấn luyện (siêu tham số, phần cứng, thời gian); chỉ số đánh giá; bảng kết quả so với baseline.  
6. **Analysis:** ablation, phân tích lỗi, trực quan hóa (attention map, Grad-CAM, ví dụ đầu ra tốt và xấu).  
7. **Conclusion:** kết quả chính và hướng phát triển.  
8. **Limitations** (bắt buộc, không tính vào 8 trang): giới hạn của dữ liệu, phương pháp và cách đánh giá.  
9. **Đóng góp của thành viên** (bắt buộc, không tính vào 8 trang): mỗi thành viên làm những phần nào, tỉ lệ đóng góp ước tính.  
10. **References.**  
11. **Appendix** (tùy chọn): kết quả bổ sung, thêm ví dụ, chi tiết siêu tham số. Phụ lục có thể không được đọc khi chấm, nên nội dung chính phải nằm trong 8 trang.

## **Mã nguồn và sản phẩm kèm theo**

* Kho GitHub (công khai hoặc mời giảng viên vào), có README hướng dẫn cài đặt, tải dữ liệu, huấn luyện và đánh giá.  
* File requirements.txt hoặc environment.yml; script hoặc notebook tái tạo được bảng kết quả chính.  
* Không đưa dữ liệu lớn hoặc trọng số mô hình lớn vào kho; dùng link tải ngoài.  
* Slide thuyết trình (10–12 phút) và demo nếu có.  
* Nộp báo cáo dạng PDF, đặt tên theo mẫu NhomXX\_DeTaiYY.pdf.

# **Tiêu chí đánh giá và quy định chung**

Điểm bài tập lớn gồm **70% điểm nhóm** và **30% điểm vấn đáp cá nhân**. Ba thành viên có chung điểm nhóm, còn điểm vấn đáp chấm riêng từng người.

## **Cơ cấu điểm**

| Thành phần | Loại điểm | Tỉ trọng | Nội dung chấm |
| :---- | :---: | :---: | :---- |
| **Kỹ thuật và thí nghiệm** | Nhóm | **25%** | Cài đặt đúng, phần tự cài đạt yêu cầu, thí nghiệm có kiểm soát. Không chấm theo độ chính xác cao hay thấp |
| **Phân tích và báo cáo** | Nhóm | **25%** | Ablation, phân tích lỗi, giải thích kết quả, đúng format ACL, trích dẫn đúng |
| **Quá trình** | Nhóm | **10%** | Proposal và milestone đúng hạn, lịch sử commit và log huấn luyện đều đặn trong 6 tuần |
| **Thuyết trình và demo** | Nhóm | **10%** | Trình bày rõ ràng, demo chạy được |
| **Vấn đáp** | Cá nhân | **30%** | Theo rubric vấn đáp bên dưới |

## **Rubric vấn đáp (thang 10\)**

Mỗi sinh viên được hỏi khoảng 5–7 phút, gồm câu hỏi về phần mình phụ trách và ít nhất một câu về phần của bạn khác trong nhóm. Giảng viên có thể chọn ngẫu nhiên một đoạn code trong repo để hỏi.

| Tiêu chí | Điểm | Câu hỏi mẫu |
| :---- | :---: | :---- |
| **Hiểu phần code mình phụ trách** | **3** | Hàm này làm gì? Vì sao em viết mask như thế này? |
| **Hiểu phương pháp** | **3** | Tại sao Transformer không dùng RNN mà vẫn biết thứ tự từ? |
| **Giải thích kết quả** | **2** | Vì sao mô hình sai nhiều ở lớp này? Có thêm thời gian em sẽ thử gì? |
| **Xử lý tình huống** | **2** | Đổi siêu tham số này thì kết quả thay đổi ra sao? Hoặc sửa nhanh một đoạn code tại chỗ |

**Ngưỡng vấn đáp:** Sinh viên có điểm vấn đáp dưới 4/10 thì tổng điểm bài tập lớn không vượt quá 5/10, dù điểm nhóm cao đến đâu.

## **Quy định chung**

* **Theo dõi quá trình:** Nhóm tạo repo GitHub và mời giảng viên vào ngay từ tuần 1\. Log huấn luyện (Weights & Biases hoặc TensorBoard) phải được chia sẻ với giảng viên.  
* **Kiểm tra trùng lặp:** Code được kiểm tra bằng MOSS hoặc JPlag, so giữa các nhóm và với các repo phổ biến cho cùng bài toán.  
* **Công cụ AI:** Được dùng, nhưng phải khai báo trong báo cáo đã dùng vào việc gì. Phần vấn đáp có thể hỏi cả những đoạn code do AI viết.  
* **Mã nguồn mở và mô hình pretrained:** Được dùng nhưng phải trích dẫn và ghi rõ phần nào nhóm tự cài đặt.  
* **Vượt giới hạn trang:** Phần vượt quá giới hạn không được chấm.  
* **Nộp muộn:** Trừ điểm theo quy định của môn học cho mỗi ngày nộp muộn.  
  **Kết quả không vượt baseline:** Vẫn được điểm tốt nếu thí nghiệm làm đúng và nhóm phân tích được nguyên nhân.

