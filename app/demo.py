"""Giao diện Demo Gradio cho hệ thống Nhận dạng Thực thể Có tên (NER) trên PhoNER_COVID19.

Cách chạy:
  - Chạy chế độ thử nghiệm (mô hình giả):
      python -m app.demo
  - Chạy với checkpoint thật của Tín (BiLSTM-CRF) hoặc Đức Anh (PhoBERT):
      python -m app.demo --model checkpoints/bilstm_crf_best.pt
"""

from __future__ import annotations

import argparse
from typing import Any

from app.predictor import load_model, predict_sentence

# Bảng màu phân biệt 10 loại thực thể trên giao diện HighlightedText.
COLOR_MAP: dict[str, str] = {
    "PATIENT_ID": "#ffb3ba",
    "NAME": "#ffdfba",
    "AGE": "#ffffba",
    "GENDER": "#baffc9",
    "JOB": "#bae1ff",
    "LOCATION": "#c9c9ff",
    "ORGANIZATION": "#f1cbff",
    "SYMPTOM_AND_DISEASE": "#ff9aa2",
    "TRANSPORTATION": "#b5ead7",
    "DATE": "#e2f0cb",
}

SAMPLE_SENTENCES: list[str] = [
    "Bệnh nhân BN17 (nữ, 26 tuổi) đi chuyến bay VN0054 từ Anh về Hà Nội ngày 2/3/2020, có triệu chứng sốt và ho.",
    "Theo Bộ Y tế, bệnh nhân 91 là phi công (nam, 43 tuổi) được điều trị viêm phổi tại Bệnh viện Chợ Rẫy.",
    "Ngày 27/7, chị N.H.N. (30 tuổi, điều dưỡng tại Bệnh viện Đà Nẵng) có biểu hiện sốt, khó thở.",
    "Bệnh nhân BN416 trú tại Đà Nẵng có tiền sử tăng huyết áp, được cách ly theo hướng dẫn của Sở Y tế.",
    "CDC Hà Nội thông báo ca bệnh 1347 từng di chuyển bằng chuyến bay VN54 ngày 19/3.",
]


def run_demo_inference(
    text: str,
    tokenize_mode: str,
    checkpoint_path: str,
    state_cache: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str | None]], list[list[Any]], list[list[Any]], str]:
    """Hàm callback cho nút 'Nhận dạng thực thể' trên Gradio."""
    cache = state_cache if state_cache is not None else {}
    ckpt_key = (checkpoint_path or "dummy").strip()
    try:
        if cache.get("ckpt_key") != ckpt_key or "predictor" not in cache:
            cache["predictor"] = load_model(ckpt_key)
            cache["ckpt_key"] = ckpt_key
        predictor = cache["predictor"]
        out = predict_sentence(text, predictor=predictor, tokenize_mode=tokenize_mode)
        status = (
            f"Mô hình đang dùng: {out['model_name']} | "
            f"Số token sau tách từ: {len(out['words'])} | "
            f"Số thực thể phát hiện: {len(out['entities'])}"
        )
        return out["highlights"], out["entities"], out["token_rows"], status
    except Exception as err:  # noqa: BLE001
        return [(text, None)], [], [], f"Lỗi khi nạp mô hình / dự đoán: {err}"


def build_demo(default_model_path: str = "dummy"):
    """Khởi tạo đối tượng `gr.Blocks` (chỉ import gradio khi gọi hàm này)."""
    import gradio as gr

    shared_cache: dict[str, Any] = {
        "ckpt_key": default_model_path,
        "predictor": load_model(default_model_path),
    }

    with gr.Blocks(title="PhoNER_COVID19 — Demo Nhận dạng Thực thể Tiếng Việt") as demo:
        gr.Markdown(
            "## Nhận dạng Thực thể Có tên (NER) Tiếng Việt trên miền COVID-19 (`PhoNER_COVID19`)\n"
            "Nhập câu tiếng Việt bên dưới. Hệ thống tự động tách từ, gán nhãn BIO cho 10 loại thực thể "
            "(`PATIENT_ID`, `NAME`, `AGE`, `GENDER`, `JOB`, `LOCATION`, `ORGANIZATION`, "
            "`SYMPTOM_AND_DISEASE`, `TRANSPORTATION`, `DATE`) và trích xuất danh sách thực thể."
        )

        with gr.Row():
            ckpt_box = gr.Textbox(
                value=default_model_path,
                label="Đường dẫn Checkpoint (.pt) hoặc 'dummy' để chạy thử",
                scale=3,
            )
            tok_radio = gr.Radio(
                choices=[("Tách từ tự động (Underthesea / word-level)", "underthesea"),
                         ("Giữ nguyên theo khoảng trắng (syllable / đã tách sẵn)", "whitespace")],
                value="underthesea",
                label="Chế độ tách token",
                scale=2,
            )

        input_box = gr.Textbox(
            value=SAMPLE_SENTENCES[0],
            lines=3,
            label="Câu tiếng Việt đầu vào",
            placeholder="Nhập câu tiếng Việt cần nhận dạng thực thể...",
        )
        run_btn = gr.Button("Nhận dạng thực thể (Analyze)", variant="primary")
        status_md = gr.Textbox(label="Trạng thái", interactive=False)

        highlight_out = gr.HighlightedText(
            label="Kết quả tô màu thực thể",
            combine_adjacent=False,
            show_legend=True,
            color_map=COLOR_MAP,
        )

        with gr.Row():
            entity_table = gr.Dataframe(
                headers=["#", "Thực thể", "Token mức từ", "Nhãn (Data)", "Nhãn (Báo cáo)", "Span [start, end)"],
                datatype=["number", "str", "str", "str", "str", "str"],
                label="Danh sách thực thể trích xuất",
                interactive=False,
            )
            token_table = gr.Dataframe(
                headers=["Chỉ số", "Token", "Nhãn BIO"],
                datatype=["number", "str", "str"],
                label="Chuỗi nhãn BIO từng token",
                interactive=False,
            )

        gr.Examples(
            examples=[[s] for s in SAMPLE_SENTENCES],
            inputs=[input_box],
            label="Các câu ví dụ mẫu (bấm để chọn)",
        )

        def _cb(text: str, tok_mode: str, ckpt: str):
            return run_demo_inference(text, tok_mode, ckpt, state_cache=shared_cache)

        run_btn.click(_cb, inputs=[input_box, tok_radio, ckpt_box],
                      outputs=[highlight_out, entity_table, token_table, status_md])
        input_box.submit(_cb, inputs=[input_box, tok_radio, ckpt_box],
                         outputs=[highlight_out, entity_table, token_table, status_md])
        demo.load(_cb, inputs=[input_box, tok_radio, ckpt_box],
                  outputs=[highlight_out, entity_table, token_table, status_md])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy Demo Gradio cho PhoNER_COVID19")
    parser.add_argument("--model", default="dummy", help="Đường dẫn checkpoint .pt hoặc 'dummy'")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Tạo link public (dùng trên Colab/Kaggle)")
    args = parser.parse_args()

    demo = build_demo(default_model_path=args.model)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
