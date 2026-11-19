r"""Sinh tự động các bảng LaTeX chuẩn ACL từ file kết quả JSON (schema_version = 1.0).

Tại sao viết như vậy:
- Tuyệt đối không chép tay số liệu từ terminal vào báo cáo LaTeX để tránh sai sót và không bịa số.
- Khi chưa có file JSON kết quả thật (hoặc khi file có tiền tố `DUMMY_` mà không bật `allow_dummy=True`),
  module tự động xuất chuỗi `[CHỜ SỐ LIỆU: <tên chỉ số>]` đúng yêu cầu của đề bài.
- Khi Tín và Đức Anh chạy xong trên Kaggle/Colab và commit file `.json` vào `outputs/metrics/`, chỉ cần
  chạy `python scripts/export_latex_tables.py` là toàn bộ bảng kết quả chính, bảng F1 theo 10 loại thực thể
  và bảng Ablation Study (kèm độ chênh lệch $\Delta$F1 và số vi phạm BIO) được cập nhật tức thì.
- Tự động tô đậm (`\\textbf{...}`) giá trị cao nhất ở từng cột khi tất cả dòng đã có số liệu thật.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from src.data.labels import ENTITY_TYPES, SHORT_NAMES, display_name
from src.evaluation.metrics import METRICS_DIR, load_all_metrics

LATEX_OUT_DIR = Path("outputs/tables")


def latex_escape(text: str) -> str:
    """Escape các ký tự đặc biệt của LaTeX trong tên mô hình/nhãn."""
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def is_dummy_result(result: dict[str, Any]) -> bool:
    """Kiểm tra xem một dict kết quả có phải dữ liệu giả (`DUMMY_`) hay không."""
    name = str(result.get("experiment_name", ""))
    desc = str(result.get("description", ""))
    return name.upper().startswith("DUMMY") or "[DUMMY]" in desc.upper() or "(DUMMY)" in desc.upper()


def filter_results(
    results: Sequence[dict[str, Any]],
    split: str = "test",
    group: str | None = None,
    allow_dummy: bool = False,
) -> list[dict[str, Any]]:
    """Lọc danh sách kết quả theo `split`, `group` và cờ `allow_dummy`."""
    out = []
    for r in results:
        if r.get("split") != split:
            continue
        if group is not None and r.get("group") != group:
            continue
        if not allow_dummy and is_dummy_result(r):
            continue
        out.append(r)
    return out


def _fmt_scores(values: Sequence[float | None], labels: Sequence[str], decimals: int = 2) -> list[str]:
    """Định dạng một cột điểm (%) sang chuỗi LaTeX; tô đậm giá trị lớn nhất nếu đã có số thật."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    max_v = max(valid) if valid else None
    out: list[str] = []
    for v, ph in zip(values, labels):
        if v is None or math.isnan(v):
            out.append(f"[CHỜ SỐ LIỆU: {latex_escape(ph)}]")
        else:
            s = f"{v:.{decimals}f}"
            if max_v is not None and len(valid) >= 2 and abs(v - max_v) < 1e-6:
                s = rf"\textbf{{{s}}}"
            out.append(s)
    return out


def generate_main_results_table(
    results: Sequence[dict[str, Any]],
    split: str = "test",
    allow_dummy: bool = False,
    expected_models: Sequence[tuple[str, str]] = (
        ("bilstm_crf_full", "BiLSTM + CharCNN + CRF (From Scratch)"),
        ("phobert_full", "PhoBERT-base-v2 Fine-tune"),
    ),
) -> str:
    """Bảng 1: So sánh kết quả chính (Micro-P/R/F1, Macro-F1, số vi phạm BIO) trên tập test."""
    pool = {r["experiment_name"]: r for r in filter_results(results, split=split, allow_dummy=allow_dummy)}
    # Bổ sung các mô hình thuộc group='main' chưa có trong expected_models
    rows_spec: list[tuple[str, str, dict[str, Any] | None]] = []
    seen = set()
    for exp_key, disp in expected_models:
        rows_spec.append((exp_key, disp, pool.get(exp_key)))
        seen.add(exp_key)
    for exp_key, r in pool.items():
        if exp_key not in seen and r.get("group") == "main":
            rows_spec.append((exp_key, r.get("description", exp_key), r))

    p_vals = [100 * r["overall"]["micro"]["precision"] if r else None for _, _, r in rows_spec]
    r_vals = [100 * r["overall"]["micro"]["recall"] if r else None for _, _, r in rows_spec]
    f1_vals = [100 * r["overall"]["micro"]["f1"] if r else None for _, _, r in rows_spec]
    mac_vals = [100 * r["overall"]["macro"]["f1"] if r else None for _, _, r in rows_spec]

    p_strs = _fmt_scores(p_vals, [f"{d} Micro-P" for _, d, _ in rows_spec])
    r_strs = _fmt_scores(r_vals, [f"{d} Micro-R" for _, d, _ in rows_spec])
    f1_strs = _fmt_scores(f1_vals, [f"{d} Micro-F1" for _, d, _ in rows_spec])
    mac_strs = _fmt_scores(mac_vals, [f"{d} Macro-F1" for _, d, _ in rows_spec])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Mô hình} & \textbf{Micro-P (\%)} & \textbf{Micro-R (\%)} & \textbf{Micro-F1 (\%)} & \textbf{Macro-F1 (\%)} & \textbf{Vi phạm BIO} \\",
        r"\hline",
    ]
    for i, (_, disp, r) in enumerate(rows_spec):
        if r is None:
            viol_str = f"[CHỜ SỐ LIỆU: {latex_escape(disp)} BIO]"
        else:
            viol_str = str(r["bio_violations"]["pred"]["total"])
        lines.append(
            f"{latex_escape(disp)} & {p_strs[i]} & {r_strs[i]} & {f1_strs[i]} & {mac_strs[i]} & {viol_str} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{Kết quả đánh giá tổng thể mức thực thể (strict IOB2) trên tập \texttt{{{split}}} của PhoNER\_COVID19.}}",
        r"\label{tab:main_results}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def generate_per_entity_table(
    results: Sequence[dict[str, Any]],
    split: str = "test",
    allow_dummy: bool = False,
    expected_models: Sequence[tuple[str, str]] = (
        ("bilstm_crf_full", "BiLSTM-CRF"),
        ("phobert_full", "PhoBERT"),
    ),
    supports_override: dict[str, int] | None = None,
) -> str:
    """Bảng 2: Điểm F1 (%) theo từng loại trong 10 loại thực thể trên tập test."""
    pool = {r["experiment_name"]: r for r in filter_results(results, split=split, allow_dummy=allow_dummy)}
    cols: list[tuple[str, str, dict[str, Any] | None]] = []
    seen = set()
    for exp_key, disp in expected_models:
        cols.append((exp_key, disp, pool.get(exp_key)))
        seen.add(exp_key)
    for exp_key, r in pool.items():
        if exp_key not in seen and r.get("group") == "main":
            cols.append((exp_key, r.get("description", exp_key), r))

    col_spec = "lr" + "c" * len(cols)
    header_models = " & ".join(rf"\textbf{{{latex_escape(disp)}}}" for _, disp, _ in cols)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        rf"\textbf{{Loại thực thể}} & \textbf{{Số lượng}} & {header_models} \\",
        r"\hline",
    ]

    for etype in ENTITY_TYPES:
        dname = display_name(etype)
        sup: int | str = "—"
        if supports_override and etype in supports_override:
            sup = supports_override[etype]
        else:
            for _, _, r in cols:
                if r is not None:
                    sup = r["per_entity"][etype]["support"]
                    break
        vals = [100 * r["per_entity"][etype]["f1"] if r is not None else None for _, _, r in cols]
        strs = _fmt_scores(vals, [f"F1 {dname} ({disp})" for _, disp, _ in cols])
        lines.append(rf"{latex_escape(dname)} (\texttt{{{SHORT_NAMES[etype]}}}) & {sup} & {' & '.join(strs)} \\")

    lines.append(r"\hline")
    for metric_key, label in (("micro", "Micro-F1"), ("macro", "Macro-F1")):
        vals = [100 * r["overall"][metric_key]["f1"] if r is not None else None for _, _, r in cols]
        strs = _fmt_scores(vals, [f"{label} ({disp})" for _, disp, _ in cols])
        lines.append(rf"\textbf{{{label}}} & — & {' & '.join(strs)} \\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{Điểm F1 (\%) theo từng loại thực thể trên tập \texttt{{{split}}}.}}",
        r"\label{tab:per_entity_f1}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_ablation_table(
    results: Sequence[dict[str, Any]],
    group: str = "ablation_bilstm",
    split: str = "test",
    allow_dummy: bool = False,
    expected_rows: Sequence[tuple[str, str]] | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    r"""Bảng 3: Kết quả Ablation Study (Micro-F1, Macro-F1, $\Delta$Micro-F1, Vi phạm BIO)."""
    if expected_rows is None:
        if group == "ablation_bilstm":
            expected_rows = (
                ("bilstm_crf_full", "BiLSTM + CharCNN + CRF (Đầy đủ)"),
                ("bilstm_no_char", "  $-$ Character Embedding (BiLSTM + CRF)"),
                ("bilstm_softmax", "  $-$ Tầng CRF (BiLSTM + CharCNN + Softmax)"),
            )
        else:
            expected_rows = (
                ("phobert_full", "PhoBERT Full Fine-tune + Last Layer + Softmax"),
                ("phobert_concat4", "  $+$ Nối 4 tầng ẩn cuối (Concat Last-4)"),
                ("phobert_crf", "  $+$ Tầng Linear-chain CRF (PhoBERT + CRF)"),
                ("phobert_freeze", "  $-$ Đóng băng Backbone (Freeze PhoBERT)"),
            )

    pool = {r["experiment_name"]: r for r in filter_results(results, split=split, allow_dummy=allow_dummy)}
    rows_spec: list[tuple[str, str, dict[str, Any] | None]] = []
    seen = set()
    for exp_key, disp in expected_rows:
        rows_spec.append((exp_key, disp, pool.get(exp_key)))
        seen.add(exp_key)
    for exp_key, r in pool.items():
        if exp_key not in seen and r.get("group") == group:
            rows_spec.append((exp_key, r.get("description", exp_key), r))

    # Xác định mô hình tham chiếu (is_ablation_reference=True hoặc dòng đầu tiên)
    ref_f1: float | None = None
    for _, _, r in rows_spec:
        if r is not None and r.get("is_ablation_reference"):
            ref_f1 = 100.0 * r["overall"]["micro"]["f1"]
            break
    if ref_f1 is None and rows_spec and rows_spec[0][2] is not None:
        ref_f1 = 100.0 * rows_spec[0][2]["overall"]["micro"]["f1"]

    mic_vals = [100 * r["overall"]["micro"]["f1"] if r else None for _, _, r in rows_spec]
    mac_vals = [100 * r["overall"]["macro"]["f1"] if r else None for _, _, r in rows_spec]
    mic_strs = _fmt_scores(mic_vals, [f"{d} Micro-F1" for _, d, _ in rows_spec])
    mac_strs = _fmt_scores(mac_vals, [f"{d} Macro-F1" for _, d, _ in rows_spec])

    cap = caption or f"Nghiên cứu cắt bỏ (Ablation Study) nhóm \\texttt{{{latex_escape(group)}}} trên tập \\texttt{{{split}}}."
    lab = label or f"tab:{group}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Cấu hình} & \textbf{Micro-F1} & \textbf{Macro-F1} & \textbf{$\Delta$F1} & \textbf{Lỗi BIO} \\",
        r"\hline",
    ]
    for i, (_, disp, r) in enumerate(rows_spec):
        if r is None:
            delta_str = "—"
            bio_str = f"[CHỜ SỐ LIỆU: {latex_escape(disp)} BIO]"
        else:
            cur_f1 = 100.0 * r["overall"]["micro"]["f1"]
            if ref_f1 is None or abs(cur_f1 - ref_f1) < 1e-6:
                delta_str = "0.00"
            else:
                diff = cur_f1 - ref_f1
                delta_str = f"{diff:+.2f}"
            bio_str = str(r["bio_violations"]["pred"]["total"])
        lines.append(f"{latex_escape(disp)} & {mic_strs[i]} & {mac_strs[i]} & {delta_str} & {bio_str} \\\\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{cap}}}",
        rf"\label{{{lab}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def export_all_tables(
    metrics_dir: Path | str = METRICS_DIR,
    out_dir: Path | str = LATEX_OUT_DIR,
    split: str = "test",
    allow_dummy: bool = False,
    supports_override: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Đọc toàn bộ file JSON trong `metrics_dir` và xuất 4 bảng LaTeX chuẩn ACL vào `out_dir`."""
    results = load_all_metrics(metrics_dir, split=split)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "table_main_results.tex": generate_main_results_table(results, split=split, allow_dummy=allow_dummy),
        "table_per_entity_f1.tex": generate_per_entity_table(
            results, split=split, allow_dummy=allow_dummy, supports_override=supports_override
        ),
        "table_ablation_bilstm.tex": generate_ablation_table(
            results, group="ablation_bilstm", split=split, allow_dummy=allow_dummy,
            caption="Ablation Study cho mô hình BiLSTM-CRF tự cài đặt.",
            label="tab:ablation_bilstm",
        ),
        "table_ablation_phobert.tex": generate_ablation_table(
            results, group="ablation_phobert", split=split, allow_dummy=allow_dummy,
            caption="Ablation Study cho mô hình PhoBERT fine-tune.",
            label="tab:ablation_phobert",
        ),
    }
    paths: dict[str, Path] = {}
    for fname, content in tables.items():
        p = out_dir / fname
        p.write_text(content + "\n", encoding="utf-8")
        paths[fname] = p
    return paths
