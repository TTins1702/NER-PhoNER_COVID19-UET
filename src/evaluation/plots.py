"""Hàm vẽ biểu đồ chuẩn ACL (đọc tốt ở thang xám) cho phân tích kết quả & lỗi NER.

Tại sao viết như vậy:
- Quy định báo cáo ACL yêu cầu mọi hình vẽ phải đọc rõ khi in đen trắng (grayscale): không chỉ dựa vào
  màu sắc mà kết hợp **độ đậm xám + hoa văn gạch chéo (hatch) + kiểu đường (nét liền/đứt)**.
- Tách hàm vẽ ra module `src/evaluation/plots.py` giúp `notebooks/04_error_analysis.ipynb` gọn nhẹ,
  đồng thời có thể gọi từ script hoặc kiểm thử tự động trong `tests/`.
- Xuất đồng thời `.pdf` (vector, nhúng font Type 42 cho LaTeX) và `.png` (xem nhanh trên GitHub/notebook).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.data.labels import ENTITY_TYPES, SHORT_NAMES
from src.evaluation.error_analysis import ERROR_NAMES_VI, ERROR_TYPES, NONE_LABEL

FIGURES_DIR = Path("outputs/figures")

# Bảng sắc độ xám và hoa văn (hatch) giúp phân biệt tới 4 mô hình khi in đen trắng.
GRAY_PALETTE = ("0.15", "0.50", "0.78", "0.92")
HATCH_PATTERNS = ("", "///", "...", "xxx")
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")


def configure_acl_style() -> None:
    """Thiết lập thông số matplotlib phù hợp khổ cột báo cáo ACL (cột đơn ~3.6 in, cột đôi ~7.2 in)."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, name: str, out_dir: Path | str = FIGURES_DIR) -> tuple[Path, Path]:
    """Lưu đồng thời file PDF (cho LaTeX) và PNG (cho notebook/slide)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=200)
    return pdf_path, png_path


def plot_per_entity_f1(
    results: Sequence[dict[str, Any]],
    labels: Sequence[str] | None = None,
    sort_by_first: bool = True,
    title: str = "F1 theo từng loại thực thể (%)",
) -> plt.Figure:
    """Biểu đồ cột ngang so sánh F1 (%) của 10 loại thực thể giữa các mô hình."""
    configure_acl_style()
    if not results:
        raise ValueError("Danh sách kết quả rỗng")
    model_labels = list(labels) if labels is not None else [
        r.get("description") or r.get("experiment_name", f"Model {i + 1}")
        for i, r in enumerate(results)
    ]

    types = list(ENTITY_TYPES)
    if sort_by_first:
        types.sort(key=lambda t: results[0]["per_entity"][t]["f1"])

    n_models = len(results)
    y = np.arange(len(types))
    bar_h = min(0.36, 0.78 / max(1, n_models))

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for k, (res, m_label) in enumerate(zip(results, model_labels)):
        offset = (k - (n_models - 1) / 2.0) * bar_h
        vals = [100.0 * res["per_entity"][t]["f1"] for t in types]
        ax.barh(
            y + offset,
            vals,
            height=bar_h,
            color=GRAY_PALETTE[k % len(GRAY_PALETTE)],
            hatch=HATCH_PATTERNS[k % len(HATCH_PATTERNS)],
            edgecolor="black",
            linewidth=0.6,
            label=f"{m_label} (Micro={100 * res['overall']['micro']['f1']:.1f}%)",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_NAMES[t] for t in types])
    ax.set_xlim(0, 108)
    ax.set_xlabel("F1-score (%)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5, color="0.4")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    mat: np.ndarray,
    labels: Sequence[str],
    normalize_rows: bool = True,
    exclude_outside_diagonal: bool = True,
    title: str = "Ma trận nhầm lẫn mức thực thể (%)",
) -> plt.Figure:
    """Vẽ heatmap ma trận nhầm lẫn ở thang xám (`Greys`) kèm số trên từng ô.

    Nếu `normalize_rows=True`, mỗi hàng được chuẩn hóa theo tổng hàng (%) để các loại thực thể
    hiếm (JOB, TRANSPORTATION) vẫn nhìn rõ tỉ lệ nhầm lẫn.
    """
    configure_acl_style()
    mat_disp = mat.astype(float).copy()
    if exclude_outside_diagonal and labels[-1] == NONE_LABEL:
        mat_disp[-1, -1] = 0.0

    if normalize_rows:
        row_sums = mat_disp.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        values = 100.0 * mat_disp / row_sums
    else:
        values = mat_disp

    short_labels = [SHORT_NAMES.get(x, x) for x in labels]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(values, cmap="Greys", vmin=0.0, vmax=max(1.0, float(values.max())))

    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Nhãn dự đoán (Predicted)")
    ax.set_ylabel("Nhãn thật (Gold)")
    ax.set_title(title)

    thresh = 0.55 * float(values.max() if values.max() > 0 else 1.0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if v <= 0:
                continue
            txt = f"{v:.0f}" if normalize_rows else f"{int(mat_disp[i, j])}"
            color = "white" if v > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% theo hàng" if normalize_rows else "Số lượng")
    fig.tight_layout()
    return fig


def plot_error_breakdown(
    reports: dict[str, dict[str, int]],
    title: str = "Phân loại 4 nhóm lỗi thực thể",
) -> plt.Figure:
    """Biểu đồ cột so sánh số lượng 4 nhóm lỗi (Boundary, Type, FP, FN) giữa các mô hình."""
    configure_acl_style()
    model_names = list(reports.keys())
    x = np.arange(len(ERROR_TYPES))
    n_models = len(model_names)
    width = min(0.35, 0.78 / max(1, n_models))

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for k, m_name in enumerate(model_names):
        offset = (k - (n_models - 1) / 2.0) * width
        counts = [reports[m_name].get(et, 0) for et in ERROR_TYPES]
        bars = ax.bar(
            x + offset,
            counts,
            width=width,
            color=GRAY_PALETTE[k % len(GRAY_PALETTE)],
            hatch=HATCH_PATTERNS[k % len(HATCH_PATTERNS)],
            edgecolor="black",
            linewidth=0.6,
            label=m_name,
        )
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2.0, h + 0.5, f"{int(h)}",
                        ha="center", va="bottom", fontsize=7)

    max_count = max([reports[m].get(et, 0) for m in model_names for et in ERROR_TYPES] + [1])
    ax.set_ylim(0, max_count * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels([ERROR_NAMES_VI[et] for et in ERROR_TYPES])
    ax.set_ylabel("Số lượng lỗi")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color="0.4")
    ax.legend(frameon=False, loc="upper center", ncol=min(2, len(model_names)))
    fig.tight_layout()
    return fig


def plot_bio_violations(
    results: Sequence[dict[str, Any]],
    labels: Sequence[str] | None = None,
    title: str = "Số vi phạm ràng buộc BIO trong dự đoán",
) -> plt.Figure:
    """Biểu đồ cột so sánh tổng số nhãn I-X vi phạm luật BIO (chứng minh tác dụng của tầng CRF)."""
    configure_acl_style()
    model_labels = list(labels) if labels is not None else [
        r.get("description") or r.get("experiment_name", f"Model {i + 1}")
        for i, r in enumerate(results)
    ]
    totals = [r["bio_violations"]["pred"]["total"] for r in results]
    x = np.arange(len(model_labels))

    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    bars = ax.bar(
        x,
        totals,
        width=0.52,
        color=[GRAY_PALETTE[i % len(GRAY_PALETTE)] for i in range(len(x))],
        hatch=[HATCH_PATTERNS[i % len(HATCH_PATTERNS)] for i in range(len(x))],
        edgecolor="black",
        linewidth=0.6,
    )
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + max(0.3, 0.02 * max(totals + [1])),
                f"{int(h)}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(totals + [1]) * 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.set_ylabel("Số nhãn I-X không hợp lệ")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color="0.4")
    fig.tight_layout()
    return fig


def plot_recall_by_length(
    length_recalls: dict[str, dict[str, dict[str, float]]],
    title: str = "Recall theo độ dài thực thể (số từ)",
) -> plt.Figure:
    """Đường biểu diễn Recall (%) theo độ dài thực thể (1, 2, 3, >=4 từ) giữa các mô hình."""
    configure_acl_style()
    model_names = list(length_recalls.keys())
    buckets = list(next(iter(length_recalls.values())).keys())
    x = np.arange(len(buckets))

    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    for k, m_name in enumerate(model_names):
        ys = [100.0 * length_recalls[m_name][b]["recall"] for b in buckets]
        ax.plot(
            x,
            ys,
            linestyle=LINE_STYLES[k % len(LINE_STYLES)],
            marker=MARKERS[k % len(MARKERS)],
            color="0.1" if k == 0 else "0.45",
            linewidth=1.5,
            label=m_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} từ" for b in buckets])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Độ dài thực thể (số từ)")
    ax.set_ylabel("Recall (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5, color="0.4")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
