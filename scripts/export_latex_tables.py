"""CLI xuất các bảng LaTeX chuẩn ACL từ các file kết quả JSON trong outputs/metrics/.

Cách chạy:
  - Xuất bảng cho báo cáo (chỗ chưa có kết quả thật sẽ ghi [CHỜ SỐ LIỆU: ...]):
      python scripts/export_latex_tables.py
  - Chạy thử trên kết quả giả (dummy) để kiểm tra cú pháp bảng LaTeX:
      python scripts/export_latex_tables.py --metrics-dir outputs/metrics/dummy --allow-dummy --out-dir outputs/tables/dummy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.labels import DISPLAY_NAMES, ENTITY_TYPES  # noqa: E402
from src.evaluation.latex_tables import LATEX_OUT_DIR, export_all_tables  # noqa: E402
from src.evaluation.metrics import METRICS_DIR  # noqa: E402


def _load_test_supports_from_eda(metrics_dir: Path) -> dict[str, int] | None:
    """Lấy cột Support thật của tập test từ `data_stats.json` (nếu đã chạy EDA)."""
    stats_file = metrics_dir / "data_stats.json"
    if not stats_file.is_file():
        stats_file = Path("outputs/metrics/data_stats.json")
    if not stats_file.is_file():
        return None
    try:
        data = json.loads(stats_file.read_text(encoding="utf-8"))
        test_counts = data.get("entity_counts_word", {}).get("test", {})
        inv_disp = {v: k for k, v in DISPLAY_NAMES.items()}
        out = {}
        for k, v in test_counts.items():
            etype = inv_disp.get(k, k)
            if etype in ENTITY_TYPES:
                out[etype] = int(v)
        return out if len(out) == len(ENTITY_TYPES) else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất bảng LaTeX chuẩn ACL từ file JSON")
    parser.add_argument("--metrics-dir", default=str(METRICS_DIR))
    parser.add_argument("--out-dir", default=str(LATEX_OUT_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--allow-dummy", action="store_true", help="Cho phép điền số từ file DUMMY_* để chạy thử")
    args = parser.parse_args()

    supports = _load_test_supports_from_eda(Path(args.metrics_dir))
    paths = export_all_tables(
        metrics_dir=args.metrics_dir,
        out_dir=args.out_dir,
        split=args.split,
        allow_dummy=args.allow_dummy,
        supports_override=supports,
    )
    for name, p in paths.items():
        print(f"[latex_tables] Đã xuất {name} -> {p}")


if __name__ == "__main__":
    main()
