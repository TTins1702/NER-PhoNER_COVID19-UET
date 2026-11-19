"""Tải và đọc bộ dữ liệu PhoNER_COVID19.

Tại sao viết như vậy:
- Logic được chuyển từ notebook `download_phoner_covid19.ipynb` sang module Python để mọi notebook
  (EDA, train, error analysis) trên Kaggle/Colab gọi chung một hàm: `python -m src.data.download`.
- Chỉ giải nén thư mục `data/` của repo gốc VinAI (không lấy code/README của họ) để repo nhóm
  không chứa mã nguồn của người khác.
- Dữ liệu không bao giờ được commit (thư mục `data/` nằm trong `.gitignore`); mỗi phiên máy ảo
  tự tải lại (~vài giây).
- Hàm đọc kiểm tra ngay `len(words) == len(tags)` và nhãn hợp lệ, để lỗi dữ liệu lộ ra sớm thay vì
  thành lỗi shape khó hiểu lúc huấn luyện.
- Mức từ / mức âm tiết chỉ khác nhau ở tham số `level` -> đổi cấu hình là đổi được thí nghiệm.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import unicodedata
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from src.data.labels import LABEL2ID

# Repo chính thức của bộ dữ liệu (Truong et al., 2021).
REPO_ZIP_URL = "https://github.com/VinAIResearch/PhoNER_COVID19/archive/refs/heads/main.zip"
ZIP_DATA_PREFIX = "PhoNER_COVID19-main/data/"
DEFAULT_DATA_DIR = Path("data")

LEVELS = ("word", "syllable")
SPLITS = ("train", "dev", "test")

Sample = dict  # {"words": list[str], "tags": list[str]}


# ---------------------------------------------------------------------------
# Tải dữ liệu
# ---------------------------------------------------------------------------
def expected_files(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[Path]:
    """Danh sách file JSON bắt buộc phải có sau khi tải (2 mức x 3 tập)."""
    data_dir = Path(data_dir)
    return [data_dir / level / f"{split}_{level}.json" for level in LEVELS for split in SPLITS]


def is_downloaded(data_dir: Path | str = DEFAULT_DATA_DIR) -> bool:
    """True nếu đã có đủ 6 file JSON."""
    return all(p.is_file() and p.stat().st_size > 0 for p in expected_files(data_dir))


def _fetch_bytes(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    """Tải URL, thử lại vài lần vì mạng trên Kaggle/Colab đôi khi chập chờn."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001 - muốn thử lại với mọi lỗi mạng
            last_err = err
            print(f"[download] Lần {attempt}/{retries} thất bại: {err}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Không tải được {url}: {last_err}")


def download_phoner_covid19(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    force: bool = False,
    zip_url: str = REPO_ZIP_URL,
) -> Path:
    """Tải repo PhoNER_COVID19 dạng zip và chỉ giải nén thư mục `data/` vào `data_dir`.

    Kết quả: `data_dir/word/{train,dev,test}_word.{json,conll}` và
    `data_dir/syllable/{train,dev,test}_syllable.{json,conll}`.
    Bỏ qua nếu đã đủ file, trừ khi `force=True`.
    """
    data_dir = Path(data_dir)
    if is_downloaded(data_dir) and not force:
        print(f"[download] '{data_dir}' đã có đủ dữ liệu. Bỏ qua (dùng --force để tải lại).")
        return data_dir

    print(f"[download] Đang tải: {zip_url}")
    raw = _fetch_bytes(zip_url)
    print(f"[download] Đã tải {len(raw) / (1024 * 1024):.2f} MB, đang giải nén thư mục data/ ...")

    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.startswith(ZIP_DATA_PREFIX):
                continue
            rel_path = member.filename[len(ZIP_DATA_PREFIX):]
            target = data_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    missing = [str(p) for p in expected_files(data_dir) if not p.is_file()]
    if missing:
        raise RuntimeError(f"Giải nén xong nhưng thiếu file: {missing}")
    print(f"[download] Hoàn tất: {data_dir.resolve()}")
    return data_dir


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def normalize_text(token: str) -> str:
    """Chuẩn hóa Unicode NFC.

    Tiếng Việt có thể mã hóa một chữ như 'ế' bằng 1 ký tự (NFC) hoặc 'e' + dấu mũ + dấu sắc (NFD).
    Nếu không chuẩn hóa, cùng một từ sẽ có hai id khác nhau trong vocab.
    """
    return unicodedata.normalize("NFC", token)


def validate_sample(sample: Sample, index: int = -1, check_labels: bool = True) -> None:
    """Kiểm tra một câu: có `words`, `tags`, cùng độ dài, khác rỗng, nhãn thuộc 21 nhãn."""
    if "words" not in sample or "tags" not in sample:
        raise ValueError(f"Câu #{index} thiếu khóa 'words' hoặc 'tags': {list(sample)}")
    words, tags = sample["words"], sample["tags"]
    if len(words) != len(tags):
        raise ValueError(f"Câu #{index}: {len(words)} từ nhưng {len(tags)} nhãn")
    if len(words) == 0:
        raise ValueError(f"Câu #{index} rỗng")
    if check_labels:
        unknown = [t for t in tags if t not in LABEL2ID]
        if unknown:
            raise ValueError(f"Câu #{index} có nhãn lạ: {sorted(set(unknown))}")


def load_jsonl(path: Path | str, validate: bool = True, normalize: bool = True) -> list[Sample]:
    """Đọc file JSONL: mỗi dòng là `{"words": [...], "tags": [...]}`."""
    path = Path(path)
    samples: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample = {
                "words": [normalize_text(w) if normalize else w for w in obj["words"]],
                "tags": list(obj["tags"]),
            }
            if validate:
                validate_sample(sample, index=line_no)
            samples.append(sample)
    return samples


def load_conll(path: Path | str, validate: bool = True, normalize: bool = True) -> list[Sample]:
    """Đọc file CoNLL: mỗi dòng `token<TAB/space>tag`, câu cách nhau bởi dòng trống."""
    samples: list[Sample] = []
    words: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        if words:
            sample = {"words": list(words), "tags": list(tags)}
            if validate:
                validate_sample(sample, index=len(samples))
            samples.append(sample)
            words.clear()
            tags.clear()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                flush()
                continue
            parts = line.split()
            token, tag = " ".join(parts[:-1]), parts[-1]
            words.append(normalize_text(token) if normalize else token)
            tags.append(tag)
    flush()
    return samples


def get_split_path(split: str, level: str = "word", data_dir: Path | str = DEFAULT_DATA_DIR,
                   fmt: str = "json") -> Path:
    """Đường dẫn file `{data_dir}/{level}/{split}_{level}.{fmt}`."""
    if level not in LEVELS:
        raise ValueError(f"level phải thuộc {LEVELS}, nhận '{level}'")
    if split not in SPLITS:
        raise ValueError(f"split phải thuộc {SPLITS}, nhận '{split}'")
    return Path(data_dir) / level / f"{split}_{level}.{fmt}"


def load_split(split: str, level: str = "word", data_dir: Path | str = DEFAULT_DATA_DIR,
               fmt: str = "json", validate: bool = True) -> list[Sample]:
    """Đọc một tập (train/dev/test) ở mức `word` hoặc `syllable`."""
    path = get_split_path(split, level, data_dir, fmt)
    if not path.is_file():
        raise FileNotFoundError(
            f"Không thấy {path}. Hãy chạy `python -m src.data.download --data-dir {data_dir}` trước."
        )
    return load_jsonl(path, validate) if fmt == "json" else load_conll(path, validate)


def load_all_splits(level: str = "word", data_dir: Path | str = DEFAULT_DATA_DIR,
                    splits: Iterable[str] = SPLITS) -> dict[str, list[Sample]]:
    """Đọc cả ba tập, trả về `{"train": [...], "dev": [...], "test": [...]}`."""
    return {split: load_split(split, level, data_dir) for split in splits}


def save_jsonl(samples: Iterable[Sample], path: Path | str) -> None:
    """Ghi danh sách câu ra JSONL (dùng cho test và dữ liệu giả)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"words": s["words"], "tags": s["tags"]}, ensure_ascii=False) + "\n")


def _summary(data_dir: Path) -> None:
    for level in LEVELS:
        for split in SPLITS:
            samples = load_split(split, level, data_dir)
            n_tok = sum(len(s["words"]) for s in samples)
            print(f"  {level:8s} {split:5s}: {len(samples):5d} câu | {n_tok:6d} token")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải và kiểm tra PhoNER_COVID19")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--force", action="store_true", help="Tải lại kể cả khi đã có dữ liệu")
    args = parser.parse_args()
    data_dir = download_phoner_covid19(args.data_dir, force=args.force)
    print("[download] Kiểm tra dữ liệu:")
    _summary(Path(data_dir))


if __name__ == "__main__":
    main()
