import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

REPO_ZIP_URL = "https://github.com/VinAIResearch/PhoNER_COVID19/archive/refs/heads/main.zip"


def download_phoner_covid19(output_dir: str | Path = "data", force: bool = False) -> Path:
    """Download and extract only the 'data/' folder of PhoNER_COVID19 into output_dir."""
    out_path = Path(output_dir)
    if out_path.exists() and any(out_path.iterdir()) and not force:
        print(f"Data directory '{out_path}' already exists. Skipping download.")
        return out_path

    print(f"Downloading PhoNER_COVID19 from {REPO_ZIP_URL} ...")
    with urllib.request.urlopen(REPO_ZIP_URL, timeout=60) as resp:
        raw_bytes = resp.read()

    out_path.mkdir(parents=True, exist_ok=True)
    prefix = "PhoNER_COVID19-main/data/"
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            rel_path = member.filename[len(prefix):]
            target_path = out_path / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target_path.open("wb") as dst:
                dst.write(src.read())

    print(f"Extracted PhoNER_COVID19 to {out_path.resolve()}")
    return out_path


def load_jsonl(path: str | Path) -> list[dict[str, list[str]]]:
    """Load JSONL records from a PhoNER_COVID19 .json file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_phoner_split(
    data_dir: str | Path,
    level: str = "word",
    split: str = "train",
) -> list[dict[str, list[str]]]:
    """Load a specific PhoNER_COVID19 split ('train', 'dev', 'test') and level ('word', 'syllable')."""
    if level not in ("word", "syllable"):
        raise ValueError(f"Unsupported level {level!r}; expected 'word' or 'syllable'")
    if split not in ("train", "dev", "test"):
        raise ValueError(f"Unsupported split {split!r}; expected 'train', 'dev', or 'test'")

    file_path = Path(data_dir) / level / f"{split}_{level}.json"
    return load_jsonl(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PhoNER_COVID19 dataset")
    parser.add_argument("--output-dir", default="data", help="Destination directory for dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if directory is non-empty")
    args = parser.parse_args()
    download_phoner_covid19(output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
