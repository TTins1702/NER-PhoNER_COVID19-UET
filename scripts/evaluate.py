import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.data.download import load_phoner_split
from src.training.trainer import Trainer, build_model_and_dataloaders
from src.utils.config import load_config


def run_evaluation(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    """Load a saved checkpoint and evaluate it on the specified PhoNER_COVID19 split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    effective_config = config or ckpt.get("config", {})
    data_cfg = effective_config.get("data", {})
    data_dir = data_cfg.get("data_dir", "data")
    level = str(data_cfg.get("level", "word"))

    records = load_phoner_split(data_dir, level=level, split=split)
    model, loaders, _ = build_model_and_dataloaders(
        effective_config,
        {split: records},
        vocab_state=ckpt.get("vocab_state"),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    trainer = Trainer(model=model, config=effective_config, device=device)
    return trainer.evaluate(loaders[split])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved PhoNER_COVID19 NER checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="Optional config overrides in key.path=value format",
    )
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.override)
    metrics = run_evaluation(config, checkpoint_path=args.checkpoint, split=args.split)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
