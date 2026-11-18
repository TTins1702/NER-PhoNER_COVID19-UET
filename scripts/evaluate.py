import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.data.download import load_phoner_split
from src.training.trainer import Trainer, build_model_and_dataloaders
from src.utils.config import _parse_override_value, _set_nested_value, load_config


def _merge_eval_config(
    ckpt_config: dict[str, Any] | None,
    user_config: dict[str, Any] | None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve evaluation config, preserving checkpoint model architecture when available."""
    base = copy.deepcopy(ckpt_config) if ckpt_config else {}
    if user_config:
        if not base:
            base = copy.deepcopy(user_config)
        else:
            for section in ("data", "logging", "training"):
                if section in user_config:
                    base[section] = {**base.get(section, {}), **user_config[section]}
            if "model" not in base and "model" in user_config:
                base["model"] = copy.deepcopy(user_config["model"])

    if not base:
        raise ValueError("No configuration found in --config or checkpoint")

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must follow 'key.path=value' format, got {item!r}")
            key_path, raw_val = item.split("=", 1)
            _set_nested_value(base, key_path, _parse_override_value(raw_val))

    return base


def run_evaluation(
    config: dict[str, Any] | None,
    checkpoint_path: str | Path,
    split: str = "test",
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load a saved checkpoint and evaluate it on the specified PhoNER_COVID19 split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    effective_config = _merge_eval_config(ckpt.get("config"), config, overrides=overrides)
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
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Optional YAML config file (defaults to config stored inside checkpoint)",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="Optional config overrides in key.path=value format",
    )
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    metrics = run_evaluation(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        overrides=args.override,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
