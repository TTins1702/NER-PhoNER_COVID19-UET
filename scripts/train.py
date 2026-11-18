import argparse
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
from src.utils.config import load_config
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Train a NER model from config, evaluate dev/test splits, and save metrics JSON."""
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "data")
    level = str(data_cfg.get("level", "word"))

    splits = {
        split: load_phoner_split(data_dir, level=level, split=split)
        for split in ("train", "dev", "test")
    }
    model, loaders, vocab_state = build_model_and_dataloaders(config, splits)

    logger = ExperimentLogger(config)
    try:
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            logger=logger,
            vocab_state=vocab_state,
        )
        fit_summary = trainer.fit(loaders["train"], loaders["dev"])
        dev_metrics = trainer.evaluate(loaders["dev"])
        test_metrics = trainer.evaluate(loaders["test"])
        logger.log_metrics(
            {
                "final_dev_micro_f1": float(dev_metrics["micro_f1"]),
                "final_test_micro_f1": float(test_metrics["micro_f1"]),
                "final_test_macro_f1": float(test_metrics["macro_f1"]),
                "final_test_bio_violations": float(test_metrics["bio_violations"]),
            }
        )
    finally:
        logger.finish()

    experiment_name = str(config.get("experiment_name", "experiment"))
    output_dir = Path(config.get("logging", {}).get("output_dir", "outputs/metrics"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "best_epoch": fit_summary["best_epoch"],
        "best_dev_f1": fit_summary["best_dev_f1"],
        "checkpoint_path": fit_summary["checkpoint_path"],
        "history": fit_summary["history"],
        "dev": dev_metrics,
        "test": test_metrics,
        "config": config,
    }
    out_path = output_dir / f"{experiment_name}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhoNER_COVID19 NER model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="Optional config overrides in key.path=value format",
    )
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.override)
    summary = run_training(config)
    print(
        f"Training finished: {summary['experiment_name']} | "
        f"dev_f1={summary['dev']['micro_f1']:.4f} | test_f1={summary['test']['micro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
