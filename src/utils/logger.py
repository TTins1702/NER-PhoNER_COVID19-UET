import os
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Log training/evaluation metrics to console and optionally Weights & Biases."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        log_cfg = config.get("logging", {})
        self.use_wandb = bool(log_cfg.get("use_wandb", False))
        self.output_dir = Path(log_cfg.get("output_dir", "outputs/metrics"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run: Any = None

        if self.use_wandb:
            if not os.environ.get("WANDB_API_KEY"):
                os.environ.setdefault("WANDB_MODE", "offline")
            import wandb

            entity = log_cfg.get("wandb_entity") or os.environ.get("WANDB_ENTITY") or None
            init_kwargs: dict[str, Any] = {
                "project": log_cfg.get("wandb_project", "ner-phoner-covid19-uet"),
                "name": config.get("experiment_name"),
                "config": config,
            }
            if entity:
                init_kwargs["entity"] = entity
            self._run = wandb.init(**init_kwargs)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Print metrics to console and forward them to W&B when enabled."""
        formatted = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        prefix = f"[step {step}] " if step is not None else ""
        print(f"{prefix}{formatted}")
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        """Close the active W&B run if one was initialized."""
        if self._run is not None:
            self._run.finish()
            self._run = None
