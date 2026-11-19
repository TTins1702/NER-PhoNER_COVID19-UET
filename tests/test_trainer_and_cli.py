import json
import os
import subprocess
import sys
from pathlib import Path
from scripts.evaluate import run_evaluation
from scripts.train import run_training


def test_end_to_end_bilstm_crf_train_and_evaluate(tmp_path: Path) -> None:
    word_dir = tmp_path / "data" / "word"
    word_dir.mkdir(parents=True)
    sample_lines = (
        '{"words": ["Bệnh_nhân", "19", "ở", "Hà_Nội"], "tags": ["O", "B-PATIENT_ID", "O", "B-LOCATION"]}\n'
        '{"words": ["Bệnh_nhân", "20", "sốt"], "tags": ["O", "B-PATIENT_ID", "B-SYMPTOM_AND_DISEASE"]}\n'
    )
    for split in ("train", "dev", "test"):
        (word_dir / f"{split}_word.json").write_text(sample_lines, encoding="utf-8")

    ckpt_dir = tmp_path / "checkpoints"
    metrics_dir = tmp_path / "metrics"
    cfg = {
        "experiment_name": "test_bilstm_e2e",
        "seed": 42,
        "data": {"level": "word", "data_dir": str(tmp_path / "data"), "min_word_freq": 1, "max_word_len": 15},
        "model": {
            "name": "bilstm_crf",
            "word_emb_dim": 16,
            "use_char": True,
            "char_encoder_type": "cnn",
            "char_emb_dim": 8,
            "char_out_dim": 12,
            "char_kernel_size": 3,
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.1,
            "use_crf": True,
        },
        "training": {
            "batch_size": 2,
            "epochs": 2,
            "lr": 0.01,
            "weight_decay": 0.0,
            "grad_clip": 5.0,
            "patience": 2,
            "checkpoint_dir": str(ckpt_dir),
        },
        "logging": {"use_wandb": False, "wandb_project": "test", "output_dir": str(metrics_dir)},
    }

    summary = run_training(cfg)
    best_ckpt = ckpt_dir / "best_model.pt"
    out_json = metrics_dir / "test_bilstm_e2e.json"
    assert best_ckpt.exists()
    assert out_json.exists()
    assert "test" in summary and "micro_f1" in summary["test"]

    eval_metrics = run_evaluation(cfg, checkpoint_path=best_ckpt, split="test")
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert abs(eval_metrics["micro_f1"] - saved["test"]["micro_f1"]) < 1e-6


def test_notebooks_are_valid_json() -> None:
    nb_names = [
        "01_eda_data_stats.ipynb",
        "02_train_bilstm_crf.ipynb",
        "03_train_phobert.ipynb",
        "04_error_analysis.ipynb",
    ]
    for name in nb_names:
        path = Path("notebooks") / name
        assert path.exists(), f"Missing {path}"
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        assert data.get("nbformat") == 4
        assert len(data.get("cells", [])) >= 3
        assert "<your-org>" not in text
    nb02_text = (Path("notebooks") / "02_train_bilstm_crf.ipynb").read_text(encoding="utf-8")
    assert "NER-PhoNER_COVID19-UET" in nb02_text
    nb03_text = (Path("notebooks") / "03_train_phobert.ipynb").read_text(encoding="utf-8")
    assert "NER-PhoNER_COVID19-UET" in nb03_text
    assert "tin-bilstm-crf" in nb03_text


def test_cli_scripts_run_as_subprocess_without_pythonpath() -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for script in ("scripts/train.py", "scripts/evaluate.py"):
        proc = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, f"{script} failed: {proc.stderr}"


def test_evaluate_uses_checkpoint_config_for_ablation(tmp_path: Path) -> None:
    word_dir = tmp_path / "data" / "word"
    word_dir.mkdir(parents=True)
    sample_lines = (
        '{"words": ["Bệnh_nhân", "19", "ở", "Hà_Nội"], "tags": ["O", "B-PATIENT_ID", "O", "B-LOCATION"]}\n'
        '{"words": ["Bệnh_nhân", "20", "sốt"], "tags": ["O", "B-PATIENT_ID", "B-SYMPTOM_AND_DISEASE"]}\n'
    )
    for split in ("train", "dev", "test"):
        (word_dir / f"{split}_word.json").write_text(sample_lines, encoding="utf-8")

    ckpt_dir = tmp_path / "checkpoints"
    metrics_dir = tmp_path / "metrics"
    ablation_cfg = {
        "experiment_name": "test_bilstm_softmax",
        "seed": 42,
        "data": {"level": "word", "data_dir": str(tmp_path / "data"), "min_word_freq": 1, "max_word_len": 15},
        "model": {
            "name": "bilstm_crf",
            "word_emb_dim": 16,
            "use_char": False,
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.1,
            "use_crf": False,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.01,
            "weight_decay": 0.0,
            "grad_clip": 5.0,
            "patience": 2,
            "checkpoint_dir": str(ckpt_dir),
        },
        "logging": {"use_wandb": False, "wandb_project": "test", "output_dir": str(metrics_dir)},
    }
    summary = run_training(ablation_cfg)
    best_ckpt = ckpt_dir / "best_model.pt"

    metrics_no_cfg = run_evaluation(None, checkpoint_path=best_ckpt, split="test")
    assert abs(metrics_no_cfg["micro_f1"] - summary["test"]["micro_f1"]) < 1e-6

    mismatched_cfg = {
        **ablation_cfg,
        "model": {**ablation_cfg["model"], "use_char": True, "use_crf": True},
    }
    metrics_mismatched = run_evaluation(mismatched_cfg, checkpoint_path=best_ckpt, split="test")
    assert abs(metrics_mismatched["micro_f1"] - summary["test"]["micro_f1"]) < 1e-6


def test_end_to_end_phobert_train_and_evaluate(tmp_path: Path, monkeypatch) -> None:
    import torch
    from transformers import RobertaConfig, RobertaModel
    from src.training.trainer import Trainer, build_model_and_dataloaders
    from src.utils.config import load_config

    word_dir = tmp_path / "data" / "word"
    word_dir.mkdir(parents=True)
    sample_lines = (
        '{"words": ["Bệnh_nhân", "19", "ở", "Hà_Nội"], "tags": ["O", "B-PATIENT_ID", "O", "B-LOCATION"]}\n'
        '{"words": ["Bệnh_nhân", "20", "sốt"], "tags": ["O", "B-PATIENT_ID", "B-SYMPTOM_AND_DISEASE"]}\n'
    )
    for split in ("train", "dev", "test"):
        (word_dir / f"{split}_word.json").write_text(sample_lines, encoding="utf-8")

    class MockPhoBERTTokenizer:
        cls_token_id = 0
        pad_token_id = 1
        sep_token_id = 2
        unk_token_id = 3

        def encode(self, word: str, add_special_tokens: bool = False) -> list[int]:
            return [10, 11] if "_" in word else [10]

    def _make_tiny_backbone(*args, **kwargs):
        return RobertaModel(
            RobertaConfig(
                vocab_size=100,
                hidden_size=16,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=32,
            )
        )

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: MockPhoBERTTokenizer())
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", _make_tiny_backbone)
    monkeypatch.setattr("src.models.phobert_ner.AutoModel.from_pretrained", _make_tiny_backbone)

    ckpt_dir = tmp_path / "checkpoints"
    metrics_dir = tmp_path / "metrics"
    cfg = load_config(
        "configs/phobert.yaml",
        overrides=[
            "experiment_name=test_phobert_crf_e2e",
            f"data.data_dir={tmp_path / 'data'}",
            "model.use_crf=true",
            "model.feature_mode=concat_last_4",
            "training.batch_size=2",
            "training.epochs=1",
            f"training.checkpoint_dir={ckpt_dir}",
            "logging.use_wandb=false",
            f"logging.output_dir={metrics_dir}",
        ],
    )

    summary = run_training(cfg)
    best_ckpt = ckpt_dir / "best_model.pt"
    out_json = metrics_dir / "test_phobert_crf_e2e.json"
    assert best_ckpt.exists()
    assert out_json.exists()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert abs(summary["test"]["micro_f1"] - saved["test"]["micro_f1"]) < 1e-6

    eval_no_cfg = run_evaluation(None, checkpoint_path=best_ckpt, split="test")
    assert abs(eval_no_cfg["micro_f1"] - saved["test"]["micro_f1"]) < 1e-6

    base_cfg = load_config("configs/phobert.yaml")
    base_cfg["data"]["data_dir"] = str(tmp_path / "data")
    eval_with_base_cfg = run_evaluation(base_cfg, checkpoint_path=best_ckpt, split="test")
    assert abs(eval_with_base_cfg["micro_f1"] - saved["test"]["micro_f1"]) < 1e-6

    # Verify Trainer.train_epoch passes decode=False to PhoBERTForNER (skipping CRF decode)
    records = [
        {"words": ["Bệnh_nhân", "19", "ở", "Hà_Nội"], "tags": ["O", "B-PATIENT_ID", "O", "B-LOCATION"]},
    ]
    model, loaders, _ = build_model_and_dataloaders(cfg, {"train": records})
    trainer = Trainer(model=model, config=cfg, device=torch.device("cpu"))

    def _fail_decode(*args, **kwargs):
        raise AssertionError("CRF decode should not run during Trainer.train_epoch")

    monkeypatch.setattr(model.crf, "decode", _fail_decode)
    train_loss = trainer.train_epoch(loaders["train"])
    assert train_loss >= 0.0
