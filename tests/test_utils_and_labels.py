from pathlib import Path
from src.data.download import load_jsonl, load_phoner_split
from src.data.labels import ENTITY_TYPES, ID2LABEL, IGNORE_INDEX, LABEL2ID, LABELS, decode_tags, encode_tags
from src.utils.config import load_config
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed


def test_labels_schema_and_roundtrip() -> None:
    assert len(ENTITY_TYPES) == 10
    assert list(ENTITY_TYPES) == sorted(ENTITY_TYPES)
    assert len(LABELS) == 21
    assert LABELS[0] == "O"
    assert LABEL2ID["O"] == 0
    assert LABEL2ID["B-AGE"] == 1
    assert LABEL2ID["I-AGE"] == 2
    assert LABEL2ID["I-TRANSPORTATION"] == 20
    assert IGNORE_INDEX == -100

    sample = ["O", "B-PATIENT_ID", "O", "B-LOCATION", "I-LOCATION"]
    assert decode_tags(encode_tags(sample)) == sample
    assert decode_tags([0, -100, 1]) == ["O", "B-AGE"]


def test_load_config_cli_overrides_parse_bool_and_scientific_float() -> None:
    cfg = load_config(
        "configs/bilstm_crf.yaml",
        overrides=["model.use_crf=false", "model.use_char=true", "training.lr=3e-5", "training.epochs=5"],
    )
    assert cfg["model"]["use_crf"] is False
    assert cfg["model"]["use_char"] is True
    assert abs(cfg["training"]["lr"] - 3e-5) < 1e-12
    assert cfg["training"]["epochs"] == 5


def test_load_jsonl_and_split(tmp_path: Path) -> None:
    word_dir = tmp_path / "word"
    word_dir.mkdir(parents=True)
    f = word_dir / "train_word.json"
    f.write_text('{"words": ["Bệnh_nhân", "19"], "tags": ["O", "B-PATIENT_ID"]}\n', encoding="utf-8")

    records = load_phoner_split(tmp_path, level="word", split="train")
    assert len(records) == 1
    assert records[0]["words"] == ["Bệnh_nhân", "19"]
    assert load_jsonl(f) == records


def test_seed_and_offline_logger(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    set_seed(42)
    logger = ExperimentLogger({"experiment_name": "test_run", "logging": {"use_wandb": False, "output_dir": str(tmp_path)}})
    logger.log_metrics({"dev_f1": 0.85}, step=1)
    logger.finish()


def test_logger_passes_wandb_entity_from_config(monkeypatch, tmp_path: Path) -> None:
    import sys
    from types import SimpleNamespace

    captured: dict = {}

    class DummyRun:
        def log(self, metrics, step=None):
            pass

        def finish(self):
            pass

    def fake_init(**kwargs):
        captured.update(kwargs)
        return DummyRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    monkeypatch.setenv("WANDB_API_KEY", "dummy-key")

    for cfg_path in ("configs/bilstm_crf.yaml", "configs/phobert.yaml"):
        cfg = load_config(cfg_path, overrides=[f"logging.output_dir={tmp_path}"])
        assert cfg["logging"]["wandb_entity"] == "NER-UET"

    cfg = load_config("configs/phobert.yaml", overrides=[f"logging.output_dir={tmp_path}"])
    logger = ExperimentLogger(cfg)
    logger.finish()
    assert captured["entity"] == "NER-UET"
    assert captured["project"] == "ner-phoner-covid19-uet"
