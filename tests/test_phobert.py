import json
import os
import tempfile
import unittest
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

from src.data.phobert_dataset import (
    PhoBERTDataset,
    decode_word_level_predictions,
    extract_word_level_labels,
    get_label_maps,
    inspect_bpe_sample,
    phobert_collate_fn,
)
from src.models.phobert_ner import PhoBERTForNER, PhoBERTNER


class MockTokenizer:
    cls_token_id = 0
    pad_token_id = 1
    sep_token_id = 2
    unk_token_id = 3

    def tokenize(self, word):
        if "_" in word:
            return [p for p in word.split("_") if p]
        return [word]

    def encode(self, word, add_special_tokens=False):
        if "_" in word:
            return [10, 11]
        return [10]


class DummyCRF(nn.Module):
    def __init__(self, num_tags=21):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, emissions, tags, mask):
        return (emissions * mask.unsqueeze(-1)).pow(2).mean() * self.scale

    def decode(self, emissions, mask):
        preds = emissions.argmax(dim=-1)
        return [preds[i][mask[i]].tolist() for i in range(preds.size(0))]


def make_tiny_backbone():
    cfg = RobertaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
    )
    return RobertaModel(cfg)


class TestPhoBERTPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tokenizer = MockTokenizer()
        self.label2id, self.id2label = get_label_maps()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write_jsonl(self, name, rows):
        path = os.path.join(self.tmp_dir.name, name)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_first_subword_alignment(self):
        bpe_pairs = inspect_bpe_sample(self.tokenizer, ["Bệnh_nhân", "ở", "Vĩnh_Phúc"])
        self.assertEqual(bpe_pairs[0][1], ["Bệnh", "nhân"])

        path = self._write_jsonl(
            "sample.jsonl",
            [
                {
                    "words": ["Bệnh_nhân", "ở", "Vĩnh_Phúc"],
                    "tags": ["O", "O", "B-LOCATION"],
                }
            ],
        )
        ds = PhoBERTDataset(path, self.tokenizer, label2id=self.label2id)
        item = ds[0]

        self.assertEqual(item["input_ids"].tolist(), [0, 10, 11, 10, 10, 11, 2])
        self.assertEqual(
            item["word_mask"].tolist(), [False, True, False, True, True, False, False]
        )
        self.assertEqual(
            item["labels"].tolist(),
            [
                -100,
                self.label2id["O"],
                -100,
                self.label2id["O"],
                self.label2id["B-LOCATION"],
                -100,
                -100,
            ],
        )

        pred_tensor = item["labels"].clone().unsqueeze(0)
        pred_tensor[pred_tensor < 0] = 0
        decoded = decode_word_level_predictions(
            pred_tensor, item["word_mask"].unsqueeze(0), self.id2label
        )
        self.assertEqual(decoded[0], ["O", "O", "B-LOCATION"])
        self.assertEqual(len(decoded[0]), 3)

    @unittest.skipUnless(
        os.path.exists("data/word/dev_word.json"),
        "PhoNER_COVID19 dataset not downloaded locally",
    )
    def test_real_dev_dataset_and_collate(self):
        ds = PhoBERTDataset("data/word/dev_word.json", self.tokenizer, label2id=self.label2id)
        batch = phobert_collate_fn([ds[0], ds[1]], pad_token_id=self.tokenizer.pad_token_id)

        self.assertEqual(batch["input_ids"].shape, batch["attention_mask"].shape)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertEqual(batch["input_ids"].shape, batch["word_mask"].shape)
        self.assertEqual(
            int(batch["word_mask"][0].sum().item()), len(ds.samples[0]["words"])
        )
        self.assertEqual(
            int(batch["word_mask"][1].sum().item()), len(ds.samples[1]["words"])
        )

        gold = extract_word_level_labels(
            batch["labels"], batch["word_mask"], self.id2label
        )
        self.assertEqual(gold[0], ds.samples[0]["tags"])
        self.assertEqual(gold[1], ds.samples[1]["tags"])

    def test_truncation_and_stats(self):
        path = self._write_jsonl(
            "long.jsonl",
            [
                {"words": ["a_b"] * 10, "tags": ["O"] * 10},
                {"words": ["x", "y"], "tags": ["O", "O"]},
            ],
        )
        ds = PhoBERTDataset(
            path, self.tokenizer, label2id=self.label2id, max_seq_length=12
        )
        item = ds[0]

        self.assertLessEqual(item["input_ids"].size(0), 12)
        self.assertEqual(int(item["input_ids"][0].item()), self.tokenizer.cls_token_id)
        self.assertEqual(int(item["input_ids"][-1].item()), self.tokenizer.sep_token_id)
        self.assertEqual(int(item["word_mask"].sum().item()), 5)
        self.assertEqual(item["orig_length"], 10)
        self.assertEqual(len(item["words"]), 10)
        self.assertEqual(len(item["gold_tags"]), 10)
        self.assertEqual(ds.get_truncation_stats(), {"total": 2, "truncated": 1})

        decoded = decode_word_level_predictions(
            [[self.label2id["B-LOCATION"]] * 5],
            [item["orig_length"]],
            self.id2label,
        )
        self.assertEqual(decoded, [["B-LOCATION"] * 5 + ["O"] * 5])

    def test_phobert_softmax_last_and_concat4(self):
        path = self._write_jsonl(
            "batch.jsonl",
            [
                {"words": ["Bệnh_nhân", "A", "B"], "tags": ["O", "B-NAME", "I-NAME"]},
                {"words": ["Hà_Nội", "nay"], "tags": ["B-LOCATION", "O"]},
            ],
        )
        ds = PhoBERTDataset(path, self.tokenizer, label2id=self.label2id)
        batch = phobert_collate_fn([ds[0], ds[1]], pad_token_id=self.tokenizer.pad_token_id)

        for mode in ("last_layer", "concat_last_4"):
            model = PhoBERTNER(
                num_tags=len(self.label2id),
                feature_mode=mode,
                use_crf=False,
                backbone=make_tiny_backbone(),
            )
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["word_mask"],
                batch["labels"],
            )
            self.assertEqual(out["loss"].ndim, 0)
            out["loss"].backward()

            preds = model.decode(
                batch["input_ids"], batch["attention_mask"], batch["word_mask"]
            )
            self.assertEqual([len(p) for p in preds], [3, 2])

    def test_phobert_freeze_and_differential_lr(self):
        frozen_model = PhoBERTNER(
            num_tags=len(self.label2id),
            freeze_backbone=True,
            backbone=make_tiny_backbone(),
        )
        self.assertTrue(
            all(not p.requires_grad for p in frozen_model.backbone.parameters())
        )
        self.assertTrue(
            all(p.requires_grad for p in frozen_model.classifier.parameters())
        )
        frozen_groups = frozen_model.build_optimizer_params(
            backbone_lr=2e-5, head_lr=1e-3
        )
        self.assertEqual(len(frozen_groups), 1)
        self.assertEqual(frozen_groups[0]["lr"], 1e-3)

        full_model = PhoBERTNER(
            num_tags=len(self.label2id),
            freeze_backbone=False,
            backbone=make_tiny_backbone(),
        )
        full_groups = full_model.build_optimizer_params(backbone_lr=2e-5, head_lr=1e-3)
        self.assertEqual(len(full_groups), 2)
        self.assertEqual([g["lr"] for g in full_groups], [2e-5, 1e-3])

    def test_phobert_crf_head(self):
        path = self._write_jsonl(
            "crf_batch.jsonl",
            [
                {"words": ["Bệnh_nhân", "A", "B"], "tags": ["O", "B-NAME", "I-NAME"]},
                {"words": ["Hà_Nội", "nay"], "tags": ["B-LOCATION", "O"]},
            ],
        )
        ds = PhoBERTDataset(path, self.tokenizer, label2id=self.label2id)
        batch = phobert_collate_fn([ds[0], ds[1]], pad_token_id=self.tokenizer.pad_token_id)

        model = PhoBERTNER(
            num_tags=len(self.label2id),
            use_crf=True,
            backbone=make_tiny_backbone(),
            crf_layer=DummyCRF(len(self.label2id)),
        )
        out = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_mask"],
            batch["labels"],
        )
        self.assertEqual(out["emissions"].shape, (2, 3, len(self.label2id)))
        self.assertEqual(out["loss"].ndim, 0)
        out["loss"].backward()

        preds = model.decode(
            batch["input_ids"], batch["attention_mask"], batch["word_mask"]
        )
        self.assertEqual([len(p) for p in preds], [3, 2])

    def test_trainer_compatibility_contract(self):
        records = [
            {"words": ["Bệnh_nhân", "A", "B"], "tags": ["O", "B-NAME", "I-NAME"]},
            {"words": ["Hà_Nội", "nay"], "tags": ["B-LOCATION", "O"]},
        ]
        ds = PhoBERTDataset(records, tokenizer=self.tokenizer, max_seq_length=32)
        batch = phobert_collate_fn([ds[0], ds[1]], pad_token_id=self.tokenizer.pad_token_id)

        self.assertEqual(batch["orig_lengths"], [3, 2])
        self.assertEqual(batch["gold_tags"], [["O", "B-NAME", "I-NAME"], ["B-LOCATION", "O"]])

        model = PhoBERTForNER(
            model_name_or_path="vinai/phobert-base-v2",
            num_tags=len(self.label2id),
            backbone=make_tiny_backbone(),
        )
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_mask=batch["word_mask"],
            labels=batch["labels"],
        )
        decoded = decode_word_level_predictions(out["predictions"], batch["orig_lengths"])
        self.assertEqual([len(seq) for seq in decoded], [3, 2])

    def test_tagless_inference_dataset_and_forward(self):
        ds = PhoBERTDataset(
            [{"words": ["Bệnh_nhân", "ở", "Hà_Nội"]}],
            tokenizer=self.tokenizer,
            max_seq_length=32,
        )
        batch = phobert_collate_fn([ds[0]], pad_token_id=self.tokenizer.pad_token_id)
        self.assertIsNone(batch["labels"])
        self.assertEqual(batch["gold_tags"], [[]])
        self.assertEqual(batch["orig_lengths"], [3])

        model = PhoBERTForNER(
            num_tags=len(self.label2id),
            use_crf=False,
            backbone=make_tiny_backbone(),
        )
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_mask=batch["word_mask"],
            labels=batch["labels"],
        )
        self.assertIsNone(out["loss"])
        self.assertEqual(len(out["predictions"][0]), 3)

    def test_phobert_real_linear_chain_crf_and_decode_false(self):
        records = [
            {"words": ["Bệnh_nhân", "A", "B"], "tags": ["O", "B-NAME", "I-NAME"]},
            {"words": ["Hà_Nội", "nay"], "tags": ["B-LOCATION", "O"]},
        ]
        ds = PhoBERTDataset(records, tokenizer=self.tokenizer, max_seq_length=32)
        batch = phobert_collate_fn([ds[0], ds[1]], pad_token_id=self.tokenizer.pad_token_id)

        model = PhoBERTForNER(
            num_tags=len(self.label2id),
            use_crf=True,
            backbone=make_tiny_backbone(),
        )
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_mask=batch["word_mask"],
            labels=batch["labels"],
            decode=True,
        )
        self.assertIsNotNone(out["loss"])
        out["loss"].backward()
        self.assertEqual([len(p) for p in out["predictions"]], [3, 2])

        def _fail_decode(*args, **kwargs):
            raise AssertionError("CRF decode should not run when decode=False")

        orig_decode = model.crf.decode
        model.crf.decode = _fail_decode
        try:
            out_no_decode = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                word_mask=batch["word_mask"],
                labels=batch["labels"],
                decode=False,
            )
            self.assertIsNotNone(out_no_decode["loss"])
            self.assertEqual(out_no_decode["predictions"], [])
        finally:
            model.crf.decode = orig_decode


if __name__ == "__main__":
    unittest.main()
