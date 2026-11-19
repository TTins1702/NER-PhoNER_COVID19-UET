import torch
import torch.nn as nn
from transformers import AutoModel


class PhoBERTNER(nn.Module):
    def __init__(
        self,
        num_tags=21,
        model_name="vinai/phobert-base-v2",
        model_name_or_path=None,
        feature_mode="last_layer",
        use_crf=False,
        freeze_backbone=False,
        dropout=0.1,
        ignore_index=-100,
        backbone=None,
        crf_layer=None,
    ):
        super().__init__()
        if model_name_or_path is not None:
            model_name = model_name_or_path
        self.feature_mode = feature_mode
        self.use_crf = use_crf
        self.backbone = (
            backbone if backbone is not None else AutoModel.from_pretrained(model_name)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        feat_dim = hidden_size * 4 if feature_mode == "concat_last_4" else hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        if use_crf:
            if crf_layer is not None:
                self.crf = crf_layer
            else:
                from src.models.crf import LinearChainCRF

                self.crf = LinearChainCRF(num_tags)
        else:
            self.crf = None

    def _extract_features(self, input_ids, attention_mask):
        if self.feature_mode == "concat_last_4":
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return torch.cat(outputs.hidden_states[-4:], dim=-1)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def _gather_word_features(self, features, word_mask, labels=None):
        batch_size = features.size(0)
        feat_dim = features.size(-1)
        device = features.device

        word_feats_list = [features[i][word_mask[i]] for i in range(batch_size)]
        max_words = max(1, max(w.size(0) for w in word_feats_list))

        padded_feats = torch.zeros(
            (batch_size, max_words, feat_dim), dtype=features.dtype, device=device
        )
        word_level_mask = torch.zeros(
            (batch_size, max_words), dtype=torch.bool, device=device
        )
        word_labels = (
            torch.zeros((batch_size, max_words), dtype=torch.long, device=device)
            if labels is not None
            else None
        )

        for i, feats in enumerate(word_feats_list):
            num_words = feats.size(0)
            if num_words > 0:
                padded_feats[i, :num_words] = feats
                word_level_mask[i, :num_words] = True
                if labels is not None:
                    word_labels[i, :num_words] = labels[i][word_mask[i]]

        return padded_feats, word_level_mask, word_labels

    def forward(
        self, input_ids, attention_mask, word_mask=None, labels=None, decode=True
    ):
        features = self._extract_features(input_ids, attention_mask)

        if self.use_crf:
            word_feats, word_level_mask, word_labels = self._gather_word_features(
                features, word_mask, labels
            )
            emissions = self.classifier(self.dropout(word_feats))
            result = {"emissions": emissions, "word_level_mask": word_level_mask}
            if labels is not None:
                result["loss"] = self.crf(emissions, word_labels, word_level_mask)
            if decode:
                result["predictions"] = self.crf.decode(emissions, word_level_mask)
            return result

        logits = self.classifier(self.dropout(features))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        if decode and word_mask is not None:
            pred_ids = logits.argmax(dim=-1)
            result["predictions"] = [
                pred_ids[i][word_mask[i]].tolist() for i in range(pred_ids.size(0))
            ]
        return result

    def decode(self, input_ids, attention_mask, word_mask):
        out = self.forward(
            input_ids, attention_mask, word_mask=word_mask, decode=True
        )
        return out["predictions"]

    def build_optimizer_params(
        self, backbone_lr=2e-5, head_lr=1e-3, weight_decay=0.01
    ):
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        head_params = [
            p
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("backbone.")
        ]

        groups = []
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                }
            )
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                }
            )
        return groups


PhoBERTForNER = PhoBERTNER
