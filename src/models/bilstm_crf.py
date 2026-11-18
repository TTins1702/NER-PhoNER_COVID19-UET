from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.labels import IGNORE_INDEX
from src.models.char_encoder import CharEncoder
from src.models.crf import LinearChainCRF


class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF sequence tagger with optional character encoder and CRF/Softmax head."""

    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_tags: int = 21,
        word_emb_dim: int = 100,
        use_char: bool = True,
        char_encoder_type: str = "cnn",
        char_emb_dim: int = 30,
        char_out_dim: int = 50,
        char_kernel_size: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_crf: bool = True,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.use_char = use_char
        self.use_crf = use_crf
        self.pad_idx = pad_idx

        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=pad_idx)
        if use_char:
            self.char_encoder: CharEncoder | None = CharEncoder(
                num_chars=char_vocab_size,
                char_emb_dim=char_emb_dim,
                char_out_dim=char_out_dim,
                encoder_type=char_encoder_type,
                kernel_size=char_kernel_size,
                dropout=dropout,
                pad_idx=pad_idx,
            )
            lstm_input_dim = word_emb_dim + char_out_dim
        else:
            self.char_encoder = None
            lstm_input_dim = word_emb_dim

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(2 * hidden_dim, num_tags)

        if use_crf:
            self.crf: LinearChainCRF | None = LinearChainCRF(num_tags)
            self.loss_fn = None
        else:
            self.crf = None
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        tags: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute emissions, optional loss, and decoded tag ID sequences."""
        if mask is None:
            mask = word_ids != self.pad_idx

        word_repr = self.word_emb(word_ids)
        if self.use_char:
            if char_ids is None:
                raise ValueError("char_ids must be provided when use_char=True")
            assert self.char_encoder is not None
            char_repr = self.char_encoder(char_ids)
            token_repr = torch.cat([word_repr, char_repr], dim=-1)
        else:
            token_repr = word_repr

        lengths = mask.long().sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            self.dropout(token_repr),
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=word_ids.size(1),
        )
        emissions = self.classifier(self.dropout(lstm_out))

        if self.use_crf:
            assert self.crf is not None
            loss = self.crf(emissions, tags, mask=mask) if tags is not None else None
            predictions = self.crf.decode(emissions, mask=mask)
        else:
            assert self.loss_fn is not None
            loss = (
                self.loss_fn(emissions.view(-1, self.num_tags), tags.view(-1))
                if tags is not None
                else None
            )
            pred_ids = emissions.argmax(dim=-1)
            seq_lengths = mask.long().sum(dim=1).tolist()
            predictions = [pred_ids[b, :length].tolist() for b, length in enumerate(seq_lengths)]

        return {
            "loss": loss,
            "emissions": emissions,
            "predictions": predictions,
        }
