import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class CharEncoder(nn.Module):
    """Character-level word encoder supporting 1D-CNN and bidirectional LSTM."""

    def __init__(
        self,
        num_chars: int,
        char_emb_dim: int = 30,
        char_out_dim: int = 50,
        encoder_type: str = "cnn",
        kernel_size: int = 3,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if encoder_type not in ("cnn", "bilstm"):
            raise ValueError(f"Unsupported encoder_type {encoder_type!r}; expected 'cnn' or 'bilstm'")
        if encoder_type == "bilstm" and char_out_dim % 2 != 0:
            raise ValueError(f"char_out_dim must be even for BiLSTM char encoder, got {char_out_dim}")

        self.encoder_type = encoder_type
        self.char_out_dim = char_out_dim
        self.pad_idx = pad_idx
        self.char_emb = nn.Embedding(num_chars, char_emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        if encoder_type == "cnn":
            self.conv = nn.Conv1d(
                in_channels=char_emb_dim,
                out_channels=char_out_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.char_lstm = None
        else:
            self.conv = None
            self.char_lstm = nn.LSTM(
                input_size=char_emb_dim,
                hidden_size=char_out_dim // 2,
                bidirectional=True,
                batch_first=True,
            )

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Encode character IDs of shape [B, L, W] into [B, L, char_out_dim]."""
        batch_size, seq_len, word_len = char_ids.shape
        valid_word_mask = (char_ids != self.pad_idx).any(dim=-1)  # [B, L]

        emb = self.dropout(self.char_emb(char_ids))  # [B, L, W, E]
        emb_flat = emb.view(batch_size * seq_len, word_len, -1)

        if self.encoder_type == "cnn":
            assert self.conv is not None
            conv_in = emb_flat.transpose(1, 2)  # [B*L, E, W]
            conv_out = F.relu(self.conv(conv_in))
            pooled = conv_out.max(dim=-1).values
        else:
            assert self.char_lstm is not None
            char_lengths = (char_ids != self.pad_idx).sum(dim=-1).view(-1)
            clamped_lengths = char_lengths.clamp(min=1).cpu()
            packed = pack_padded_sequence(
                emb_flat,
                clamped_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hn, _) = self.char_lstm(packed)
            pooled = torch.cat([hn[-2], hn[-1]], dim=-1)

        out = pooled.view(batch_size, seq_len, self.char_out_dim)
        return out * valid_word_mask.unsqueeze(-1).to(out.dtype)
