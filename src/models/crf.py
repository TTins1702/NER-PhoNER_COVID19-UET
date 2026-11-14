from typing import List, Optional

import torch
import torch.nn as nn


def log_sum_exp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Compute numerically stable log-sum-exp along ``dim``."""
    max_score, _ = x.max(dim=dim, keepdim=True)
    out = max_score + torch.log(torch.sum(torch.exp(x - max_score), dim=dim, keepdim=True))
    if not keepdim:
        out = out.squeeze(dim)
    return out


class LinearChainCRF(nn.Module):
    """Linear-chain Conditional Random Field implemented from scratch in PyTorch.

    Convention: ``transitions[i, j]`` is the transition score from tag ``i`` at
    timestep ``t - 1`` to tag ``j`` at timestep ``t``.
    """

    def __init__(self, num_tags: int) -> None:
        if num_tags <= 0:
            raise ValueError(f"num_tags must be > 0, got {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have shape (batch_size, seq_len, num_tags), got {tuple(emissions.shape)}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"Expected emissions last dim {self.num_tags}, got {emissions.size(2)}"
            )
        if emissions.size(1) == 0:
            raise ValueError("Sequence length must be > 0")

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            if mask.shape != emissions.shape[:2]:
                raise ValueError(
                    f"mask must have shape {tuple(emissions.shape[:2])}, got {tuple(mask.shape)}"
                )
            mask = mask.bool()

        if not mask[:, 0].all():
            raise ValueError("First timestep must have mask=True for all sequences in the batch")

        if tags is not None and tags.shape != emissions.shape[:2]:
            raise ValueError(
                f"tags must have shape {tuple(emissions.shape[:2])}, got {tuple(tags.shape)}"
            )
        return mask

    def _compute_gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        mask_float = mask.to(emissions.dtype)
        batch_idx = torch.arange(batch_size, device=emissions.device)

        # Zero out padded tag indices (e.g. -100) before indexing into emissions/transitions.
        safe_tags = tags * mask.long()

        score = (
            self.start_transitions[safe_tags[:, 0]]
            + emissions[batch_idx, 0, safe_tags[:, 0]]
        )

        for t in range(1, seq_len):
            emit_score = emissions[batch_idx, t, safe_tags[:, t]]
            trans_score = self.transitions[safe_tags[:, t - 1], safe_tags[:, t]]
            score = score + (emit_score + trans_score) * mask_float[:, t]

        last_tag_idx = mask.long().sum(dim=1) - 1
        last_tags = safe_tags[batch_idx, last_tag_idx]
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = emissions.size(1)
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        for t in range(1, seq_len):
            # Broadcast [B, C, 1] + [1, C, C] + [B, 1, C] -> [B, C, C]
            next_alpha = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )
            next_alpha = log_sum_exp(next_alpha, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return log_sum_exp(alpha, dim=1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss ``log Z(x) - s(x, y)``."""
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"Invalid reduction mode: {reduction}")

        mask = self._validate(emissions, tags=tags, mask=mask)
        gold_score = self._compute_gold_score(emissions, tags, mask)
        log_partition = self._compute_log_partition(emissions, mask)
        nll = log_partition - gold_score

        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            return nll.mean()
        return nll.sum() / mask.to(emissions.dtype).sum()

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Decode optimal tag sequences using the Viterbi algorithm."""
        mask = self._validate(emissions, mask=mask)
        batch_size, seq_len, _ = emissions.shape

        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history: List[torch.Tensor] = []

        for t in range(1, seq_len):
            cand = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_prev_score, best_prev_tag = cand.max(dim=1)
            next_score = best_prev_score + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
            history.append(best_prev_tag)

        seq_lengths = mask.long().sum(dim=1)
        best_paths: List[List[int]] = []

        for b in range(batch_size):
            seq_len_b = int(seq_lengths[b].item())
            best_tag = int((score[b] + self.end_transitions).argmax(dim=0).item())
            best_tags = [best_tag]

            for backpointer in reversed(history[: seq_len_b - 1]):
                best_tag = int(backpointer[b, best_tag].item())
                best_tags.append(best_tag)

            best_tags.reverse()
            best_paths.append(best_tags)

        return best_paths
