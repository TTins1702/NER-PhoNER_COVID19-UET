import itertools
import pytest
import torch

from src.models.crf import LinearChainCRF, log_sum_exp


def test_log_sum_exp_numerical_stability() -> None:
    x_large = torch.tensor([[1000.0, 1001.0, 999.0]])
    x_small = torch.tensor([[-1000.0, -1001.0, -999.0]])

    for x in (x_large, x_small):
        actual = log_sum_exp(x, dim=-1, keepdim=False)
        expected = torch.logsumexp(x, dim=-1, keepdim=False)
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual, expected, atol=1e-6)

        actual_kd = log_sum_exp(x, dim=-1, keepdim=True)
        expected_kd = torch.logsumexp(x, dim=-1, keepdim=True)
        assert actual_kd.shape == (1, 1)
        assert torch.allclose(actual_kd, expected_kd, atol=1e-6)


def test_crf_matches_bruteforce() -> None:
    torch.manual_seed(42)
    num_tags = 3
    seq_len = 4
    crf = LinearChainCRF(num_tags=num_tags)
    emissions = torch.randn(1, seq_len, num_tags)
    mask = torch.ones(1, seq_len, dtype=torch.bool)

    all_scores = []
    best_seq = None
    best_score = float("-inf")

    for seq in itertools.product(range(num_tags), repeat=seq_len):
        tags = torch.tensor([list(seq)], dtype=torch.long)
        score = crf._compute_gold_score(emissions, tags, mask).item()
        all_scores.append(score)
        if score > best_score:
            best_score = score
            best_seq = list(seq)

    expected_log_z = torch.logsumexp(torch.tensor(all_scores), dim=0)
    actual_log_z = crf._compute_log_partition(emissions, mask).squeeze(0)

    assert torch.allclose(actual_log_z, expected_log_z, atol=1e-5)
    assert crf.decode(emissions, mask) == [best_seq]


def test_crf_padding_invariance_and_ignored_pad_tags() -> None:
    torch.manual_seed(7)
    crf = LinearChainCRF(num_tags=5)

    e_short = torch.randn(1, 3, 5)
    t_short = torch.tensor([[1, 2, 0]], dtype=torch.long)
    m_short = torch.tensor([[True, True, True]])

    e_pad = torch.cat([e_short, torch.randn(1, 4, 5) * 100.0], dim=1)
    t_pad = torch.cat([t_short, torch.full((1, 4), -100, dtype=torch.long)], dim=1)
    m_pad = torch.tensor([[True, True, True, False, False, False, False]])

    loss_short = crf(e_short, t_short, m_short, reduction="none")
    loss_pad = crf(e_pad, t_pad, m_pad, reduction="none")

    assert torch.allclose(loss_short, loss_pad, atol=1e-5)
    assert crf.decode(e_short, m_short) == crf.decode(e_pad, m_pad)


def test_crf_reduction_modes() -> None:
    torch.manual_seed(0)
    crf = LinearChainCRF(num_tags=4)
    emissions = torch.randn(2, 3, 4)
    tags = torch.tensor([[0, 1, 2], [3, 1, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    loss_none = crf(emissions, tags, mask, reduction="none")
    assert loss_none.shape == (2,)
    assert (loss_none >= 0).all()

    assert torch.allclose(crf(emissions, tags, mask, reduction="sum"), loss_none.sum())
    assert torch.allclose(crf(emissions, tags, mask, reduction="mean"), loss_none.mean())
    assert torch.allclose(
        crf(emissions, tags, mask, reduction="token_mean"),
        loss_none.sum() / mask.float().sum(),
    )

    with pytest.raises(ValueError):
        crf(emissions, tags, mask, reduction="invalid")


def test_crf_single_token_and_gradient_flow() -> None:
    torch.manual_seed(123)
    crf = LinearChainCRF(num_tags=5)

    e1 = torch.randn(2, 1, 5, requires_grad=True)
    t1 = torch.tensor([[0], [4]], dtype=torch.long)
    loss1 = crf(e1, t1)
    assert loss1.item() >= 0
    loss1.backward()
    assert e1.grad is not None and torch.isfinite(e1.grad).all()
    assert crf.start_transitions.grad is not None
    assert crf.end_transitions.grad is not None
    assert crf.decode(e1.detach()) == [
        [int((e1[0, 0] + crf.start_transitions + crf.end_transitions).argmax().item())],
        [int((e1[1, 0] + crf.start_transitions + crf.end_transitions).argmax().item())],
    ]

    crf.zero_grad()
    e_multi = torch.randn(2, 4, 5, requires_grad=True)
    t_multi = torch.tensor([[0, 1, 2, 3], [4, 0, 0, 0]], dtype=torch.long)
    m_multi = torch.tensor([[True, True, True, True], [True, True, False, False]])
    loss_multi = crf(e_multi, t_multi, m_multi)
    assert loss_multi.item() >= 0
    loss_multi.backward()
    assert e_multi.grad is not None and torch.isfinite(e_multi.grad).all()
    assert crf.transitions.grad is not None and torch.isfinite(crf.transitions.grad).all()


def test_crf_validation_errors() -> None:
    with pytest.raises(ValueError):
        LinearChainCRF(num_tags=0)

    crf = LinearChainCRF(num_tags=4)
    with pytest.raises(ValueError):
        crf.decode(torch.randn(2, 4))
    with pytest.raises(ValueError):
        crf.decode(torch.randn(2, 3, 5))
    with pytest.raises(ValueError):
        crf.decode(torch.randn(2, 0, 4))
    with pytest.raises(ValueError):
        crf.decode(torch.randn(2, 3, 4), mask=torch.tensor([[False, True, True], [True, True, True]]))
    with pytest.raises(ValueError):
        crf(torch.randn(2, 3, 4), tags=torch.zeros(2, 2, dtype=torch.long))
