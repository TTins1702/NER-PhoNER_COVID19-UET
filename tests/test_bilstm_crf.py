import pytest
import torch
from src.data.labels import IGNORE_INDEX
from src.models.bilstm_crf import BiLSTMCRF
from src.models.char_encoder import CharEncoder


@pytest.mark.parametrize("encoder_type", ["cnn", "bilstm"])
def test_char_encoder_single_char_words_and_padded_words(encoder_type: str) -> None:
    encoder = CharEncoder(
        num_chars=20,
        char_emb_dim=16,
        char_out_dim=24,
        encoder_type=encoder_type,
        kernel_size=3,
    )
    # Batch 2 câu, độ dài câu L=3, độ dài từ tối đa W=1; câu thứ 2 có 2 vị trí từ <pad> (toàn số 0)
    char_ids = torch.tensor([[[2], [3], [4]], [[5], [0], [0]]], dtype=torch.long)
    out = encoder(char_ids)
    assert out.shape == (2, 3, 24)
    assert torch.isfinite(out).all()
    assert (out[1, 1:] == 0).all()


@pytest.mark.parametrize("use_char,use_crf", [(True, True), (False, True), (True, False)])
def test_bilstm_crf_ablation_modes_forward_and_decode(use_char: bool, use_crf: bool) -> None:
    torch.manual_seed(42)
    model = BiLSTMCRF(
        vocab_size=50,
        char_vocab_size=30,
        num_tags=21,
        word_emb_dim=32,
        use_char=use_char,
        char_emb_dim=12,
        char_out_dim=16,
        hidden_dim=32,
        use_crf=use_crf,
    )
    word_ids = torch.tensor([[2, 3, 4], [5, 0, 0]], dtype=torch.long)
    char_ids = torch.tensor(
        [[[2, 3], [4, 0], [5, 6]], [[7, 0], [0, 0], [0, 0]]],
        dtype=torch.long,
    )
    mask = torch.tensor([[True, True, True], [True, False, False]], dtype=torch.bool)
    tags = torch.tensor([[0, 1, 2], [3, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)

    result = model(word_ids=word_ids, char_ids=char_ids, mask=mask, tags=tags)
    assert result["loss"] is not None and result["loss"].item() >= 0
    result["loss"].backward()
    assert result["emissions"].shape == (2, 3, 21)
    assert [len(p) for p in result["predictions"]] == [3, 1]


def test_bilstm_crf_decode_false_skips_viterbi_decoding(monkeypatch) -> None:
    model = BiLSTMCRF(
        vocab_size=30,
        char_vocab_size=20,
        num_tags=21,
        word_emb_dim=16,
        char_emb_dim=8,
        char_out_dim=12,
        hidden_dim=16,
        use_crf=True,
    )
    assert model.crf is not None

    def _fail_decode(*args, **kwargs):
        raise AssertionError("CRF decode should not run when decode=False")

    monkeypatch.setattr(model.crf, "decode", _fail_decode)

    word_ids = torch.tensor([[2, 3]], dtype=torch.long)
    char_ids = torch.tensor([[[2, 3], [4, 0]]], dtype=torch.long)
    mask = torch.tensor([[True, True]], dtype=torch.bool)
    tags = torch.tensor([[0, 1]], dtype=torch.long)

    out = model(word_ids=word_ids, char_ids=char_ids, mask=mask, tags=tags, decode=False)
    assert out["loss"] is not None
    assert out["predictions"] == []
