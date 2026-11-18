from src.data.bilstm_dataset import BiLSTMDataset, bilstm_collate_fn
from src.data.labels import IGNORE_INDEX
from src.data.vocab import PAD_IDX, UNK_IDX, CharVocab, WordVocab


def test_vocab_min_freq_and_unseen_tokens_map_to_unk() -> None:
    train_sents = [["bệnh_nhân", "sốt", "bệnh_nhân"], ["ho"]]
    w_vocab = WordVocab.build(train_sents, min_freq=2)
    assert w_vocab.encode(["bệnh_nhân", "sốt", "OOV_WORD"]) == [2, UNK_IDX, UNK_IDX]
    assert PAD_IDX == 0 and UNK_IDX == 1

    c_vocab = CharVocab.build(train_sents)
    encoded_chars = c_vocab.encode_word("ho🔥", max_len=2)
    assert len(encoded_chars) == 2
    assert c_vocab.encode_word("🔥") == [UNK_IDX]

    assert WordVocab.from_dict(w_vocab.to_dict()).encode(["bệnh_nhân", "OOV"]) == [2, UNK_IDX]
    assert CharVocab.from_dict(c_vocab.to_dict()).encode_word("h") == c_vocab.encode_word("h")


def test_bilstm_dataset_and_collate_shapes() -> None:
    samples = [
        {"words": ["BN19", "ở", "Hà_Nội"], "tags": ["B-PATIENT_ID", "O", "B-LOCATION"]},
        {"words": ["Sốt"], "tags": ["B-SYMPTOM_AND_DISEASE"]},
    ]
    w_vocab = WordVocab.build([s["words"] for s in samples], min_freq=1)
    c_vocab = CharVocab.build([s["words"] for s in samples])
    ds = BiLSTMDataset(samples, w_vocab, c_vocab, max_word_len=10)
    batch = bilstm_collate_fn([ds[0], ds[1]])

    assert batch["word_ids"].shape == (2, 3)
    assert batch["char_ids"].shape == (2, 3, 6)  # len("Hà_Nội") == 6
    assert batch["mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["tags"][1].tolist() == [batch["tags"][1, 0].item(), IGNORE_INDEX, IGNORE_INDEX]
    assert (batch["char_ids"][1, 1:] == PAD_IDX).all()
