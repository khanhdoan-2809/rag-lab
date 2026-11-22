"""Verify learned rules, fixed inference, byte preservation, and boundaries."""

import pytest

from rag_lab.bpe import build_vocabulary, decode, encode, train_bpe


def test_training_learns_known_pairs_in_order() -> None:
    assert train_bpe(["abab", "abab"], 2) == [(97, 98), (256, 256)]


def test_vocabulary_keeps_all_bytes_and_adds_merged_bytes() -> None:
    vocabulary = build_vocabulary([(97, 98), (256, 256)])
    assert len(vocabulary) == 258
    assert vocabulary[0] == b"\x00"
    assert vocabulary[255] == b"\xff"
    assert vocabulary[256] == b"ab"
    assert vocabulary[257] == b"abab"


def test_inference_applies_existing_rules_to_unseen_text() -> None:
    merges = train_bpe(["abab", "abab"], 2)
    assert encode("abab", merges) == [257]
    assert encode("abx", merges) == [256, 120]
    assert merges == [(97, 98), (256, 256)]


def test_overlapping_pairs_are_merged_without_reusing_bytes() -> None:
    merges = train_bpe(["aaaaa"], 1)
    assert encode("aaaaa", merges) == [256, 256, 97]


def test_pair_frequency_counts_repeated_training_examples() -> None:
    assert train_bpe(["ab", "ab", "xy"], 1) == [(97, 98)]


def test_frequency_ties_are_deterministic() -> None:
    assert train_bpe(["cdcd", "abab"], 1) == [(97, 98)]
    assert train_bpe(["abab", "cdcd"], 1) == [(97, 98)]


def test_pairs_do_not_cross_training_entry_boundaries() -> None:
    assert train_bpe(["a", "b", "a", "b"], 10) == []


def test_unique_pairs_do_not_create_merge_rules() -> None:
    assert train_bpe(["ab"], 20) == []


def test_training_stops_when_no_pairs_remain() -> None:
    assert train_bpe(["ab", "ab"], 100) == [(97, 98)]


def test_zero_merges_leaves_byte_tokens() -> None:
    merges = train_bpe(["backup backup"], 0)
    assert merges == []
    assert encode("backup", merges) == list(b"backup")


@pytest.mark.parametrize(
    "text",
    ["", "database backups", "  a\t b\n", "caf\u00e9", "\U0001f642",
     "\u4e2d\u6587", "unknown words: xyzzy", "<|endoftext|>", "\x00"],
)
def test_round_trip_preserves_exact_text(text) -> None:
    merges = train_bpe(["backup backup backups", "database database"], 12)
    assert decode(encode(text, merges), merges) == text


def test_byte_tokens_can_split_a_unicode_character() -> None:
    assert encode("\u00e9", []) == [195, 169]
    assert decode([195, 169], []) == "\u00e9"
    with pytest.raises(UnicodeDecodeError):
        decode([195], [])


@pytest.mark.parametrize("texts", [[], [""], ["", ""]])
def test_empty_training_corpus_is_rejected(texts) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        train_bpe(texts)


def test_negative_merge_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        train_bpe(["backup"], -1)


def test_merge_rules_cannot_reference_future_tokens() -> None:
    with pytest.raises(ValueError, match="earlier"):
        build_vocabulary([(256, 97)])


@pytest.mark.parametrize("token_id", [-1, 256, 9999])
def test_unknown_token_ids_are_rejected(token_id) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        decode([token_id], [])


def test_more_merges_compress_this_repeated_example() -> None:
    one = train_bpe(["abab", "abab"], 1)
    two = train_bpe(["abab", "abab"], 2)
    assert len(encode("abab", one)) == 2
    assert len(encode("abab", two)) == 1
