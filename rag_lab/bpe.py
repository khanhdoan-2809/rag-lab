"""A small byte-level BPE tokenizer for learning, not a model-compatible tokenizer.

IDs 0..255 represent individual UTF-8 bytes. Each learned merge adds one new ID.
We omit regex pre-tokenization, normalization, and special tokens on purpose.
"""

from collections import Counter
from collections.abc import Sequence

Merge = tuple[int, int]


def _merge_pair(tokens: list[int], pair: Merge, new_id: int) -> list[int]:
    """Replace all non-overlapping occurrences of a pair, left to right."""
    merged = []
    position = 0
    while position < len(tokens):
        if position + 1 < len(tokens) and (
            tokens[position], tokens[position + 1]
        ) == pair:
            merged.append(new_id)
            position += 2
        else:
            merged.append(tokens[position])
            position += 1
    return merged


def train_bpe(texts: Sequence[str], num_merges: int = 12) -> list[Merge]:
    """Learn up to num_merges rules; stop when no pair occurs at least twice.

    Corpus entries are separate sequences, so pairs never cross entry boundaries.
    Equal-frequency pairs are resolved by their integer IDs for reproducibility.
    """
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative.")
    sequences = [list(text.encode("utf-8")) for text in texts]
    if not any(sequences):
        raise ValueError("Training requires at least one non-empty text.")

    merges = []
    for _ in range(num_merges):
        frequencies: Counter[Merge] = Counter()
        for sequence in sequences:
            frequencies.update(zip(sequence, sequence[1:]))
        if not frequencies:
            break
        pair = min(frequencies, key=lambda item: (-frequencies[item], item))
        if frequencies[pair] < 2:
            break
        new_id = 256 + len(merges)
        sequences = [_merge_pair(sequence, pair, new_id) for sequence in sequences]
        merges.append(pair)
    return merges


def build_vocabulary(merges: Sequence[Merge]) -> dict[int, bytes]:
    """Map each ID to the exact bytes it represents."""
    vocabulary = {byte: bytes([byte]) for byte in range(256)}
    for new_id, (left, right) in enumerate(merges, start=256):
        if left not in vocabulary or right not in vocabulary:
            raise ValueError("Merge rules must refer to earlier token IDs.")
        vocabulary[new_id] = vocabulary[left] + vocabulary[right]
    return vocabulary


def encode(text: str, merges: Sequence[Merge]) -> list[int]:
    """Apply fixed, already-learned rules to new text. This does not train."""
    tokens = list(text.encode("utf-8"))
    for new_id, pair in enumerate(merges, start=256):
        tokens = _merge_pair(tokens, pair, new_id)
    return tokens


def decode(token_ids: Sequence[int], merges: Sequence[Merge]) -> str:
    """Join bytes before UTF-8 decoding: one token may split a character."""
    vocabulary = build_vocabulary(merges)
    try:
        raw = b"".join(vocabulary[token_id] for token_id in token_ids)
    except KeyError as error:
        raise ValueError(f"Unknown token ID: {error.args[0]}") from error
    return raw.decode("utf-8")
