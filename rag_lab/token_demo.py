"""Lesson 2: inspect a toy BPE tokenizer and optionally compare with tiktoken."""

import argparse
import os
from pathlib import Path

from rag_lab.bpe import build_vocabulary, decode, encode, train_bpe

# A fixed, tiny training corpus. Inference text does not change these merge rules.
TRAINING_TEXTS = [
    "backup backup backups",
    "database database backups",
    "restore database backup",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def show_openai_tokens(text: str) -> None:
    """Tokenize locally; first use may download public encoding data."""
    import tiktoken

    # Keep downloaded vocabulary data in the requested project folder.
    os.environ["TIKTOKEN_CACHE_DIR"] = str(PROJECT_ROOT / ".cache" / "tiktoken")
    encoding = tiktoken.get_encoding("cl100k_base")
    token_ids = encoding.encode_ordinary(text)
    print()
    print("Production comparison: cl100k_base")
    print("Token IDs:", token_ids)
    print("Token bytes:", [encoding.decode_single_token_bytes(i) for i in token_ids])
    print("Token count:", len(token_ids))
    print("Round trip:", encoding.decode(token_ids) == text)
    print("This tokenization runs locally; no embedding or inference API was called.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="database backups")
    parser.add_argument("--merges", type=int, default=12)
    parser.add_argument(
        "--compare-openai",
        action="store_true",
        help="Compare with tiktoken; requires the tokens extra and a first-use download.",
    )
    args = parser.parse_args(argv)
    if args.merges < 0:
        parser.error("--merges must be non-negative.")

    merges = train_bpe(TRAINING_TEXTS, args.merges)
    vocabulary = build_vocabulary(merges)
    print("Lesson 2: tokens and the text-to-embedding pipeline")
    print("Text:", ascii(args.text))
    print("Whitespace pieces:", len(args.text.split()))
    print("Unicode code points:", len(args.text))
    print("UTF-8 bytes:", len(args.text.encode("utf-8")))
    print()
    print("Toy BPE learned merge rules:")
    for new_id, (left, right) in enumerate(merges, start=256):
        print(f"  {new_id}: {vocabulary[left]!r} + {vocabulary[right]!r}"
              f" -> {vocabulary[new_id]!r}")
    token_ids = encode(args.text, merges)
    print("Toy token IDs:", token_ids)
    print("Toy token bytes:", [vocabulary[token_id] for token_id in token_ids])
    print("Toy token count:", len(token_ids))
    print("Round trip:", decode(token_ids, merges) == args.text)
    print("Token IDs are vocabulary labels, not semantic vector coordinates.")

    if args.compare_openai:
        try:
            show_openai_tokens(args.text)
        except ImportError:
            parser.error('Install the comparison dependency with: uv sync --extra dev --extra tokens')


if __name__ == "__main__":
    main()
