"""Inspect published vectors offline or embed three texts through the OpenAI API."""
import argparse
import json
import os
from importlib.resources import files
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag_lab.similarity import cosine_similarity, magnitude


def embed_texts(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    """Send a single batch and associate each vector with its original input."""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Supply a nonblank string for every input.")
    request = Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": model, "input": texts, "encoding_format": "float"}).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        rows = sorted(json.load(response)["data"], key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(len(texts))):
        raise ValueError("Response indices do not match the inputs.")
    return [row["embedding"] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send three texts to OpenAI (billable).")
    parser.add_argument("--texts", nargs=3, metavar=("QUERY", "A", "B"), help="Custom live inputs.")
    parser.add_argument("--anchor", type=int, choices=(1, 2, 3), default=1,
                        help="Choose which input to compare against the other two.")
    args = parser.parse_args()
    if args.texts and not args.live:
        parser.error("--texts requires --live. The offline vectors correspond to fixed reviews.")
    if args.live:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            parser.error("Set OPENAI_API_KEY in your terminal for --live.")
        texts = args.texts or ["How do I recover a lost database?",
                               "Restore PostgreSQL from yesterday's backup.",
                               "Simmer the pasta until tender."]
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        try:
            vectors = embed_texts(texts, model, key)
        except HTTPError as exc:
            parser.exit(1, f"Embeddings API returned HTTP {exc.code}. Check the model, key and account limits.\n")
        except (URLError, TimeoutError, ValueError) as exc:
            parser.exit(1, f"Embedding request failed: {exc}\n")
        print("Live model:", model)
    else:
        data = json.loads(files("rag_lab").joinpath("review_vectors.json").read_text(encoding="utf-8"))
        texts = [sample["title"] for sample in data["samples"]]
        vectors = [sample["vector"] for sample in data["samples"]]
        print("Offline: published embeddings of full reviews, displayed by title.")
    anchor = args.anchor - 1
    for title, vector in zip(texts, vectors, strict=True):
        print(f"{title} | dimensions={len(vector)} | norm={magnitude(vector):.6f}"
              f" | first four={[round(value, 5) for value in vector[:4]]}")
    print(f"\nComparing against: {texts[anchor]}")
    for index, vector in enumerate(vectors):
        if index != anchor:
            print(f"{cosine_similarity(vectors[anchor], vector):.4f}  {texts[index]}")


if __name__ == "__main__":
    main()
