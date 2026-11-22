"""Milestone 1: inspect hand-assigned features, then compare vector metrics."""

from rag_lab.similarity import cosine_similarity, dot_product, euclidean_distance

# Coordinate order is always [database relevance, cooking relevance].
# We assigned these numbers. They are NOT generated from the text by a model.
QUERY_TEXT = "How do we back up PostgreSQL?"
QUERY_VECTOR = [1.0, 0.0]
DOCUMENTS = [
    ("PostgreSQL backup guide", [4.0, 1.0]),
    ("Pasta cooking guide", [0.0, 4.0]),
    ("Backup guide: scaled vector", [8.0, 2.0]),
]


def main() -> None:
    print("Milestone 1: vector representation and similarity")
    print("Features: [database relevance, cooking relevance]")
    print("Hand-assigned features; no model, API, or semantic search yet.")
    print(f"Question: {QUERY_TEXT}")
    print(f"Query vector: {QUERY_VECTOR}")
    print()
    print(f"{'Document':<32} {'Vector':<12} {'Dot':>7} {'L2':>9} {'Cosine':>9}")
    for title, vector in DOCUMENTS:
        dot = dot_product(QUERY_VECTOR, vector)
        distance = euclidean_distance(QUERY_VECTOR, vector)
        cosine = cosine_similarity(QUERY_VECTOR, vector)
        print(f"{title:<32} {str(vector):<12} {dot:>7.2f} {distance:>9.4f} {cosine:>9.4f}")
    print()
    print("The scaled backup vector has the same direction and cosine score.")
    print("Dot product and Euclidean distance change when its length changes.")


if __name__ == "__main__":
    main()
