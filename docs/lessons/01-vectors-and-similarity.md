# Milestone 1: vectors and similarity

Outcome: run a tiny document-comparison demo and explain why its three metrics
behave differently. This is one foundations lesson, not a search engine or RAG app.

## 1. Concept

First, place the terms:

- **AI** is the broad field of making computers perform tasks associated with intelligence.
- **Machine learning** is a way to build AI by learning patterns from examples.
- **Deep learning** is machine learning using neural networks with multiple layers.
- **An LLM** is a large language model, typically a deep neural network trained
  to model token sequences. A generative LLM predicts a distribution over possible
  next tokens and repeatedly selects tokens to produce text.
- **A token** is a model-specific unit of text. It might be a word, part of a word,
  punctuation, or another text unit. Token counts are not word counts. We will
  inspect actual tokenization in the next lesson.
- **A vector** is an ordered list of numbers, such as `[4.0, 1.0]`.
- **An embedding** maps an item such as text into a vector space where useful
  relationships can be represented geometrically. Modern text embedding models
  learn this mapping from data. An embedding model and a generative LLM have
  different output jobs: a vector versus a generated sequence.

For today's hand-assigned vectors, coordinate 0 means database relevance and
coordinate 1 means cooking relevance:

| Item | Hand-assigned vector |
| --- | --- |
| "How do we back up PostgreSQL?" | [1, 0] |
| "PostgreSQL backup guide" | [4, 1] |
| "Pasta cooking guide" | [0, 4] |

These numbers are judgments we supplied. Nothing in this lesson automatically
reads a sentence and understands its meaning. A learned embedding usually has
many dimensions whose individual coordinates do not have readable topic names.
Even matching dimension counts do not make vectors from different models comparable.

### Picture: direction versus length

![Two vectors A and B and the angle theta between them](https://assets.ibm.com/is/image/ibm/vector-search-cosine?dpr=off&ts=1771608320470)

Source: [IBM, What are distance metrics?](https://www.ibm.com/think/topics/distance-metrics),
cosine illustration. The image stays hosted at its source.

Read it step by step:

1. The two arrows labeled **A** and **B** start at the same origin.
   Each arrow represents a vector.
2. Their endpoints have fruit icons; these stand for the two items being compared.
   They do not mean the diagram's axes are literal fruit features.
3. The curved mark **theta (θ)** is the angle between the arrows.
4. In **Sim(A,B) = cos(θ) = A·B / (||A|| ||B||)**, the numerator combines
   matching coordinates, and the denominator divides out both vector lengths.
5. Extending either arrow without changing its direction leaves θ unchanged.
   That is the property we will test with `[4,1]` and `[8,2]`.

Cosine is the normalized dot product; this is also the definition used by the
[official scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).
We implement the calculation ourselves rather than installing that library.

## 2. Why it matters

A user may ask "How do I recover a database?" while a relevant document says
"PostgreSQL backup restoration." Exact word matching can miss useful relationships.

Later, an embedding model will place related questions and passages in a shared
vector space. A similarity calculation will let us compare them numerically.
Its usefulness depends on the representation: excellent arithmetic cannot rescue
a model that fails to represent the relationships we need.

Analogy: two arrows can point northeast even if one is twice as long.
Cosine asks whether their directions agree. It does not ask whether their endpoints
are nearby.

## 3. Architecture

Our current executable path:

```text
synthetic question + document titles
              |
      hand-assigned vectors
              |
      dot / L2 / cosine
              |
      printed comparison
```

Where the same math will fit later:

```text
question -> embedding model -> query vector
                                   |
                           similarity scoring <- stored chunk vectors
                                   |
                             relevant context -> LLM -> answer + sources
```

Today's comparison has no sorting, Top-K retrieval, HTTP server, storage, or LLM.
The full target architecture and technology choices are in the README.

## 4. Implementation

We build three metrics and the vector length they need.

**A. Dot product:** multiply coordinates in matching positions, then add.

```text
a · b = sum(a_i * b_i)
[1, 0] · [4, 1] = 1*4 + 0*1 = 4
```

**B. Magnitude:** the length of a vector from the origin.

```text
||v|| = sqrt(sum(v_i²))
||[4, 1]|| = sqrt(16 + 1) = sqrt(17)
```

**C. Euclidean distance:** the straight-line distance between endpoints.

```text
distance(a, b) = sqrt(sum((a_i - b_i)²))
distance([1, 0], [4, 1]) = sqrt(9 + 1) = sqrt(10)
```

**D. Cosine similarity:** dot product divided by both lengths.

```text
cosine(a, b) = (a · b) / (||a|| * ||b||)
cosine([1, 0], [4, 1]) = 4 / (1 * sqrt(17)) = 0.9701425...
```

Higher dot product or cosine means greater similarity under those respective
scoring rules. Lower Euclidean distance means closer endpoints.
Cosine lies between -1 and 1 for non-zero real vectors.

Run the example from the repository root:

```powershell
python -m rag_lab.demo
```

Install the dev environment and run pytest using the README commands.

## 5. Code walkthrough

Open `rag_lab/similarity.py` in this order:

1. **`_validate_vector`** rejects empty vectors and non-finite components.
   NaN can otherwise make a score unusable without a clear error.
2. **`_validate_pair`** checks equal dimension counts. Without that check,
   ordinary `zip` could silently drop unmatched coordinates.
3. **`dot_product`** uses a generator expression to multiply aligned numbers
   and `sum` to add them. There is no ML library hiding this operation.
4. **`magnitude`** implements the square-root formula with a small numerical
   safeguard. If `s = max(abs(v_i))`, then
   `||v|| = s * sqrt(sum((v_i/s)²))`. Squaring the scaled coordinates avoids
   overflowing large inputs or underflowing tiny ones. An all-zero vector
   returns length zero.
5. **`euclidean_distance`** subtracts the vectors coordinate by coordinate,
   then measures the length of that difference vector.
6. **`cosine_similarity`** divides each vector by its own magnitude, then takes
   their dot product. Algebraically, this is the same as dividing the original
   dot product by the length product. Normalizing first avoids large
   intermediate products. It rejects zero vectors, then clamps tiny
   floating-point overshoots back to the valid range [-1, 1].

In `rag_lab/demo.py`, `QUERY_VECTOR` and `DOCUMENTS` are ordinary Python values.
The loop prints comparisons in their original order. This makes it easy to
change the inputs and inspect each metric before implementing search.

For a vector with d dimensions, each metric takes O(d) arithmetic work.
This cosine implementation allocates normalized lists, so auxiliary space is O(d).
Searching N vectors and selecting Top-K is a separate lesson.

## 6. Example

The demo produces these values, rounded to four decimal places:

| Document vector | Dot with [1,0] | Euclidean distance | Cosine |
| --- | ---: | ---: | ---: |
| Backup [4,1] | 4 | 3.1623 | 0.9701 |
| Cooking [0,4] | 0 | 4.1231 | 0.0000 |
| Backup scaled [8,2] | 8 | 7.2801 | 0.9701 |

The scaled vector's dot product doubles, and its endpoint moves farther away.
Its ratio between coordinates stays 4:1, so its direction and cosine are unchanged.

The first score is close to 1 because the backup vector points mostly along our
database coordinate. It is **not** a 97% probability that the document answers the question.

## 7. Experiment

Before running each change, predict the result. Edit `rag_lab/demo.py`, run the
demo, and compare your prediction with its output.

1. Change `QUERY_VECTOR` from `[1.0, 0.0]` to `[0.0, 1.0]`.
   Which document should get the highest cosine? Why?
2. Replace the scaled backup vector `[8.0, 2.0]` with `[40.0, 10.0]`.
   Predict which metrics change and which stay the same.
3. Try `cosine_similarity([1, 0], [-1, 0])` in a Python REPL.
   Explain the negative result geometrically.
4. Try `cosine_similarity([0, 0], [4, 1])`.
   Explain the error, then restore the demo inputs.

No new API, database, or framework is required for these experiments.

## 8. Common mistakes

- **Treating hand-assigned features as learned embeddings.** The numbers in this
  lesson are supplied by us. A future model will learn the mapping from text.
- **Assuming each learned dimension names a topic.** Information is usually
  distributed across coordinates.
- **Comparing incompatible vectors.** They need a compatible model/space,
  dimension count, preprocessing, and coordinate order.
- **Reading cosine as a probability.** It measures geometric alignment; relevance
  needs evidence and evaluation.
- **Assuming cosine always ranges from 0 to 1.** Signed coordinates permit negative
  values. A score of -1 means opposite directions, not necessarily opposite meanings.
- **Calling zero cosine proof of unrelated meaning.** It means orthogonal vectors;
  the relationship to meaning depends on the embedding model.
- **Confusing similarity with distance.** Cosine similarity is better when larger;
  cosine distance is commonly defined as `1 - similarity`, so smaller is better.
- **Assuming longer text scales an embedding linearly.** Our scaling experiment
  is a geometric demonstration. Real models do not generally work that way.
- **Expecting a vector database to understand raw text.** It stores/indexes/searches
  representations. The embedding model supplies those representations.

## 9. Production perspective

Manual features are transparent and easy to debug, but do not scale to open-ended
language. Learned representations handle more relationships at the cost of model
dependencies, opaque coordinates, and a need to measure domain-specific retrieval.

Our Python functions teach the scoring rules. Production libraries use optimized
array operations; pgvector provides vector storage, distance operators, filtering,
and exact or approximate search in PostgreSQL. We will compare against our own
baseline when we reach that phase.

Use the metric appropriate to the model and application. If both vectors have
unit length, dot product equals cosine, and
`euclidean_distance(a,b)² = 2 - 2*cosine(a,b)`. Their rankings then agree,
apart from ties and numerical effects. For arbitrary unnormalized vectors, they
can rank candidates differently. The normalization relationship is documented
in the [scikit-learn pairwise metrics guide](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity).

Important later concerns include embedding model/version consistency, query versus
document encoding conventions, source provenance, evaluation, and access filtering.
We will add those only when their lessons arrive.

## 10. Checkpoint

Reply with your answers, using your own words:

1. What does a coordinate mean in our toy example? Why might that description
   fail for a learned embedding?
2. Why do `[4,1]` and `[8,2]` get the same cosine against `[1,0]`, while
   their dot products differ?
3. Compute cosine for `[1,0]` and `[1,1]` by hand. Show the numerator
   and both lengths.
4. Why is cosine undefined for `[0,0]`?
5. Describe one experiment you ran: your prediction, its output, and your explanation.

We will review these before moving on. The next lesson will inspect tokens and
the text-to-embedding path, so you can explain how real vectors replace our
hand-assigned features. Top-K search follows after that.
