# Lesson 3 · Measure similarity with learned embeddings

## 1. Concept

Last lesson you could decode a sequence of token IDs to recover its text. An embedding encoder uses trained weights to process those tokens into a fixed-width vector of floats. Training shapes the space so that useful relationships show up as directions. Each coordinate participates in the representation. Inspecting one coordinate rarely tells you the topic.

![Vectors A and B with angle theta between them](https://assets.ibm.com/is/image/ibm/vector-search-cosine?dpr=off&ts=1771608320470)

Start at the common origin. Follow **A** as the query and **B** as one candidate. The arc **θ** marks the angle between them. As B turns toward A, θ shrinks and its cosine rises. Lesson 1's formula gives us that same comparison in 1,536 dimensions. [Diagram by IBM](https://www.ibm.com/think/topics/distance-metrics)

## 2. Why it matters

A runbook titled “Restore PostgreSQL from backup” could help someone asking “How do I recover lost data?” An embedding lets you test whether the model places these texts close together even when their wording differs.

## 3. Architecture

```text
three strings → embedding model → three vectors of the same width
                                      ↓
                       select one vector as the query
                                      ↓
                    compare it with each other vector
```

The offline exercise starts at the three-vector stage. The live exercise also runs the encoder. Both use the same Python scoring function.

## 4. Implementation

From the project root run the offline exercise. The three published vectors are included with the package.

```powershell
cd D:\Projects\rag-lab
.\.venv\Scripts\python.exe -m rag_lab.embedding_demo
```

To generate fresh embeddings, configure an API key in this terminal. In PowerShell 7 the first command prompts without displaying the key. The second makes one billable HTTPS request with the three example strings.

```powershell
$env:OPENAI_API_KEY = Read-Host -MaskInput "OpenAI API key"
.\.venv\Scripts\python.exe -m rag_lab.embedding_demo --live
```

The default live model is `text-embedding-3-small`. Override it via `OPENAI_EMBEDDING_MODEL`. [OpenAI quickstart](https://developers.openai.com/api/docs/quickstart) · [Embedding guide](https://developers.openai.com/api/docs/guides/embeddings)

## 5. Code walkthrough

Open [embedding_demo.py](../../rag_lab/embedding_demo.py). In `embed_texts`, the JSON field `input` contains three ordinary strings. The service tokenizes them using the selected model and returns one embedding per input. The code reads the response indices to associate each vector with the right sentence.

In `main`, inspect the vector width with `len(vector)`, its length with `magnitude(vector)`, and the first four values with `vector[:4]`. Follow the final call to `cosine_similarity(vectors[anchor], vector)`. It scores the full vectors. Printing four coordinates only keeps the display readable.

## 6. Example

The offline data comes from zero-based rows 20, 24 and 16 of the pinned CSV. Each vector represents the combined title and review body. Titles in the output identify the rows. [Cookbook data at the source revision](https://github.com/openai/openai-cookbook/blob/f1e13cfcc7e0f2a9015c562eede167b76b1d60fc/examples/data/fine_food_reviews_with_embeddings_1k.csv)

For the first coffee review the first four values are `[-0.04113, -0.00598, -0.00514, -0.01416]`. All three vectors have 1,536 coordinates and norms that round to 1.000000.

| Candidate compared with the first coffee review | Measured cosine |
| --- | ---: |
| Decaf coffee review | 0.4978 |
| Shampoo review | 0.3263 |

The coffee reviews align more closely. Both candidates are product reviews, so a shared writing style or vocabulary could also contribute. This example measures relative similarity for one query.

## 7. Experiment

Choose the second review as the query. Predict its score against the first coffee review before running the command.

```powershell
.\.venv\Scripts\python.exe -m rag_lab.embedding_demo --anchor 2
```

With API access, try paraphrasing a sentence or reversing its meaning. Record your prediction and the two measured scores.

```powershell
.\.venv\Scripts\python.exe -m rag_lab.embedding_demo --live --texts "Can we recover deleted records?" "Restore the missing rows from backup." "The backup cannot restore missing rows."
```

## 8. Common mistakes

A high score can reflect a shared topic even when two sentences contradict each other. Reading the value as a probability gives it a meaning the calculation does not support. Feeding Lesson 2's toy IDs directly to a model would also change their meaning because those IDs belong to our own vocabulary.

## 9. Production perspective

Encode stored passages and incoming queries in a compatible model space. Record model identity and dimension alongside stored vectors. Two models can return equally wide vectors with different coordinate systems. Re-embed the corpus when changing to an incompatible model. The offline sample and live batch are each compared within their own space.

## 10. Checkpoint

1. Why does switching the anchor preserve the score for the same pair?
2. Which input could support a recovery answer, and what would you check beyond its cosine score?
3. If you double the number of words in a review, what do you expect to happen to token count and embedding width?

Next lesson uses these comparisons to implement exact Top-K search over a collection.
