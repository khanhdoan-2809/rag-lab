# Document Intelligence & RAG Lab

Build a document question-answering platform while learning why every component
exists and how it works internally.

**Current lesson 3 · Inspect learned text embeddings**

Run `python -m rag_lab.embedding_demo` to inspect three published OpenAI Cookbook vectors offline. Use `--live` with an environment API key to generate fresh embeddings. Both paths reuse the cosine function from Lesson 1.

[Lesson 1 · Vector math](docs/lessons/01-vectors-and-similarity.md) · [Lesson 2 · Tokens](docs/lessons/02-tokens-and-embeddings.md) · [Lesson 3 · Learned embeddings](docs/lessons/03-learned-embeddings.md)

## The application we will build

Users will upload PDF, TXT, and Markdown documents, search their contents, and ask
questions whose answers cite the original document passages.

This is the target architecture:

```text
INGESTION
PDF / TXT / Markdown
        |
        v
FastAPI upload -> document storage + metadata
        |
        v
extract text -> normalize -> chunk + source metadata
                                  |
                      +-----------+-----------+
                      v                       v
               embedding model        searchable text
                      |                       |
                      v                       v
              PostgreSQL + pgvector    BM25 retrieval
                      |                       |
                      +-----------+-----------+
                                  |
QUESTION ANSWERING                  |
question -> query processing ------+
                      |
                      v
        semantic / keyword retrieval
                      |
                      v
              fuse -> rerank
                      |
                      v
        select context + source identifiers
                      |
                      v
              prompt -> LLM
                      |
                      v
             answer + source passages

EVALUATION: question set -> retrieval metrics + answer checks
OPERATIONS: jobs, retries, tenancy, deletion, tracing, cost tracking
```

The diagram shows the full destination. We will begin with one retrieval method;
fusion and reranking arrive only after we can measure a baseline.

### Why each component exists

| Component | Responsibility and trade-off |
| --- | --- |
| FastAPI + Pydantic | HTTP contracts and validation; added after the local math/search lessons. |
| Extraction + cleaning | Convert file formats to usable text; PDF layout and OCR need special treatment. |
| Chunking + metadata | Create retrievable passages and preserve document/page/offset provenance; smaller chunks can lose context. |
| Embedding model | Convert text into learned numerical representations; retrieval quality depends on the model and data. |
| PostgreSQL + pgvector | Keep relational metadata and vectors together; exact search is simple, approximate indexes trade recall for speed. |
| BM25 | Score lexical matches using term frequency, rarity, and document length; helps with exact names and identifiers. We will implement it manually first. |
| Fusion + reranker | Combine complementary candidates and reorder them; adds latency and evaluation work. |
| Context builder + LLM | Select passages that fit the budget and generate grounded responses; retrieved content can still be insufficient or misleading. |
| Citation handling | Carry passage identifiers into the prompt and resolve citations back to stored sources; citations still need verification. |
| Evaluation | Compare retrieval and answer behavior using fixed questions and relevance labels. |

PostgreSQL can store vectors as ordinary data even without an extension.
Traditional B-tree indexes and equality/substring matching do not directly solve
semantic nearest-neighbor search. pgvector adds vector operations and specialized
indexes to an existing relational database.

## Learning roadmap

Each phase contains multiple small lessons. We stop at each lesson's checkpoint.

| Phase | Lessons and concrete result |
| --- | --- |
| 1. Foundations | AI/ML/deep learning, LLMs and tokens; vector representations, learned embeddings, dimensions, dot product, Euclidean distance, cosine. **Lessons 1–2 implement vector math and byte-level tokenization.** |
| 2. Search from scratch | Generate real embeddings; store them in Python; implement brute-force Top-K; measure O(Nd) scoring and ranking costs; compare models/dimensions. |
| 3. Ingestion | TXT/Markdown/PDF upload and extraction, normalization, fixed/overlapping/recursive chunks, metadata, batch embedding, document/chunk persistence. Introduce FastAPI/Pydantic here. |
| 4. Vector database | Docker Compose with PostgreSQL/pgvector; SQL operators, filters, exact search, SQLAlchemy/Alembic, HNSW and IVFFlat; measure recall versus latency against the Python baseline. |
| 5. Retrieval engineering | Manual BM25, semantic versus keyword search, RRF hybrid retrieval, metadata filtering, query rewriting, multi-query retrieval, contextual retrieval, reranking and relevance experiments. |
| 6. RAG | Retrieval -> context -> prompt -> LLM -> answer with sources. Teach RAG versus fine-tuning, context windows/budgets, compression, abstention, faithfulness and conversation history. |
| 7. Evaluation | Label questions, expected answers and relevant passages; implement Precision/Recall/Hit Rate@K, MRR, NDCG; assess faithfulness, answer relevance and context precision/recall. Compare configurations. |
| 8. Advanced RAG | Measure when expansion, contextual/parent-child/recursive/self-query retrieval, HyDE, GraphRAG, routing, adaptive retrieval and agents help; compare with our baseline. |
| 9. Production | Background jobs, async work, batching, retries/rate limits, caching, traces, cost tracking, security, tenant isolation, deletion, versions, idempotency, recovery and scaling. |

We will use OpenAI APIs when the embedding and generation lessons arrive.
Model choices, pricing, and SDK details will be checked at that time.
LangChain and LlamaIndex are deferred until the underlying pipeline is understood.

## Repository structure

Files that exist now:

```text
rag-lab/
├── AGENTS.md                   # Teaching and development rules for future lessons
├── README.md                   # Architecture, roadmap, setup, current learning state
├── pyproject.toml              # Python package and pytest configuration
├── uv.lock                     # Reproducible dependency resolution
├── rag_lab/
│   ├── __init__.py
│   ├── similarity.py           # Inspectable vector arithmetic
│   ├── demo.py                 # Hand-assigned document/query vectors
│   ├── bpe.py                  # Train, encode, and decode a toy byte-level BPE
│   ├── embedding_demo.py       # Inspect offline or live vectors
│   ├── review_vectors.json     # Three published vectors with a pinned source
│   └── token_demo.py           # Token inspection and optional tiktoken comparison
├── tests/
│   ├── test_similarity.py      # Known values, identities, invalid inputs
│   ├── test_embedding_demo.py  # Batch response ordering and input validation
│   └── test_bpe.py             # Merge learning, boundaries, Unicode round trips
└── docs/
    └── lessons/
        ├── 01-vectors-and-similarity.md
        ├── 02-tokens-and-embeddings.md
        └── 03-learned-embeddings.md
```

Directories we will add as needed: `rag_lab/api/` for HTTP endpoints,
`rag_lab/ingestion/` for document processing, `rag_lab/retrieval/` for search,
`rag_lab/rag/` for context/prompt/answer handling, `rag_lab/db/` and
`alembic/` for persistence, `evals/` for labeled questions and measurements,
and `data/samples/` for small synthetic documents. Docker Compose will be added
with the first database lesson. Empty future directories are intentionally deferred.

## Run Milestone 1

Prerequisite: Python 3.12 or newer. The demo uses only the standard library:

```powershell
python -m rag_lab.demo
```

Run that command from this directory, which contains `pyproject.toml`.
All current data is synthetic and lives in `rag_lab/demo.py`.
There is no active configuration or API key to supply at this milestone.

For isolated dependencies and tests, use either option below.
Virtual environments and installation are covered in the
[official pytest getting started guide](https://pytest.org/en/stable/getting-started.html).

With uv and the checked-in lockfile:

```powershell
uv sync --extra dev --locked
uv run --extra dev --locked pytest -q
uv run --locked python -m rag_lab.demo
```

Or standard Python on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m rag_lab.demo
```

On Linux/macOS, use `.venv/bin/python` in place of the Windows executable.
The pip option resolves versions from `pyproject.toml`; use uv for the locked
environment. No shell activation is required with either option.

## Run Lesson 2

```powershell
python -m rag_lab.token_demo
```

For the production tokenizer comparison:

```powershell
uv sync --extra dev --extra tokens --locked --cache-dir .cache/uv
.\.venv\Scripts\python.exe -m rag_lab.token_demo --compare-openai
```

The first comparison downloads public encoding data into `.cache/tiktoken/`.
Text is tokenized locally; no embedding or generation API call is made.
The core tests run offline without the optional `tokens` dependency.

## What the lessons teach

- A vector is an ordered list of coordinates. Coordinate order and meaning must agree.
- An embedding model learns a representation; our readable features are only a teaching example.
- Dot product combines component products and is sensitive to length.
- Euclidean distance measures straight-line separation.
- Cosine similarity compares direction and ignores positive rescaling.
- An all-zero vector has no direction, so this implementation rejects its cosine.
- A similarity score is not a calibrated probability or proof of relevance.
- Numerical and shape validation matter before a database is involved.

Lesson 2 adds byte-level BPE training versus encoding, fixed vocabularies,
lossless decoding, whitespace/Unicode behavior, and model-specific token counts.
Token IDs identify vocabulary entries; learned embeddings represent items as
vectors. They have different jobs.

Each lesson includes a worked example, code walkthrough, experiments, and checkpoint.

## Verification and learning state

Verified on Windows with Python 3.13.15 and pytest 9.2.0: **69 tests passed**.
The runnable demos match the worked examples. The optional comparison was also
verified with tiktoken 0.14.0 using the `cl100k_base` encoding. Python 3.12
is the minimum supported version; it was not separately tested in this environment.

Implementation and automated verification are separate from learner understanding.
The checked-in tests cover known scores, symmetry, positive rescaling, zero vectors,
unequal dimensions, empty inputs, NaN/infinity, and floating-point range behavior.
Tokenizer tests cover learned merge order, fixed inference rules, non-overlapping
replacement, corpus boundaries, byte preservation, Unicode, and invalid inputs.

- Lesson 3 demonstrates fixed embedding width, vector norms and pairwise cosine on real review embeddings.
- Offline comparison verified. Live request construction and response handling were tested with a stand-in response. Live inference requires an API key.
- Lesson 3 checkpoint pending. Next lesson implements exact Top-K search.
