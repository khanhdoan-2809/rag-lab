# Document Intelligence & RAG Lab

Build a document question-answering platform while learning why every component
exists and how it works internally.

**Current milestone: 1 — vector representation and similarity.**
The code uses hand-assigned features and runs locally without API calls,
credentials, Docker, or runtime dependencies. The learner checkpoint is pending;
later milestones are plans, not implemented features.

Start with [the complete first lesson](docs/lessons/01-vectors-and-similarity.md).

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
| 1. Foundations | AI/ML/deep learning, LLMs and tokens; vector representations, learned embeddings, dimensions, dot product, Euclidean distance, cosine. **Milestone 1 implements only vector math and a hand-assigned example.** |
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
│   └── demo.py                 # Hand-assigned document/query vectors
├── tests/
│   └── test_similarity.py      # Known values, identities, invalid inputs
└── docs/
    └── lessons/
        └── 01-vectors-and-similarity.md
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

## What this milestone teaches

- A vector is an ordered list of coordinates. Coordinate order and meaning must agree.
- An embedding model learns a representation; our readable features are only a teaching example.
- Dot product combines component products and is sensitive to length.
- Euclidean distance measures straight-line separation.
- Cosine similarity compares direction and ignores positive rescaling.
- An all-zero vector has no direction, so this implementation rejects its cosine.
- A similarity score is not a calibrated probability or proof of relevance.
- Numerical and shape validation matter before a database is involved.

See the lesson for the worked example, code walkthrough, experiments, and checkpoint.

## Verification and learning state

Verified on Windows with Python 3.13.15 and pytest 9.2.0: **37 tests passed**.
The runnable demo output matches the worked example in the lesson. Python 3.12
is the minimum supported version; it was not separately tested in this environment.

Implementation and automated verification are separate from learner understanding.
The checked-in tests cover known scores, symmetry, positive rescaling, zero vectors,
unequal dimensions, empty inputs, NaN/infinity, and floating-point range behavior.

- Implementation: Milestone 1 only.
- Learner understanding: awaiting checkpoint answers.
- Next lesson, after the checkpoint: tokens and how a trained model turns text
  into embeddings. We will inspect a real embedding before building Top-K search.
