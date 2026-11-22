# Lesson 2: tokens and the text-to-embedding pipeline

Outcome: inspect text as bytes, train a tiny byte-level BPE tokenizer, apply its
fixed rules to new text, and compare with a production tokenizer.
Work from `D:\Projects\rag-lab`. No API key or paid inference is needed.

The learner requested this lesson after Lesson 1. That progression is recorded
without claiming the previous checkpoint was assessed.

## 1. Concept

A **token** is a piece of text represented by an integer ID in a vocabulary.
Depending on the tokenizer and text, a token may represent a word, a subword,
punctuation, whitespace, or part of a Unicode character's bytes.

A **vocabulary** maps token IDs to their text bytes. An ID is a label.
ID 267 is not semantically "closer" to ID 268 than to ID 12494. Those numbers
cannot be used as embedding coordinates or passed to our cosine function
to measure the meaning of two sentences.

Consider three different counts:

```python
text = "database backups"
len(text.split())          # 2 whitespace-separated pieces
len(text)                 # 16 Unicode code points
len(text.encode("utf-8"))  # 16 bytes for this ASCII example
```

None of these necessarily equals a model's token count. Code points are also
not always the same as the number of characters a reader perceives.

### Existing picture: words can become smaller pieces

![A subword tokenizer splitting the sentence Let's do tokenization](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/bpe_subword.svg)

Source: [Hugging Face LLM Course — Tokenizers](https://huggingface.co/learn/llm-course/en/chapter2/4).
The original image remains hosted by Hugging Face.

Read the diagram from the sentence **Let's do tokenization!** to its smaller
pieces. The split around **tokenization** illustrates that one word can yield
multiple tokens. The divisions represent vocabulary units, not word meanings
or embedding dimensions. Exact boundaries depend on the tokenizer.

### Why byte-pair encoding helps

Using one token per possible word would need a large vocabulary and a plan for
unknown words. Using one token per byte can represent every valid UTF-8 string,
but creates long sequences. **Byte-level BPE** starts with 256 byte tokens and
learns merges of frequent adjacent tokens. It balances vocabulary size against
sequence length.

During **training**, we count adjacent pairs in a corpus and learn ordered rules.
During **encoding**, the vocabulary and rules are fixed. We apply them to new
text without retraining. Our implementation learns counts, not semantics.

For example, a corpus containing `abab` twice learns:

```text
Initial IDs:       a=97, b=98
First merge:       (97, 98) -> 256, whose bytes are b'ab'
Second merge:      (256, 256) -> 257, whose bytes are b'abab'
Encode "abab":     [257]
Encode "abx":      [256, 120]
```

The unknown letter x still has its original byte ID. It needs no special unknown
token. This guarantees representability for valid UTF-8 text, not model understanding.

References: [Hugging Face's BPE training explanation](https://huggingface.co/learn/llm-course/en/chapter6/5)
and [OpenAI's tiktoken project](https://github.com/openai/tiktoken).

## 2. Why it matters

Later, our embedding requests and RAG prompts must fit token-based input limits.
A chunk measured in characters or words does not reliably meet a token budget.

For a future RAG request, the available passage budget will depend on the model's
rules and everything else in the input: instructions, question, history, source
labels, formatting, and any reserved output/reasoning allowance. Counting only
retrieved text is insufficient.

Tokenization is also model-specific. Use the tokenizer appropriate to the selected
model; do not assume an embedding model and a generation model share one.

## 3. Architecture

```text
Text
  |
  v
Tokenizer: text -> token IDs                  <-- implemented in this lesson
  |
  v
Trained model: IDs -> internal token vectors
              -> context-dependent representations
              -> model-specific output
                     |
                     +-- embedding model -> text embedding
                     +-- generative LLM  -> next-token predictions -> generated text
```

The tokenizer identifies pieces; the model's learned weights provide numerical
representations and process relationships between pieces.

At a high level, a model can use token IDs to select rows from a learned
embedding table. Subsequent neural-network layers process these representations
together. A text embedding model then produces a vector for the input according
to its architecture. The returned text embedding is not simply the token-ID
list, and we should not assume a proprietary model just averages token vectors.

We are implementing the tokenizer stage. We are not training an embedding model
or claiming our toy IDs are compatible with any existing model.

Token count varies with input length. For a fixed embedding model/configuration,
the returned embedding dimension is fixed even when token counts differ. Two
tokens do not mean a two-dimensional text embedding.

## 4. Implementation

### A. Run the tokenizer we built ourselves

```powershell
Set-Location D:\Projects\rag-lab
.\.venv\Scripts\python.exe -m rag_lab.token_demo
```

The demo uses a fixed synthetic training corpus. It prints every learned merge,
the query's token IDs, their exact bytes, and a reconstruction check.

No package beyond Python is required for the toy demo:

```powershell
python -m rag_lab.token_demo
```

### B. Compare with the production tokenizer

The optional `tokens` dependency group adds `tiktoken` after the manual algorithm.

```powershell
uv sync --extra dev --extra tokens --locked --cache-dir .cache/uv
.\.venv\Scripts\python.exe -m rag_lab.token_demo --compare-openai
```

It uses a named encoding, `cl100k_base`, to make the comparison explicit.
OpenAI's [model-to-encoding mapping](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py)
currently associates this encoding with the text-embedding-3 models. We will
verify model support again when using a real embedding API.

The first comparison may download public encoding data into `.cache/tiktoken/`.
Subsequent runs use the cache. The text is tokenized locally; no text is sent to
an embedding or generation endpoint. The plain toy demo always works offline.

### C. Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The core test suite does not need tiktoken, vocabulary downloads, API keys, or
network access. It checks the code we wrote rather than retesting the library.

## 5. Code walkthrough

Read `rag_lab/bpe.py` in this order:

1. **`_merge_pair`** walks left to right and replaces non-overlapping matches.
   In `aaaaa`, merging aa gives `aa | aa | a`. Reusing the middle a
   in two replacements would corrupt the original bytes.
2. **`train_bpe`** converts corpus strings to UTF-8 byte IDs, counts adjacent
   pairs, chooses the most frequent pair, assigns ID `256 + merge_index`,
   replaces occurrences, and repeats.
3. A tie uses the smaller integer-ID pair so examples are reproducible.
   Corpus entries are separate sequences: pairs do not span entries.
   We stop at the requested limit or when no pair appears at least twice.
4. **`build_vocabulary`** reconstructs a merged token's bytes by concatenating
   the bytes of its two earlier tokens.
5. **`encode`** starts from bytes and applies the learned rules in order.
   It does not change those rules to suit the new input.
6. **`decode`** joins token bytes before decoding UTF-8. An individual token
   can end inside a multibyte character; it may not decode by itself.

In `rag_lab/token_demo.py`, the optional comparison calls:

```python
encoding = tiktoken.get_encoding("cl100k_base")
ids = encoding.encode_ordinary(text)
pieces = [encoding.decode_single_token_bytes(token_id) for token_id in ids]
```

`encode_ordinary` treats special-token-looking strings as ordinary input text.
This exercise is not a chat-message serializer.

This toy applies M merge rules in O(Mn) work for an input of n bytes.
Training repeatedly scans the shrinking corpus, giving an O(MB) upper bound
for B original corpus bytes. It favors readable loops over an optimized encoder.

## 6. Example

Verified with tiktoken 0.14.0 and the checked-in toy corpus:

| Measure for `database backups` | Result |
| --- | --- |
| Whitespace pieces | 2 |
| Unicode code points | 16 |
| UTF-8 bytes | 16 |
| Toy BPE tokens | 3 |
| cl100k_base tokens | 2 |

```text
Toy IDs:       [267, 261, 115]
Toy bytes:     [b'database', b' backup', b's']

cl100k IDs:    [12494, 60766]
cl100k bytes:  [b'database', b' backups']
```

Both reconstruct the exact input. The space before backups is included in a token.
Neither count tells us whether the text is relevant to a question.

A multibyte example:

```python
from rag_lab.bpe import decode, encode

encode("é", [])        # [195, 169]: two byte tokens with no merges
decode([195, 169], []) # "é": join the bytes, then decode once
```

## 7. Experiment

Predict each result before running it.

1. Disable merges:

   ```powershell
   .\.venv\Scripts\python.exe -m rag_lab.token_demo --merges 0
   ```

   For our ASCII example, what will the token count become?

2. Compare a word, capitalization, leading spaces, and an identifier:

   ```powershell
   .\.venv\Scripts\python.exe -m rag_lab.token_demo --text "database" --compare-openai
   .\.venv\Scripts\python.exe -m rag_lab.token_demo --text " Database" --compare-openai
   .\.venv\Scripts\python.exe -m rag_lab.token_demo --text "customer_invoice_id_2026" --compare-openai
   ```

   Record the pieces, not just the counts. Explain why word counting is unreliable.

3. In a Python REPL, inspect `encode("🙂", [])` and then decode the result.
   Compare code-point count, byte count, and token count.

4. Change the fixed training corpus to repeat a different domain word, then
   rerun the toy demo. Explain why changing a production model's tokenizer
   independently would break the relationship between IDs and learned weights.

## 8. Common mistakes

- **Equating words, code points, bytes, and tokens.** They are different units.
- **Using token IDs as semantic vectors.** IDs identify vocabulary entries.
- **Assuming a different tokenizer makes a pretrained model compatible.**
  The tokenizer and trained model must agree on IDs and conventions.
- **Learning merges from each incoming question.** Inference applies fixed rules.
- **Calling BPE training semantic learning.** It learns recurring byte patterns.
- **Decoding each byte token separately.** A character can span multiple tokens.
- **Assuming all byte-level encoders match.** Corpora, vocabularies, merge priorities,
  regex boundaries, normalization, and special tokens can differ.
- **Assuming fewer tokens guarantees better meaning or retrieval.** Compression
  is different from semantic quality.
- **Treating the plain-text count as the full chat-request count.** Protocol
  formatting and model-specific overhead are separate concerns.

## 9. Production perspective

Our tokenizer is an inspectable teaching implementation. It can merge across
spaces within a corpus entry. It has no regex pre-tokenization or special-token
machinery and does not produce OpenAI-compatible IDs.

tiktoken supplies a trained vocabulary and optimized BPE implementation, together
with encoding-specific pre-tokenization and special-token handling.
The [official educational implementation](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py)
is useful for comparing the same steps with a production library.

Production RAG systems count tokens using the selected model's tokenizer, preserve
source metadata while chunking, and budget for the full request. We will implement
chunking when we reach ingestion rather than adding it prematurely here.

## 10. Checkpoint

Reply with your reasoning and one observed result:

1. Why is token ID 267 not an embedding coordinate?
2. Starting from `[97,98,97,98]`, apply `(97,98)->256`, then
   `(256,256)->257`. What sequence remains?
3. Why can two tokenizers produce different counts while decoding to the same text?
4. What happens when you run the demo with zero merges? Explain the count.
5. Why should our later chunk budget count tokens rather than just words?

We will discuss your results before the next lesson. Next we will generate a real
text embedding, inspect its dimensions and values, and reuse Lesson 1's cosine
function to compare related and unrelated sentences. Top-K search comes afterward.
