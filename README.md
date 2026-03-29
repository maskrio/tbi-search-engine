# TBI Search Engine

## Feature Highlights

- Two indexing pipelines: `BSBI` and `SPIMI`
- Two term-dictionary modes: `IdMap` and `Patricia`
- Three postings compression codecs: `Standard`, `VBE`, `Elias-Gamma`
- Ranked lexical retrieval: `TF-IDF` and `BM25`
- Top-k acceleration with `WAND`
- Semantic retrieval with `LSI + FAISS` (vector nearest-neighbor)
- Batch evaluation using IR metrics: `RBP`, `DCG`, `NDCG`, `AP`

## Quick Start (bash)

```bash
pip install -r requirements.txt

# 1) Build lexical index
python bsbi.py --compression vbe --index-mode bsbi --term-dict patricia

# 2) Build vector index for LSI+FAISS
python vector_build.py --components 128

# 3) Run search demo (lexical)
python search.py --compression vbe --term-dict patricia --scoring bm25 --retrieval wand --k 10

# 4) Run search demo (vector)
python search.py --retrieval faiss --k 10

# 5) Evaluate lexical retrieval
python evaluation.py --compression vbe --term-dict patricia --scoring bm25 --retrieval full --k 1000

# 6) Evaluate vector retrieval
python evaluation.py --retrieval faiss --k 1000
```

## Core Concepts

### Indexing

- `BSBI`: parses term-document pairs per block, then inverts and merges.
- `SPIMI`: performs direct in-memory inversion per block, then merges.

Both produce intermediate block indexes and one final merged main index.

### Term Dictionary

- `idmap`: Python dictionary mapping (`str -> int`)
- `patricia`: Patricia-tree dictionary persisted as `terms.patricia`

### Retrieval Modes

- `full`: full scoring over all matching candidates
- `wand`: dynamic pruning for faster top-k retrieval
- `faiss`: vector retrieval over LSI embeddings

### Scoring

- `tfidf`
- `bm25`

### Compression

- `standard`: no integer compression, larger index, simple decoding
- `vbe`: variable-byte compression, good size-speed tradeoff (recommended default)
- `elias-gamma`: stronger bit-level coding, usually smaller but slower decoding

### LSI + FAISS Design

1. Build sparse TF-IDF document matrix.
2. Apply `TruncatedSVD` to get low-dimensional LSI vectors.
3. Apply L2 normalization to all document LSI vectors.
4. For every query, project query TF-IDF into LSI space and apply L2 normalization.
5. Index vectors in FAISS (`IndexFlatL2`) for nearest-neighbor top-k retrieval.

Efficient SVD strategy:

- Uses `TruncatedSVD` (partial/randomized SVD), not full dense SVD.
- Effective components are capped by matrix size:
  - `actual_components = min(requested_components, n_docs - 1, n_terms - 1)`

### Evaluation

- `RBP`: rank-biased precision
- `DCG@10`: discounted gain at top-10
- `NDCG@10`: DCG normalized by ideal ranking
- `AP`: average precision across relevant hits

## Program Usage

### Build Lexical Index

```bash
python bsbi.py [--compression {standard|vbe|elias-gamma}] [--index-mode {bsbi|spimi}] [--term-dict {idmap|patricia}] [--data-dir collection] [--output-dir index]
```

Example:

```bash
python bsbi.py --compression elias-gamma --index-mode spimi --term-dict patricia
```

### Build Vector Index (LSI+FAISS)

```bash
python vector_build.py [--data-dir collection] [--output-dir index] [--components 128]
```

Example:

```bash
python vector_build.py --components 192
```

### Search Demo

```bash
python search.py [--compression {standard|vbe|elias-gamma}] [--term-dict {idmap|patricia}] [--scoring {tfidf|bm25}] [--retrieval {full|wand|faiss}] [--k 10] [--bm25-k1 1.2] [--bm25-b 0.75] [--vector-dir index]
```

Examples:

```bash
# Lexical retrieval
python search.py --compression vbe --term-dict patricia --scoring bm25 --retrieval wand --k 5

# Vector retrieval
python search.py --retrieval faiss --k 5
```

### Evaluation

```bash
python evaluation.py [--query-file queries.txt] [--k 1000] [--compression {standard|vbe|elias-gamma}] [--term-dict {idmap|patricia}] [--scoring {tfidf|bm25}] [--retrieval {full|wand|faiss}] [--bm25-k1 1.2] [--bm25-b 0.75] [--output-dir index] [--vector-dir index]
```

Examples:

```bash
# Lexical evaluation
python evaluation.py --compression vbe --term-dict patricia --scoring tfidf --retrieval full --k 1000

# Vector evaluation
python evaluation.py --retrieval faiss --k 1000
```

## Benchmark Results

### Retrieval Benchmark (k = 10)

| Method    | Avg. Time / query (ms) | Total (s) | RBP    | DCG@10 | NDCG@10 | AP     |
| --------- | --------- | --------- | ------ | ------ | ------- | ------ |
| TF-IDF    | 25.509    | 0.77      | 0.5471 | 2.7983 | 0.8540  | 0.7488 |
| BM25      | 25.457    | 0.76      | 0.5755 | 2.9511 | 0.8744  | 0.7830 |
| BM25+WAND | 24.143    | 0.72      | 0.5755 | 2.9511 | 0.8744  | 0.7830 |
| LSI+FAISS | 7.168     | 0.22      | 0.6225 | 3.2271 | 0.8761  | 0.7922 |

On this benchmark, LSI+FAISS is the best overall method. It has the
highest effectiveness scores (RBP, DCG@10, NDCG@10, and AP) and is also the fastest
method by a large margin in average query latency.

### Index Size 

| Compression | Term Dict | Time (s) | Size (bytes) | 
| ----------- | --------- | -------- | ------------ | 
| vbe         | idmap     | 1.64     | 2,162,733    | 
| vbe         | patricia  | 1.65     | 2,331,888    | 
| elias-gamma | idmap     | 1.97     | 2,579,313    | 
| elias-gamma | patricia  | 2.06     | 2,748,468    | 
| standard    | idmap     | 1.35     | 3,259,578    | 
| standard    | patricia  | 1.50     | 3,428,733    | 

## Data Format

- Collection: `collection/` with numeric subfolders as blocks
- Query file: `queries.txt` format `QID token1 token2 ...`
- Qrels file: `qrels.txt` format `QID DOCID`
