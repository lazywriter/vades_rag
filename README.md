# VADES Style-RAG

A retrieval-augmented recommender that answers queries like:

> *"Suggest the top 5 gothic horror novels with a writing style similar to Edgar Allan Poe"*

The system learns a stylometric embedding space from Project Gutenberg books
using the VADES_Fuse model, indexes every book chunk in FAISS, and combines
genre-filtered nearest-neighbour search with an LLM for query parsing and
result explanations. The same corpus is used both to train the style model
and to populate the retrieval index.

## Architecture

```
                 ┌──────────────────────── Training ────────────────────────┐
 Gutenberg HTML  │  gutenberg_data  →  style_features (300-D handcrafted)   │
 books_dir/      │        │         →  embeddings     (512-D USE)           │
 <author>/*.html │        ▼                    │                            │
                 │  book-level split    VADES_Fuse encoder (812 → 300-D)    │
                 │                      + learned author embeddings         │
                 │                      contrastive cosine loss             │
                 └──────────────────────────────┬───────────────────────────┘
                                                ▼
                         FAISS chunk index + metadata + author vectors
                                                │
                 ┌──────────────────────── Serving ─────────────────────────┐
  user query  →  │  query_parser  →  {author, genre, top_k}                 │
                 │  style_recommender  →  genre-filtered ANN search         │
                 │                        chunk voting → ranked books       │
                 │  explain  →  one grounded sentence per result            │
                 └──────────────────────────────────────────────────────────┘
```

### Style model

- **Input features** per text chunk:
  - 512-D Universal Sentence Encoder embedding (semantic content)
  - 300-D handcrafted stylometric vector: punctuation, function-word
    frequencies, POS/NER ratios, character and word n-grams, readability
    scores and distributional statistics (standardised with a saved scaler)
- **VADES_Fuse**: the two vectors are concatenated and passed through an MLP
  (`812 → 1024 → 300`, ReLU, dropout) to produce a document style embedding.
  A learned embedding table holds one 300-D vector per author.
- **Training objective**: a contrastive cosine hinge loss pulls document
  embeddings towards their own author's vector and pushes them away from
  sampled negative authors.

### Retrieval

- Every chunk embedding is L2-normalised and stored in a FAISS
  `IndexFlatIP` (exact cosine similarity), together with metadata
  (book, author, genre tags).
- A query author is represented by their learned author embedding.
- Genre filtering restricts the search to tagged books via `IDSelectorArray`.
  Multi-word genres such as `"gothic horror"` require books tagged with
  **all** the named genres.
- Chunks are not pooled per book. A book's score is the number of its chunks
  that appear in the neighbour set, with average similarity as a tiebreaker.
  An optional `min_coverage` setting normalises by book length.

### LLM layer

- `query_parser.py` turns free text into `{author, genre, top_k}` using the
  Anthropic API, with a rule-based offline fallback.
- `explain.py` writes a short explanation for each result. It only explains
  the ranking; it never changes it.

## Project structure

| File | Purpose |
|---|---|
| `gutenberg_data.py` | Load, clean and chunk Gutenberg HTML; book-level train/val/test split |
| `style_features.py` | 300-D stylometric feature extraction and scaler persistence |
| `embeddings.py` | Universal Sentence Encoder embeddings, cached per split |
| `vades_model.py` | `VADES_Fuse` model, dataset, negative sampling and contrastive loss |
| `checkpointing.py` | Helpers for saving and resuming partial progress |
| `train_vades.py` | End-to-end training and export of the retrieval store |
| `style_faiss_index.py` | Build, save and load the chunk-level FAISS index and metadata |
| `style_recommender.py` | `VADESStyleRecommender`: genre-filtered search and book ranking |
| `query_parser.py` | Natural-language query parsing |
| `explain.py` | Per-result explanations |
| `pipeline.py` | `run_query(text)` — ties parsing, retrieval and explanation together |
| `book_genres_template.csv` | Example genre-tagging file |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If the spaCy download fails, install the model wheel directly:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## Data preparation

Organise books one folder per author:

```
books_dir/
  Edgar Allan Poe/
    book1.html
    book2.html
  Mary Shelley/
    ...
```

Each author needs enough books for a stratified train/val/test split —
about 7 books per author with the default `test_size=0.3`.

### Genre tags

Genre filtering requires a CSV with columns `book_id, genre`, where
`book_id` is `"{author}___{file_name}"`. Use one row per book/genre pair, or
comma-separated genres in a single row (see `book_genres_template.csv`).
Untagged books are still indexed but will not appear in genre-filtered
results.

## Training

```bash
python train_vades.py \
    --books_dir /path/to/gutenberg_books \
    --output_dir ./vades_run \
    --genre_csv book_genres.csv \
    --epochs 15
```

Main options:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 15 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--hidden` | 1024 | Encoder hidden size |
| `--out_dim` | 300 | Embedding size |
| `--dropout` | 0.5 | Dropout rate |
| `--k_neg` | 5 | Negative authors per sample |
| `--margin` | 0.2 | Contrastive loss margin |
| `--fresh_start` | off | Ignore existing checkpoints and retrain |

Output:

```
vades_run/
  gutenberg_chunks.csv     cleaned and chunked corpus
  splits/                  book-level train/val/test split
  feat_cache/              cached style features and USE embeddings
  checkpoints/best.pth     best model weights (lowest validation loss)
  checkpoints/latest.pth   full training state for resuming
  style_scaler.joblib      feature scaler
  vades_faiss_store/       FAISS index and chunk metadata
  author_vectors.npy       learned author embeddings
  author2id.json           author name → embedding row
```

All long-running stages (HTML cleaning, feature extraction, embedding
extraction and training) are checkpointed. If a run is interrupted, running
the same command again resumes where it stopped. Resuming with different
model settings raises an error instead of loading incompatible weights.

After training, a nearest-author-centroid accuracy on the test split is
printed as a quick sanity check.

## Querying

Set an API key for the LLM parser and explanations:

```bash
export ANTHROPIC_API_KEY=...
```

```python
import json
import numpy as np
from style_faiss_index import load_index
from style_recommender import VADESStyleRecommender
from pipeline import run_query

index, metadata_df = load_index("vades_run/vades_faiss_store")
author_vectors = np.load("vades_run/author_vectors.npy")
author2id = json.load(open("vades_run/author2id.json"))

recommender = VADESStyleRecommender(index, metadata_df, author_vectors, author2id)

results = run_query(
    "Suggest the top 5 gothic horror novels with a writing style similar to Edgar Allan Poe",
    recommender,
)
```

To query without the LLM:

```python
recs = recommender.recommend(author_name="Edgar Allan Poe", genre="Horror", top_k=5)
```

## Configuration notes

- **`over_fetch`** — initial neighbour pool size. It grows automatically when
  excluding the query author's own books leaves too few results.
- **`min_coverage`** — off by default. Enable it to stop long novels from
  ranking highly only because they contribute more chunks.
- **Index type** — `IndexFlatIP` works well up to a few hundred thousand
  chunks. For larger corpora, switch to `IndexIVFFlat` or `IndexHNSWFlat` in
  `build_faiss_index`.

## Limitations

- Query authors must be in the training set (`author2id.json`).
- Genre-filtered results depend on the coverage and consistency of manual
  genre tags.
- Explanations use one LLM call per result, which adds latency for large
  `top_k` values.
