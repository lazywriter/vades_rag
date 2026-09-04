# VADES Style-RAG: "top X novels in genre Y, style similar to author Z"

A self-contained pipeline: raw Project Gutenberg books → trained VADES_Fuse
style model → FAISS-backed genre-filtered recommender → LLM query parsing +
explanations. The same Gutenberg corpus used to train the style model is what
populates the retrieval "database" — one script produces both.

## Files

### Training (Gutenberg → trained model → FAISS store)

| File | Role |
|---|---|
| `gutenberg_data.py` | Loads/cleans/chunks Gutenberg HTML, book-level train/val/test split (no leakage) |
| `style_features.py` | Handcrafted 300-D stylometric feature vector + scaler (mean/std) persistence |
| `embeddings.py` | Universal Sentence Encoder (512-D) embedding extraction, cached per split |
| `vades_model.py` | `VADES_Fuse` model, `DocDataset`, contrastive loss, negative sampling |
| `train_vades.py` | Orchestrates all of the above → trains → exports everything below |

### Serving (query → retrieval → explanation)

| File | Role |
|---|---|
| `style_faiss_index.py` | Chunk-level FAISS index + per-chunk metadata (book, author, genre tags) |
| `style_recommender.py` | `VADESStyleRecommender` — genre-filtered ANN search + chunk-voting book ranking |
| `query_parser.py` | Free text → `{author, genre, top_k}` (LLM parser + offline fallback) |
| `explain.py` | Grounded one-line explanation per already-ranked result (never re-ranks) |
| `pipeline.py` | Wires it all into `run_query(text)` |

`requirements.txt` covers both halves; `book_genres_template.csv` shows the
genre-tagging format.

## What changed vs. the original notebooks (`vades_main.py` / `vades_recommendation_engine.py`)

- **Fixed a real ordering bug**: the notebook created `train_loader` /
  `val_loader` / `test_loader` in a cell *before* `DocDataset` was even
  defined. That only worked because Colab runs cells in whatever order you
  execute them, not top-to-bottom. `train_vades.py` defines things in actual
  dependency order, so it also runs correctly as a plain linear script — I
  verified this by running it end-to-end against a synthetic corpus (see
  below).
- **HTML cleaning is now structural, not just regex.** Modern Gutenberg HTML
  (the "Ebookmaker" format — this is what `pg11870-images.html`, the sample
  you shared, actually is) wraps its entire license blurb, machine-readable
  metadata block (Title/Author/Release date/Credits), and START/END markers
  inside `<section id="pg-header">`/`<section id="pg-footer">`. The original
  notebook's regex-only cleaning ran *after* flattening to text and never
  actually strips most of that block — verified this by running the
  original pipeline against your real file: it left ~500 words of license
  text and metadata prepended to the first chunk of every book. `gutenberg_data.py`
  now removes those `<section>` elements from the HTML tree first, then still
  runs the regex pass afterward as a second layer (for anything missed, and
  as the only cleaning for older Gutenberg HTML that doesn't use these
  semantic wrappers). See "What I verified" below for the actual before/after.
- **Checkpointing throughout**, since a full Gutenberg training run can take
  a long time and any one of these stages failing shouldn't mean starting
  over:
  - `load_or_build_chunks_df` (HTML cleaning) checkpoints every N books.
  - `cache_features` (style features) and `cache_use_embeddings` (USE
    embeddings) checkpoint every N items/batches.
  - The **training loop** checkpoints every epoch (full state: model,
    optimizer, scheduler, epoch number, loss history) to
    `checkpoints/latest.pth`, and **auto-resumes from it** the next time you
    run the same command — no flag needed. Pass `--fresh_start` to discard
    an existing checkpoint and retrain from epoch 1. A resume is checked
    against the architecture/data settings the checkpoint was trained with
    (`use_dim`, `style_in_dim`, `out_dim`, `hidden`, `n_authors`) — a
    mismatch (e.g. `--hidden` changed, or the corpus gained authors) raises
    a clear error instead of silently loading the wrong-shaped weights.
- **Removed a dummy-data bug**: the notebook's stylistic-feature-prediction
  eval cell had `Y_style = np.random.rand(512, 300)` — literally random
  noise standing in for real style features, disconnected from anything
  real. That eval isn't reproduced here; it wasn't needed for producing
  author/document embeddings and shouldn't be trusted as it was written.
  Happy to rebuild it properly against real `style_feats` if you want it.
- **Removed** `drive.mount`, `!pip install` magics (see `requirements.txt`
  instead), and hardcoded `/content/drive/...` paths — everything is a
  `--flag` now.
- **Chunk-level retrieval stays chunk-level** — no mean-pooling into one
  vector per novel (per your call: pooling loses information). A book's
  rank comes from how many of its chunks land in the ANN neighbor set for
  the query author (voting), tiebroken by average similarity.
- **FAISS replaces the sklearn brute-force cosine similarity** in the
  original `VADESRecommender` — same math (cosine via `IndexFlatIP` on
  L2-normalized vectors), but now has a real ANN path and genre-restricted
  search via `IDSelectorArray`.
- Query author vectors default to the **learned** `model.authors.weight`
  embeddings (what the contrastive loss trained to attract same-style
  documents), not a re-derived centroid.

### One naming thing worth flagging

The project description mentioned a "VIB framework." `VADES_Fuse` (in both
the original notebook and here) is a deterministic encoder trained with a
contrastive hinge loss — there's no stochastic latent or KL term, so it isn't
literally a Variational Information Bottleneck. If that's just your team's
name for this pipeline, no action needed. If you actually want a true VIB
(reparameterized stochastic bottleneck + KL regularizer), that's a
deliberate model change, not something I folded in silently — say the word
and I'll build it as its own variant so you can compare against the current
deterministic version.

## Step 1 — Get spaCy's English model

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# or, if that CDN is unreachable from your environment:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```
(`train_vades.py` will also try to auto-download it on first run if missing.)

## Step 2 — Tag genres (optional at train time, but needed for genre-filtered search)

Create a CSV with columns `book_id, genre`, where `book_id` matches
`f"{author}___{file_name}"`. See `book_genres_template.csv` for the exact
format `load_genre_map` expects (one row per book/genre pair, or
comma-separated genres in a single row). Books without a row here are
indexed with an empty genre list and are invisible to genre-filtered search
until tagged — you'll get a warning listing which ones at build time.

## Step 3 — Train + build the database in one command

```bash
python train_vades.py \
    --books_dir /path/to/gutenberg_books \
    --output_dir ./vades_run \
    --genre_csv book_genres.csv \
    --epochs 15
```

`books_dir` expects the same layout as the original notebook:
`books_dir/<author>/<book>.html`, one folder per author.

This single run produces everything the serving side needs:

```
vades_run/
  gutenberg_chunks.csv        cached cleaned+chunked corpus
  splits/                     cached book-level train/val/test split
  feat_cache/                 cached style features + USE embeddings per split
  checkpoints/best.pth        VADES_Fuse weights (best val loss)
  checkpoints/latest.pth      full resumable training state — safe to interrupt
  style_scaler.joblib         mean/std for the handcrafted style features
  vades_faiss_store/          chunk-level FAISS index + metadata  <- "the database"
  author_vectors.npy          learned per-author style embeddings
  author2id.json              author name -> row index into author_vectors
```

Every stage is checkpointed (see "What changed" above), so:
- Interrupting mid-run and re-running the **exact same command** resumes
  automatically, whichever stage it was in — HTML cleaning, feature/embedding
  extraction, or the training loop itself.
- Re-running after adding a `--genre_csv` you didn't have the first time, or
  after adding new books, only recomputes what actually changed.
- Want to discard a checkpoint and retrain clean instead? Pass `--fresh_start`.

A quick nearest-author-centroid accuracy is printed after training (against
the held-out test split) as a sanity check that the model actually learned
something before its embeddings get used to build a recommender — it's not
a replacement for the full evaluation suite (T-SNE, confusion matrix,
per-feature-group MSE) from the original notebook, just a fast go/no-go
signal.

**Minimum books per author**: the split stratifies twice (train/temp, then
val/test within temp), which needs roughly `ceil(2 / test_size)` books per
author (7 books/author at the default `test_size=0.3`) or you'll get a clear
error telling you which authors are too thin, rather than a raw sklearn
traceback.

## Step 4 — Query

```bash
export ANTHROPIC_API_KEY=...   # needed for the LLM query parser + explanations
```

```python
from style_faiss_index import load_index
from style_recommender import VADESStyleRecommender
from pipeline import run_query
import numpy as np, json

index, metadata_df = load_index("vades_run/vades_faiss_store")
author_vectors = np.load("vades_run/author_vectors.npy")
author2id = json.load(open("vades_run/author2id.json"))

recommender = VADESStyleRecommender(index, metadata_df, author_vectors, author2id)

results = run_query(
    "Suggest the top 5 gothic horror novels with a writing style similar to Edgar Allan Poe",
    recommender,
)
```

Or skip the LLM parser for quick testing:

```python
recs = recommender.recommend(author_name="Edgar Allan Poe", genre="Horror", top_k=5)
```

## Tuning notes

- **`over_fetch`** (in `recommend()`): starting size of the FAISS neighbor
  pool before aggregating to books. It auto-grows (up to the full
  genre-filtered candidate pool) if `exclude_same_author=True` would
  otherwise zero out results — verified this against a synthetic corpus
  where the query author's own books dominated a small initial pool.
- **`min_coverage`**: off by default (matches "book with the most matching
  chunks wins," as specified). Turn it on if long novels start dominating
  purely by having more chunks in the index rather than by being a closer
  style match — it normalizes match_count by each book's total chunk count.
- **Index type**: `IndexFlatIP` (exact) is fine up to roughly hundreds of
  thousands of chunks. Swap in `faiss.IndexIVFFlat` or `IndexHNSWFlat` in
  `build_faiss_index` if the corpus grows past that — nothing else needs
  to change.

## Known gaps (deferred per earlier discussion, flagging so they don't get lost)

- Unseen/new query authors aren't supported — `resolve_author` requires an
  exact match in `author2id`. (Training on more authors is the fix, not a
  code change.)
- Genre tagging is manual and external to this code; recommendation quality
  for genre-filtered queries is bounded by tagging coverage/consistency.
- `explain.py` makes one LLM call per recommendation — batch into a single
  prompt if `top_k` gets large and latency/cost matters.
- The full evaluation suite from the original notebook (T-SNE, confusion
  matrix, per-feature-group MSE via SVR) isn't reproduced in `train_vades.py`
  — only the quick nearest-centroid accuracy check. Say the word if you want
  that rebuilt as a separate `evaluate_vades.py` against real (not dummy)
  data.

## What I verified before handing this off

Since I can't reach `tfhub.dev` or `gutenberg.org` from this environment, I
built a small synthetic Gutenberg-shaped corpus and a stand-in for the USE
model, then ran the *entire* pipeline against it: HTML extraction → cleaning
→ chunking → book-level split → style features → (stubbed) embeddings →
VADES_Fuse training → checkpoint save/reload → FAISS store export → loading
that store back into `VADESStyleRecommender` → querying it. Specifically for
this round of changes:

- **HTML cleaning**: reconstructed a test file using the *actual*
  `<section id="pg-header">`/`<section id="pg-footer">` markup from your
  `pg11870-images.html` (verbatim boilerplate structure, real story text).
  Confirmed the new structural cleaning produces zero leakage of license
  text or metadata (`Release date`, `Credits`, `Title`, etc. all absent from
  the cleaned output), and separately confirmed the *original* notebook's
  regex-only approach, run against the same file, leaves ~500 words of that
  boilerplate in the first chunk of every book.
- **Checkpointing**: for each of `load_or_build_chunks_df`, `cache_features`,
  and `cache_use_embeddings`, simulated a crash partway through, confirmed
  the partial checkpoint was written, then confirmed resuming produced
  output identical to an uninterrupted run.
- **Training checkpointing**: ran a 5-epoch training run with a simulated
  crash after epoch 2, confirmed re-running the same command auto-resumes
  from epoch 3 with the optimizer/scheduler state intact (loss trajectory
  continues rather than restarting). Separately confirmed that resuming
  with a changed `--hidden` value correctly raises the architecture-mismatch
  error instead of silently loading mismatched weights, and that
  `--fresh_start` correctly bypasses an existing checkpoint.

The one thing none of this can verify is the real USE model's actual
embedding quality or real training dynamics at scale — that only shows up
once you run it against your real corpus.
