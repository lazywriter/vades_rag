# -*- coding: utf-8 -*-
"""
style_faiss_index.py

Builds and persists the CHUNK-level FAISS index over VADES document embeddings,
plus the per-chunk metadata (book_id, title, author, genre tags) needed for
genre-filtered, author-style-conditioned retrieval.

Design note: we deliberately do NOT mean-pool chunks into one vector per novel
before indexing. Retrieval happens at the chunk level; book-level ranking is
done downstream in style_recommender.py by counting how many of a book's
chunks land in the nearest-neighbor set for a given author query (voting),
since pooling was judged to lose too much information.

Expects, from the existing VADES pipeline (vades_main.py):
    all_doc_embeddings : np.ndarray [N_chunks, 300]   (VADES output, L2-normalized)
    doc_authors         : np.ndarray [N_chunks]         (author name per chunk)
    doc_texts           : np.ndarray [N_chunks]         (chunk text)
    train_df/val_df/test_df with a 'book_id' column
        (book_id = author + '___' + file_name, as defined in vades_main.py)

You additionally need a genre label per book_id — genres are assumed to be
assigned by you/your team beforehand (per your project description) and are
NOT inferred by this code. See `load_genre_map` for the expected CSV format.
"""

import os
import numpy as np
import pandas as pd
import faiss


# ----------------------------------------------------------------------
# Metadata assembly
# ----------------------------------------------------------------------

def build_chunk_metadata(book_ids, titles, authors, texts, genre_map):
    """
    Assemble one metadata row per chunk. The row index of the returned
    DataFrame IS the FAISS vector id used everywhere downstream — keep
    embeddings and this DataFrame in the same order.

    Args:
        book_ids: array-like [N_chunks] of book_id strings ("author___file.html")
        titles:   array-like [N_chunks] of display titles (file_name works if you
                  don't have separate book-title metadata yet)
        authors:  array-like [N_chunks] of author names
        texts:    array-like [N_chunks] of chunk text (used for previews/explanations
                  — only short slices are ever stored, see below)
        genre_map: dict book_id -> list[str] of genres (see load_genre_map)

    Returns:
        pd.DataFrame with columns: chunk_id, book_id, title, author, genres,
        text_preview (first 400 chars only — this is metadata for retrieval/
        display, not a text store; keep the full text elsewhere if you need it).
    """
    n = len(book_ids)
    rows = []
    missing_genre_books = set()
    for i in range(n):
        bid = book_ids[i]
        genres = genre_map.get(bid)
        if genres is None:
            genres = []
            missing_genre_books.add(bid)
        rows.append({
            "chunk_id": i,
            "book_id": bid,
            "title": titles[i],
            "author": authors[i],
            "genres": genres,          # list[str]; empty list = not yet tagged
            "text_preview": str(texts[i])[:400],
        })
    if missing_genre_books:
        print(f"WARNING: {len(missing_genre_books)} book(s) have no genre tag "
              f"and will be invisible to genre-filtered search until tagged, "
              f"e.g.: {sorted(missing_genre_books)[:5]}")
    return pd.DataFrame(rows)


def load_genre_map(csv_path):
    """
    Loads a book_id -> [genres] mapping from a CSV with columns: book_id, genre

    Either one row per (book_id, genre) pair, or comma-separated genres in a
    single row, are both accepted:

        book_id,genre
        Poe___tell_tale_heart.html,Gothic
        Poe___tell_tale_heart.html,Horror
        Austen___emma.html,"Romance,Comedy of Manners"
    """
    df = pd.read_csv(csv_path)
    genre_map = {}
    for _, row in df.iterrows():
        bid = row["book_id"]
        genres = [g.strip() for g in str(row["genre"]).split(",") if g.strip()]
        genre_map.setdefault(bid, [])
        for g in genres:
            if g not in genre_map[bid]:
                genre_map[bid].append(g)
    return genre_map


# ----------------------------------------------------------------------
# FAISS index
# ----------------------------------------------------------------------

def build_faiss_index(embeddings, already_normalized=True):
    """
    Builds a flat inner-product FAISS index (== cosine similarity, since VADES
    embeddings are L2-normalized) wrapped in an IndexIDMap so vector ids equal
    the row index into the metadata DataFrame from build_chunk_metadata.

    Scale note: IndexFlatIP is exact search, fine up to roughly hundreds of
    thousands of chunks. If your corpus grows well beyond that, swap this for
    faiss.IndexIVFFlat (needs a `.train()` call) or faiss.IndexHNSWFlat for
    approximate search — the rest of the pipeline (genre selector, voting
    aggregation) is unaffected by that swap.
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not already_normalized:
        faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)
    ids = np.arange(embeddings.shape[0]).astype(np.int64)
    index.add_with_ids(embeddings, ids)
    return index


def genre_selector(metadata_df, genre):
    """
    Builds a faiss.IDSelectorArray restricting search to chunks whose book is
    tagged with `genre` (case-insensitive, exact tag match against the tags
    you assigned via load_genre_map).

    Returns (selector, matching_chunk_ids). If no chunks match, matching_chunk_ids
    is empty — callers should check for that (e.g. VADESStyleRecommender.recommend
    returns [] and prints a warning rather than silently searching everything).
    """
    genre_lower = genre.strip().lower()
    mask = metadata_df["genres"].apply(
        lambda gs: any(g.lower() == genre_lower for g in gs)
    )
    matching_ids = metadata_df.loc[mask, "chunk_id"].to_numpy().astype(np.int64)
    selector = faiss.IDSelectorArray(matching_ids)
    return selector, matching_ids


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def save_index(index, metadata_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "chunks.faiss"))
    metadata_df.to_json(os.path.join(out_dir, "chunk_metadata.jsonl"),
                         orient="records", lines=True)
    print(f"Saved FAISS index + metadata for {len(metadata_df)} chunks to {out_dir}")


def load_index(out_dir):
    index = faiss.read_index(os.path.join(out_dir, "chunks.faiss"))
    metadata_df = pd.read_json(os.path.join(out_dir, "chunk_metadata.jsonl"),
                                orient="records", lines=True)
    return index, metadata_df
