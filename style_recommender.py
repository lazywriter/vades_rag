# -*- coding: utf-8 -*-
"""
style_recommender.py

Chunk-level, FAISS-backed, genre-filtered "novels with a writing style similar
to author Z" recommender.

Design (per project decisions):
  - No mean-pooling of chunks into a single novel vector — this was judged to
    lose too much signal. Instead we run ANN search over ALL chunks (optionally
    restricted to a genre via FAISS IDSelectorArray), then aggregate to the
    book level by VOTING: whichever book has the most chunks landing in the
    retrieved neighbor set for the query author is ranked highest, with
    average similarity as a tiebreaker.
  - Query authors are assumed to be present in the trained author table
    (model.authors.weight / author2id) — no on-the-fly encoding of unseen
    authors for now.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import faiss

from style_faiss_index import genre_selector


@dataclass
class BookRecommendation:
    book_id: str
    title: str
    author: str
    genres: List[str]
    match_count: int           # chunks of this book that landed in the retrieved set
    candidate_pool_size: int   # total chunks retrieved (across all books) this round
    avg_similarity: float      # mean cosine similarity of this book's matching chunks
    best_similarity: float     # max cosine similarity among this book's matching chunks
    coverage: float            # match_count / total chunks this book has in the index
    sample_chunks: List[str] = field(default_factory=list)  # previews, for explain.py


class VADESStyleRecommender:
    def __init__(self, index: faiss.Index, metadata_df: pd.DataFrame,
                 author_vectors: np.ndarray, author2id: Dict[str, int]):
        """
        Args:
            index: FAISS IndexIDMap over chunk embeddings (see style_faiss_index.py);
                   vector id == metadata_df['chunk_id']
            metadata_df: one row per chunk (chunk_id, book_id, title, author,
                         genres, text_preview) — see build_chunk_metadata
            author_vectors: [N_authors, 300], the LEARNED per-author style anchors
                            from VADES_Fuse, i.e. model.authors.weight.detach().cpu().numpy()
                            (these are what the contrastive loss trained to sit near
                            same-style documents — they're the right query vector
                            for "similar to author Z", as opposed to e.g. re-deriving
                            a centroid from that author's own documents)
            author2id: author name -> row index into author_vectors
        """
        self.index = index
        self.metadata_df = metadata_df.set_index("chunk_id", drop=False)
        self.author_vectors = np.ascontiguousarray(author_vectors, dtype=np.float32).copy()
        faiss.normalize_L2(self.author_vectors)  # keep cosine-consistent with chunk embeddings
        self.author2id = author2id

        # book_id -> total chunk count in the index (denominator for `coverage`)
        self._book_chunk_totals = metadata_df.groupby("book_id")["chunk_id"].count().to_dict()

    # --------------------------------------------------------------
    def resolve_author(self, author_name: str) -> int:
        """Exact/case-insensitive lookup only. Unseen authors raise KeyError
        by design (on-the-fly embedding for new authors is a deferred feature)."""
        norm = author_name.strip().lower()
        lookup = {a.strip().lower(): i for a, i in self.author2id.items()}
        if norm not in lookup:
            known = sorted(self.author2id.keys())
            raise KeyError(
                f"Author '{author_name}' is not in the trained author set "
                f"({len(known)} known authors, e.g. {known[:5]})."
            )
        return lookup[norm]

    # --------------------------------------------------------------
    def _search(self, query_vec: np.ndarray, top_m: int, selector=None):
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        if selector is not None:
            params = faiss.SearchParameters(sel=selector)
            sims, ids = self.index.search(q, top_m, params=params)
        else:
            sims, ids = self.index.search(q, top_m)
        sims, ids = sims[0], ids[0]
        valid = ids != -1
        return sims[valid], ids[valid]

    # --------------------------------------------------------------
    def recommend(self, author_name: str, genre: Optional[str] = None,
                  top_k: int = 5, over_fetch: int = 500,
                  exclude_same_author: bool = True,
                  min_coverage: float = 0.0) -> List[BookRecommendation]:
        """
        Args:
            author_name: query author ("style similar to Z")
            genre: genre tag to restrict candidates to (None = search all genres)
            top_k: number of novels to return
            over_fetch: starting number of nearest CHUNKS to pull from FAISS
                        before aggregating to books. Should be well above top_k,
                        since many neighbor chunks collapse onto the same handful
                        of books.

                        NOTE: when exclude_same_author=True this is only a
                        *starting* point — the query author's own books are
                        usually the closest match to their own style vector,
                        so a small over_fetch can be entirely made up of the
                        query author's own chunks, leaving nothing after
                        exclusion. `recommend` automatically re-searches with a
                        larger pool (up to the full candidate universe) until
                        it finds `top_k` distinct other-author books or runs
                        out of candidates — see `_over_fetch` growth below.
            exclude_same_author: drop the query author's own books from results
                        (recommended — otherwise "similar to Z" just returns Z)
            min_coverage: drop books whose match_count / total_chunks_for_book
                        is below this fraction. Off (0.0) by default to match
                        "book with the most matching chunks wins" as specified;
                        turn this up if you notice long books winning purely by
                        having more chunks in the corpus rather than by being a
                        closer style match.

        Returns:
            Top-k BookRecommendation, ranked by (match_count desc, avg_similarity desc)
        """
        author_id = self.resolve_author(author_name)
        query_vec = self.author_vectors[author_id]
        author_norm = author_name.strip().lower()

        selector = None
        universe_size = self.index.ntotal
        if genre is not None:
            selector, candidate_chunk_ids = genre_selector(self.metadata_df, genre)
            if len(candidate_chunk_ids) == 0:
                known = sorted({g for gs in self.metadata_df["genres"] for g in gs})
                print(f"No chunks are tagged with genre '{genre}'. Returning no results. "
                      f"Known genre tags: {known}")
                return []
            universe_size = len(candidate_chunk_ids)

        # ---- adaptive over-fetch: grow the search pool until exclusion
        #      still leaves enough distinct books, or we exhaust the universe ----
        current_fetch = min(over_fetch, universe_size)
        per_book_sims: Dict[str, List[float]] = {}
        per_book_chunks: Dict[str, List[int]] = {}
        pool_size = 0
        while True:
            sims, ids = self._search(query_vec, current_fetch, selector=selector)
            pool_size = int(len(ids))

            per_book_sims, per_book_chunks = {}, {}
            for sim, cid in zip(sims, ids):
                row = self.metadata_df.loc[int(cid)]
                if exclude_same_author and row["author"].strip().lower() == author_norm:
                    continue
                bid = row["book_id"]
                per_book_sims.setdefault(bid, []).append(float(sim))
                per_book_chunks.setdefault(bid, []).append(int(cid))

            if len(per_book_sims) >= top_k or current_fetch >= universe_size:
                break
            current_fetch = min(current_fetch * 4, universe_size)

        results = []
        for bid, sim_list in per_book_sims.items():
            total_chunks = self._book_chunk_totals.get(bid, len(sim_list))
            coverage = len(sim_list) / max(1, total_chunks)
            if coverage < min_coverage:
                continue
            rep_chunk_ids = per_book_chunks[bid]
            sample_row = self.metadata_df.loc[rep_chunk_ids[0]]
            results.append(BookRecommendation(
                book_id=bid,
                title=sample_row["title"],
                author=sample_row["author"],
                genres=sample_row["genres"],
                match_count=len(sim_list),
                candidate_pool_size=pool_size,
                avg_similarity=float(np.mean(sim_list)),
                best_similarity=float(np.max(sim_list)),
                coverage=coverage,
                sample_chunks=[self.metadata_df.loc[c]["text_preview"] for c in rep_chunk_ids[:3]],
            ))

        results.sort(key=lambda r: (r.match_count, r.avg_similarity), reverse=True)
        return results[:top_k]
