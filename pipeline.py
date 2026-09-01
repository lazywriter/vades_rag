# -*- coding: utf-8 -*-
"""
pipeline.py

End-to-end: free-text query -> parsed {author, genre, top_k} -> FAISS chunk
retrieval + voting-based book ranking -> LLM explanations -> printable results.

See README.md for how to plug in your actual VADES artifacts (embeddings,
author vectors, genre tags) from vades_main.py.
"""

from typing import List, Dict, Optional
import anthropic

from style_faiss_index import load_index
from style_recommender import VADESStyleRecommender
from query_parser import parse_query_llm, parse_query_fallback
from explain import explain_all


def run_query(query_text: str, recommender: VADESStyleRecommender,
              known_authors: Optional[List[str]] = None,
              known_genres: Optional[List[str]] = None,
              use_llm_parser: bool = True,
              generate_explanations: bool = True,
              client: Optional["anthropic.Anthropic"] = None) -> List[Dict]:
    """
    Args:
        query_text: the user's free-text request
        recommender: a built VADESStyleRecommender
        known_authors/known_genres: only needed if use_llm_parser=False
        use_llm_parser: if False, uses the offline regex/keyword fallback parser
        generate_explanations: if False, skips the LLM explanation calls (useful
                                for fast iteration / testing the retrieval logic
                                on its own)
        client: shared anthropic.Anthropic() instance (created if not passed)

    Returns:
        List of dicts: title, author, genres, match_count, avg_similarity, explanation
    """
    client = client or anthropic.Anthropic()

    if use_llm_parser:
        parsed = parse_query_llm(query_text, client=client)
    else:
        parsed = parse_query_fallback(query_text, known_authors or [], known_genres or [])

    if not parsed.get("author"):
        raise ValueError(f"Could not identify a target author in query: {query_text!r}")

    print(f"Parsed -> author={parsed['author']!r}, genre={parsed.get('genre')!r}, "
          f"top_k={parsed['top_k']}")

    recs = recommender.recommend(
        author_name=parsed["author"],
        genre=parsed.get("genre"),
        top_k=parsed["top_k"],
    )

    if not recs:
        print("No recommendations found for this author/genre combination.")
        return []

    explanations = (explain_all(parsed["author"], recs, client=client)
                     if generate_explanations else [""] * len(recs))

    return [
        {
            "title": rec.title,
            "author": rec.author,
            "genres": rec.genres,
            "match_count": rec.match_count,
            "candidate_pool_size": rec.candidate_pool_size,
            "avg_similarity": round(rec.avg_similarity, 4),
            "explanation": exp,
        }
        for rec, exp in zip(recs, explanations)
    ]


if __name__ == "__main__":
    # --- Wire this up to your real VADES artifacts (see README.md) ---
    index, metadata_df = load_index("vades_faiss_store")

    import numpy as np, json
    author_vectors = np.load("author_vectors.npy")   # model.authors.weight.cpu().numpy()
    author2id = json.load(open("author2id.json"))

    recommender = VADESStyleRecommender(index, metadata_df, author_vectors, author2id)

    results = run_query(
        "Suggest the top 5 gothic horror novels with a writing style similar to Edgar Allan Poe",
        recommender,
    )
    for r in results:
        print(f"\n{r['title']} by {r['author']}  "
              f"({r['match_count']}/{r['candidate_pool_size']} matches, "
              f"avg_sim={r['avg_similarity']})")
        print(f"  {r['explanation']}")
