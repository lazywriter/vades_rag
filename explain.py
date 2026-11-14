# -*- coding: utf-8 -*-
"""
explain.py

Generates a short, grounded explanation for each already-ranked recommendation.

Important: this LLM step NEVER re-orders, filters, or adds results — ranking
is fully decided by style_recommender.py's FAISS + chunk-voting pipeline
before this module ever runs. explain.py only narrates a ranking that already
exists, using the retrieved evidence (match count, similarity, chunk previews)
so the explanation stays grounded instead of inventing stylistic claims about
books the model may know nothing about.
"""

from typing import List, Optional
import anthropic

from style_recommender import BookRecommendation

DEFAULT_MODEL = "claude-sonnet-5"

EXPLAIN_SYSTEM_PROMPT = """You write one short sentence (<= 30 words) explaining why a
recommended novel was matched to a target author's writing style, given:
- the target author's name
- how many of the novel's passages matched the target author's style signature,
  out of how many passages were retrieved overall
- the average style-similarity score
- a short preview of one matching passage

Base the explanation ONLY on the information given — do not invent plot details,
biographical claims, or stylistic features not implied by the data. Do not quote
the passage preview verbatim; refer to it only in general terms (e.g. "in its
long, winding sentences"). Respond with plain text, one sentence, no preamble."""


def explain_recommendation(target_author: str, rec: BookRecommendation,
                            client: Optional["anthropic.Anthropic"] = None,
                            model: str = DEFAULT_MODEL) -> str:
    client = client or anthropic.Anthropic()
    user_msg = (
        f"Target author: {target_author}\n"
        f"Recommended novel: {rec.title} by {rec.author}\n"
        f"Matching passages: {rec.match_count} of {rec.candidate_pool_size} retrieved\n"
        f"Average style similarity: {rec.avg_similarity:.3f}\n"
        f"Sample passage preview: {rec.sample_chunks[0] if rec.sample_chunks else '(none)'}"
    )
    resp = client.messages.create(
        model=model,
        # Headroom, not the expected length (the prompt already asks for one
        # <=30-word sentence): thinking tokens count toward max_tokens on
        # current models, and the old cap of 100 could leave no room for text.
        max_tokens=1024,
        system=EXPLAIN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if resp.stop_reason in ("refusal", "max_tokens") or not text:
        # One missing explanation shouldn't sink the whole result list (the
        # ranking is already decided); say so visibly instead of returning "".
        return f"(no explanation available: model stopped with stop_reason={resp.stop_reason!r})"
    return text


def explain_all(target_author: str, recs: List[BookRecommendation],
                 client: Optional["anthropic.Anthropic"] = None) -> List[str]:
    """One explanation call per recommendation. For larger top_k, consider
    batching these into a single prompt to cut latency/cost — kept simple
    (one call each) here for clarity and easy debugging."""
    client = client or anthropic.Anthropic()
    return [explain_recommendation(target_author, r, client=client) for r in recs]
