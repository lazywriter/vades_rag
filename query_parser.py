# -*- coding: utf-8 -*-
"""
query_parser.py

Turns a free-text request like:
    "suggest 5 gothic horror novels with a writing style similar to Edgar Allan Poe"
into a structured {author, genre, top_k} dict, which style_recommender.py then
uses to drive FAISS retrieval. The LLM here ONLY parses — it never sees or
influences the ranked results (that split is intentional, see explain.py).

Two parsers are provided:
  - parse_query_llm: uses the Anthropic API. Requires an ANTHROPIC_API_KEY
                      environment variable (or a client you construct yourself).
  - parse_query_fallback: no API call — regex/keyword matching against known
                      author & genre vocab. Less robust to phrasing, but useful
                      offline, in tests, or as a sanity check on LLM output.
"""

import os
import re
import json
from typing import Dict, List, Optional

import anthropic

DEFAULT_MODEL = "claude-sonnet-5"

PARSE_SYSTEM_PROMPT = """You extract three fields from a book-recommendation request:
- author: the author whose WRITING STYLE the user wants matched (not a genre, not a book title)
- genre: the genre the user wants novels from
- top_k: how many recommendations they want (integer; use 5 if not stated)

Respond with ONLY a JSON object, nothing else, no markdown code fences:
{"author": "<string or null>", "genre": "<string or null>", "top_k": <integer>}

If the author or genre is not mentioned in the request, use null for that field."""


def parse_query_llm(query: str, client: Optional["anthropic.Anthropic"] = None,
                     model: str = DEFAULT_MODEL) -> Dict:
    """
    Args:
        query: the user's natural-language request
        client: an anthropic.Anthropic() instance. If omitted, one is created
                from the ANTHROPIC_API_KEY environment variable.
        model: model string to use for parsing (a small/cheap model is plenty
               for this task)

    Returns:
        {"author": str|None, "genre": str|None, "top_k": int}

    Raises:
        ValueError if the model stopped without a complete answer (refusal,
        or hit max_tokens) or didn't return valid JSON — don't blindly trust
        LLM output in a pipeline.
    """
    client = client or anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        # Headroom, not the expected answer length: current models think
        # adaptively by default and thinking tokens count toward max_tokens,
        # so a tight cap (this used to be 200) can cut the JSON off mid-way.
        max_tokens=1024,
        system=PARSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise ValueError(f"Query parser got no complete answer from the model "
                         f"(stop_reason={resp.stop_reason!r}) for query {query!r}")
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Query parser expected JSON from the model, got: {text[:200]!r}") from e
    try:
        parsed["top_k"] = int(parsed.get("top_k") or 5)
    except (TypeError, ValueError):
        parsed["top_k"] = 5
    return parsed


def parse_query_fallback(query: str, known_authors: List[str],
                          known_genres: List[str]) -> Dict:
    """
    Best-effort parse without any API call — plain string matching against
    known vocab. Use as an offline fallback, or to cross-check the LLM
    parser's output for a quick sanity test.

    Note: this does word-bounded string matching (longest known name/genre
    first), so it won't catch paraphrases the LLM parser would (e.g.
    "detective fiction" -> "Mystery").
    """
    q_lower = query.lower()

    def _word_pattern(term: str) -> str:
        # word-bounded, so e.g. "Poe" doesn't match inside "poetry"
        return r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"

    author = next((a for a in sorted(known_authors, key=len, reverse=True)
                   if re.search(_word_pattern(a), q_lower)), None)

    # Collect EVERY known genre named (longest first, consuming matched text so
    # "Gothic" isn't re-found inside an already-matched "Gothic Horror" tag),
    # kept in the order they appear. Several genres come back comma-joined,
    # e.g. "Gothic, Horror", which genre_selector reads as "tagged with all".
    remaining, hits = q_lower, []
    for g in sorted(known_genres, key=len, reverse=True):
        m = re.search(_word_pattern(g), remaining)
        if m:
            hits.append((m.start(), g))
            remaining = remaining[:m.start()] + " " * len(m.group()) + remaining[m.end():]
    genre = ", ".join(g for _, g in sorted(hits)) or None

    # "top 5", or a count up to 4 words before novels/books ("4 gothic horror novels")
    m = re.search(r"\btop\s*(\d+)\b"
                  r"|\b(\d+)\b(?=(?:\s+[\w'-]+){0,4}?\s+(?:novels?|books?|titles?)\b)",
                  q_lower)
    top_k = int(m.group(1) or m.group(2)) if m else 5

    return {"author": author, "genre": genre, "top_k": top_k}
