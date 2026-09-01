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
        json.JSONDecodeError if the model didn't return valid JSON (rare, but
        don't blindly trust LLM output in a pipeline — check for this upstream).
    """
    client = client or anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        system=PARSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    parsed = json.loads(text)
    if not parsed.get("top_k"):
        parsed["top_k"] = 5
    return parsed


def parse_query_fallback(query: str, known_authors: List[str],
                          known_genres: List[str]) -> Dict:
    """
    Best-effort parse without any API call — plain string matching against
    known vocab. Use as an offline fallback, or to cross-check the LLM
    parser's output for a quick sanity test.

    Note: this does simple substring matching, so it works best when
    known_authors/known_genres are checked longest-first (to avoid e.g.
    matching "Poe" inside a longer name first) and won't catch paraphrases
    the LLM parser would (e.g. "detective fiction" -> "Mystery").
    """
    q_lower = query.lower()

    author = next((a for a in sorted(known_authors, key=len, reverse=True)
                   if a.lower() in q_lower), None)
    genre = next((g for g in sorted(known_genres, key=len, reverse=True)
                  if g.lower() in q_lower), None)

    m = re.search(r"\btop\s*(\d+)\b|\b(\d+)\s+novels?\b", q_lower)
    top_k = int(m.group(1) or m.group(2)) if m else 5

    return {"author": author, "genre": genre, "top_k": top_k}
