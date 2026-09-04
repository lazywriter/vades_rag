# -*- coding: utf-8 -*-
"""
style_features.py

Handcrafted 300-D stylometric feature vector (punctuation, function words,
POS/NER ratios, char/word n-grams, readability, distributional stats) —
unchanged in method from vades_main.py's build_style_vector, just split into
its own module and given a scaler save/load helper so train_vades.py can
persist mean/std alongside the model checkpoint. You need that same mean/std
to featurize any new text consistently at inference time.
"""

import os
import re
import string
from collections import Counter

import joblib
import numpy as np
import spacy
import textstat
from tqdm import tqdm

from checkpointing import save_partial, load_partial, clear_partial

FUNCTION_WORDS = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
                  "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
                  "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
                  "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
                  "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take"]


def load_spacy_model(name: str = "en_core_web_sm"):
    """Loads the spaCy model, downloading it first if it isn't installed yet."""
    try:
        return spacy.load(name, disable=["parser"])
    except OSError:
        from spacy.cli import download
        print(f"spaCy model '{name}' not found locally — downloading...")
        download(name)
        return spacy.load(name, disable=["parser"])


def build_style_vector(text: str, nlp, dim: int = 300) -> np.ndarray:
    """Unchanged from vades_main.py's build_style_vector, aside from taking
    `nlp` as an explicit argument instead of relying on a module-level global
    (makes this module import-safe without side effects / model downloads)."""
    feats = []
    words = text.split()
    nw = len(words)

    # Basic (5)
    feats += [len(text), nw, max(1, len(re.split(r'[.!?]+', text))),
              nw / max(1, len(re.split(r'[.!?]+', text))),
              sum(len(w) for w in words) / max(1, nw)]

    # Punctuation (11)
    punc_counts = Counter(ch for ch in text if ch in string.punctuation)
    for k in ['.', ',', ';', '!', '?', "'", '"', ':', '-', '(', ')']:
        feats.append(punc_counts.get(k, 0) / max(1, nw))

    # Function words (60)
    word_lower = [w.lower() for w in re.findall(r"\w+", text)]
    wc = Counter(word_lower)
    feats += [wc.get(fw, 0) / max(1, len(word_lower)) for fw in FUNCTION_WORDS[:60]]

    # Lexical (3)
    types = set(word_lower)
    feats += [
        len(types) / max(1, len(word_lower)),
        sum(1 for _, c in wc.items() if c == 1) / max(1, nw),
        0.0
    ]

    # Syllables (2)
    try:
        sylls = textstat.syllable_count(text)
    except Exception:
        sylls = sum(max(1, sum(1 for c in w if c.lower() in 'aeiou')) for w in words)
    feats += [sylls, sylls / max(1, nw)]

    # Readability (4)
    try:
        feats += [
            textstat.flesch_reading_ease(text),
            textstat.flesch_kincaid_grade(text),
            textstat.smog_index(text),
            textstat.coleman_liau_index(text)
        ]
    except Exception:
        feats += [0.0] * 4

    # POS + NER (28)
    doc = nlp(text)
    pos_c = Counter(t.pos_ for t in doc)
    ner_c = Counter(e.label_ for e in doc.ents)

    for k in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'PROPN',
              'PART', 'INTJ', 'SYM', 'X', 'PUNCT']:
        feats.append(pos_c.get(k, 0) / max(1, len(doc)))

    for k in ['PERSON', 'NORP', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
              'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY']:
        feats.append(ner_c.get(k, 0) / max(1, len(doc)))

    # Char n-grams (50)
    s = re.sub(r'\s+', '', text.lower())
    char_ng = Counter()
    for n in [2, 3, 4]:
        for i in range(len(s) - n + 1):
            char_ng[s[i:i + n]] += 1

    top = [g for g, _ in char_ng.most_common(50)]
    tot = max(1, sum(char_ng.values()))
    feats += [char_ng.get(g, 0) / tot for g in top]
    if len(feats) < 85 + 50:
        feats += [0.0] * (85 + 50 - len(feats))

    # Word bigrams (50)
    word_bg = Counter()
    for i in range(len(word_lower) - 1):
        word_bg[" ".join(word_lower[i:i + 2])] += 1
    top = [g for g, _ in word_bg.most_common(50)]
    tot = max(1, sum(word_bg.values()))
    feats += [word_bg.get(g, 0) / tot for g in top]
    if len(feats) < 185:
        feats += [0.0] * (185 - len(feats))

    # Distributional (2)
    probs = np.array([v / max(1, sum(wc.values())) for v in wc.values()], dtype=float)
    feats += [
        -np.sum(probs * np.log2(probs + 1e-12)),
        max(wc.values()) / max(1, sum(wc.values())) if wc else 0.0
    ]

    # Extra (3)
    feats += [
        sum(1 for c in text if c.isdigit()) / max(1, len(text)),
        sum(1 for w in words if w.isupper()) / max(1, nw),
        sum(1 for w in word_lower if w in spacy.lang.en.stop_words.STOP_WORDS) / max(1, len(word_lower))
    ]

    feats = feats[:dim] + [0.0] * (dim - len(feats))
    return np.array(feats, dtype=np.float32)


def cache_features(df, path: str, nlp, text_col: str = "text",
                    checkpoint_every: int = 500) -> np.ndarray:
    """
    Checkpointed: on a large corpus this is one of the slower stages (spaCy
    parsing per chunk), so progress is saved every `checkpoint_every` chunks
    to "<path>.partial.joblib". An interrupted run resumes from there instead
    of recomputing every feature vector from scratch.
    """
    if os.path.exists(path):
        print(f"Loading cached features from {os.path.basename(path)}")
        return joblib.load(path)

    ckpt_path = path + ".partial.joblib"
    state = load_partial(ckpt_path)
    if state is not None:
        feats = state["feats"]
        start_idx = state["next_idx"]
        print(f"Resuming feature extraction from checkpoint: {start_idx}/{len(df)} done")
    else:
        feats, start_idx = [], 0

    texts = df[text_col].tolist()
    since_checkpoint = 0
    for i in tqdm(range(start_idx, len(texts)), desc="Style features",
                  initial=start_idx, total=len(texts)):
        feats.append(build_style_vector(texts[i], nlp))
        since_checkpoint += 1
        if since_checkpoint >= checkpoint_every:
            save_partial(ckpt_path, {"feats": feats, "next_idx": i + 1})
            since_checkpoint = 0

    feats = np.stack(feats)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(feats, path)
    clear_partial(ckpt_path)
    return feats


def fit_scaler(train_feats: np.ndarray):
    """Fit mean/std on TRAIN ONLY — never on val/test, to avoid leakage."""
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def save_scaler(mean, std, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"mean": mean, "std": std}, path)


def load_scaler(path):
    d = joblib.load(path)
    return d["mean"], d["std"]
