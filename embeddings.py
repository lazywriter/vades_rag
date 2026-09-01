# -*- coding: utf-8 -*-
"""
embeddings.py

Universal Sentence Encoder (USE) embeddings for each text chunk — the
semantic half of VADES_Fuse's input (concatenated with the 300-D handcrafted
style vector from style_features.py before the fusion MLP).

Kept as TF-Hub USE to match the existing architecture's use_dim=512. Worth
flagging: the original vades_main.py notebook also `pip install`s
sentence-transformers at the top but never actually calls it — USE (via
tensorflow_hub) is what's actually wired into VADES_Fuse downstream, so
that's what's kept here. If you'd rather standardize on sentence-transformers
instead (fewer/lighter dependencies than the TF stack), that's a valid
alternative, but it changes use_dim and means retraining VADES_Fuse — flag it
if you want that swapped instead of just noted.
"""

import os

import joblib
import numpy as np
from tqdm import tqdm

DEFAULT_USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"


def load_use_model(url: str = DEFAULT_USE_URL):
    import tensorflow_hub as hub
    print(f"Loading Universal Sentence Encoder from {url} ...")
    model = hub.load(url)
    print("USE loaded (512-D output).")
    return model


def cache_use_embeddings(df, path: str, use_model, text_col: str = "text",
                          batch_size: int = 64) -> np.ndarray:
    if os.path.exists(path):
        print(f"Loading cached USE embeddings from {os.path.basename(path)}")
        return joblib.load(path)

    texts = df[text_col].tolist()
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="USE embeddings"):
        batch = texts[i:i + batch_size]
        emb = use_model(batch).numpy()
        all_embs.append(emb)
    all_embs = np.vstack(all_embs)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(all_embs, path)
    return all_embs
