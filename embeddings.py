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
that's what's kept here.
"""

import os

import joblib
import numpy as np
from tqdm import tqdm

from checkpointing import save_partial, load_partial, clear_partial

DEFAULT_USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"


def load_use_model(url: str = DEFAULT_USE_URL):
    import tensorflow_hub as hub
    print(f"Loading Universal Sentence Encoder from {url} ...")
    model = hub.load(url)
    print("USE loaded (512-D output).")
    return model


def cache_use_embeddings(df, path: str, use_model, text_col: str = "text",
                          batch_size: int = 64, checkpoint_every_batches: int = 20) -> np.ndarray:
    """
    Checkpointed: USE inference over a large corpus can run long, so progress
    is saved every `checkpoint_every_batches` batches to "<path>.partial.joblib".
    An interrupted run resumes from the next un-embedded batch instead of
    recomputing everything already done.
    """
    if os.path.exists(path):
        print(f"Loading cached USE embeddings from {os.path.basename(path)}")
        return joblib.load(path)

    texts = df[text_col].tolist()
    starts = list(range(0, len(texts), batch_size))

    ckpt_path = path + ".partial.joblib"
    state = load_partial(ckpt_path)
    if state is not None:
        all_embs = state["all_embs"]
        start_batch_idx = state["next_batch_idx"]
        print(f"Resuming USE embedding extraction from checkpoint: "
              f"{start_batch_idx}/{len(starts)} batches done")
    else:
        all_embs, start_batch_idx = [], 0

    since_checkpoint = 0
    for b in tqdm(range(start_batch_idx, len(starts)), desc="USE embeddings",
                  initial=start_batch_idx, total=len(starts)):
        i = starts[b]
        batch = texts[i:i + batch_size]
        emb = use_model(batch).numpy()
        all_embs.append(emb)
        since_checkpoint += 1
        if since_checkpoint >= checkpoint_every_batches:
            save_partial(ckpt_path, {"all_embs": all_embs, "next_batch_idx": b + 1})
            since_checkpoint = 0

    all_embs = np.vstack(all_embs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(all_embs, path)
    clear_partial(ckpt_path)
    return all_embs
