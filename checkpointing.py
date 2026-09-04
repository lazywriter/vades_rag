# -*- coding: utf-8 -*-
"""
checkpointing.py

Small shared helpers for resumable long-running steps. HTML cleaning across a
large corpus, style-feature extraction, and USE embedding extraction all use
this so a crash or interrupt partway through doesn't mean starting over.

Training itself is checkpointed separately in train_vades.py via
torch.save/torch.load (needs optimizer/scheduler state, not just plain
picklable arrays) but follows the same atomic-write pattern defined here.
"""

import os
import joblib


def atomic_write(path: str, write_fn) -> None:
    """Writes to a temp file then renames into place. Without this, a
    crash/kill exactly mid-write leaves a truncated, unreadable checkpoint —
    which is worse than no checkpoint, since it fails on resume too."""
    tmp_path = path + ".tmp"
    write_fn(tmp_path)
    os.replace(tmp_path, path)


def save_partial(path: str, state) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    atomic_write(path, lambda tmp: joblib.dump(state, tmp))


def load_partial(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def clear_partial(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
