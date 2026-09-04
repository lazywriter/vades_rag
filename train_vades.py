# -*- coding: utf-8 -*-
"""
train_vades.py

Trains VADES_Fuse (document + author style embeddings) on the Project
Gutenberg corpus and exports everything the RAG pipeline
(style_faiss_index.py / style_recommender.py / pipeline.py) needs to run:

    <output_dir>/vades_faiss_store/     chunk-level FAISS index + metadata
                                         (this is "the database" — same corpus
                                         used for training now backs retrieval)
    <output_dir>/author_vectors.npy     learned per-author style embeddings
    <output_dir>/author2id.json         author name -> row index into author_vectors
    <output_dir>/checkpoints/best.pth   VADES_Fuse weights only (best val loss so far)
    <output_dir>/checkpoints/latest.pth full resumable training state (model +
                                         optimizer + scheduler + epoch + loss
                                         history) — see "Checkpointing" below
    <output_dir>/style_scaler.joblib    mean/std used to scale handcrafted style
                                         features (needed to featurize any NEW
                                         text consistently at inference time)

Fixes/additions vs. the original notebook (vades_main.py):
  - DocDataset/DataLoader ordering: the notebook created train_loader before
    DocDataset was even defined — it only ran because Colab executes cells
    out of the order they're printed in. This script defines things in real
    dependency order, so it also works as a plain linear script.
  - Removed the placeholder `Y_style = np.random.rand(512, 300)` from the
    stylistic-feature-prediction eval cell in the notebook — that was random
    dummy data standing in for real style features and was never reconnected
    to anything real. That eval isn't reproduced here; see README if you want
    it rebuilt against real style_feats.
  - Colab-specific bits (drive.mount, `!pip install` magics) removed — see
    requirements.txt for the equivalent installs.
  - HTML cleaning now strips Project Gutenberg's boilerplate structurally
    (by <section id="pg-header">/<section id="pg-footer">) before flattening
    to text, rather than only via regex afterwards — verified against your
    actual pg11870-images.html sample, where the old regex-only approach left
    ~500 words of license text and metadata in the first chunk of every book.
    See gutenberg_data.py.
  - Checkpointing throughout: corpus cleaning, style-feature extraction, and
    USE embedding extraction each resume from partial progress instead of
    starting over (see gutenberg_data.py / style_features.py / embeddings.py).
    The training loop itself now also checkpoints every epoch and resumes
    automatically if interrupted — see "Checkpointing" below.

Checkpointing (training loop)
------------------------------
After every epoch, full training state (model, optimizer, scheduler, epoch
number, best validation loss, loss history) is saved to
checkpoints/latest.pth. If that file exists the next time you run this
script with the same output_dir, training resumes from the next epoch
automatically — no flag needed. To ignore an existing checkpoint and retrain
from scratch instead, pass --fresh_start.

A resumed run is checked against the architecture/data settings the
checkpoint was originally trained with (use_dim, style_in_dim, out_dim,
hidden, n_authors) — a mismatch (e.g. you changed --hidden, or the corpus
gained new authors) raises a clear error rather than silently loading
mismatched weights or crashing deep inside torch's state_dict loading.

Usage:
    python train_vades.py --books_dir /path/to/gutenberg_books --output_dir ./vades_run
    python train_vades.py --books_dir /path/to/gutenberg_books --output_dir ./vades_run \\
        --genre_csv book_genres.csv --epochs 15
    # interrupted? just run the exact same command again — it resumes.
    python train_vades.py --books_dir /path/to/gutenberg_books --output_dir ./vades_run --fresh_start
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpointing import atomic_write
from gutenberg_data import load_or_build_chunks_df, book_level_split
from style_features import load_spacy_model, cache_features, fit_scaler, apply_scaler, save_scaler
from embeddings import load_use_model, cache_use_embeddings
from vades_model import VADES_Fuse, DocDataset, contrastive_cosine_loss, sample_negatives
from style_faiss_index import build_chunk_metadata, build_faiss_index, save_index, load_genre_map


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--books_dir", required=True,
                   help="Root folder of Gutenberg books: one subfolder per author, .html files inside")
    p.add_argument("--output_dir", default="./vades_run",
                   help="Where caches, checkpoints, and the final RAG artifacts get written")
    p.add_argument("--genre_csv", default=None,
                   help="Optional book_id,genre CSV (see book_genres_template.csv). If omitted, "
                        "books are indexed with empty genre tags and can be backfilled later — "
                        "re-run just the export step (see README) once tags exist.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dim", type=int, default=300)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--k_neg", type=int, default=5)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh_start", action="store_true",
                   help="Ignore any existing checkpoints/latest.pth and retrain from epoch 1.")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = os.path.join(args.output_dir, "feat_cache")
    split_dir = os.path.join(args.output_dir, "splits")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    for d in [cache_dir, split_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # 1. Load & chunk the Gutenberg corpus, book-level train/val/test split
    # ---------------------------------------------------------------
    chunks_csv = os.path.join(args.output_dir, "gutenberg_chunks.csv")
    df = load_or_build_chunks_df(args.books_dir, chunks_csv)
    train_df, val_df, test_df = book_level_split(df, split_dir, random_state=args.seed)

    # ---------------------------------------------------------------
    # 2. Handcrafted style features (300-D), scaled on TRAIN stats only
    # ---------------------------------------------------------------
    nlp = load_spacy_model()
    train_feats = cache_features(train_df, os.path.join(cache_dir, "train_feats.joblib"), nlp)
    val_feats = cache_features(val_df, os.path.join(cache_dir, "val_feats.joblib"), nlp)
    test_feats = cache_features(test_df, os.path.join(cache_dir, "test_feats.joblib"), nlp)

    style_mean, style_std = fit_scaler(train_feats)
    train_feats = apply_scaler(train_feats, style_mean, style_std)
    val_feats = apply_scaler(val_feats, style_mean, style_std)
    test_feats = apply_scaler(test_feats, style_mean, style_std)
    save_scaler(style_mean, style_std, os.path.join(args.output_dir, "style_scaler.joblib"))

    # ---------------------------------------------------------------
    # 3. USE embeddings (512-D), cached per split
    # ---------------------------------------------------------------
    use_model = load_use_model()
    train_embs = cache_use_embeddings(train_df, os.path.join(cache_dir, "train_use.joblib"), use_model)
    val_embs = cache_use_embeddings(val_df, os.path.join(cache_dir, "val_use.joblib"), use_model)
    test_embs = cache_use_embeddings(test_df, os.path.join(cache_dir, "test_use.joblib"), use_model)

    # ---------------------------------------------------------------
    # 4. Author id mapping (shared across all splits)
    # ---------------------------------------------------------------
    authors = sorted(pd.concat([train_df["author"], val_df["author"], test_df["author"]]).unique())
    author2id = {a: i for i, a in enumerate(authors)}
    num_authors = len(author2id)
    print(f"Authors: {num_authors}")

    train_df["author_id"] = train_df["author"].map(author2id)
    val_df["author_id"] = val_df["author"].map(author2id)
    test_df["author_id"] = test_df["author"].map(author2id)

    # ---------------------------------------------------------------
    # 5. Datasets/loaders — defined BEFORE use (fixes the notebook's ordering bug)
    # ---------------------------------------------------------------
    train_ds = DocDataset(train_embs, train_feats, train_df["author_id"].values)
    val_ds = DocDataset(val_embs, val_feats, val_df["author_id"].values)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_ds = DocDataset(test_embs, test_feats, test_df["author_id"].values)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ---------------------------------------------------------------
    # 6. Model, optimizer, training loop (checkpointed + resumable)
    # ---------------------------------------------------------------
    model = VADES_Fuse(use_dim=train_embs.shape[1], style_in_dim=train_feats.shape[1],
                        out_dim=args.out_dim, hidden=args.hidden, n_authors=num_authors,
                        dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    arch_signature = {
        "use_dim": int(train_embs.shape[1]),
        "style_in_dim": int(train_feats.shape[1]),
        "out_dim": args.out_dim,
        "hidden": args.hidden,
        "n_authors": num_authors,
    }

    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    latest_ckpt = os.path.join(ckpt_dir, "latest.pth")

    best_val = float("inf")
    loss_history = {"train": [], "val": []}
    start_epoch = 1

    if os.path.exists(latest_ckpt) and not args.fresh_start:
        print(f"Found existing checkpoint at {latest_ckpt} — resuming.")
        resumed_epoch, best_val, loss_history = _load_training_checkpoint(
            latest_ckpt, model, optimizer, scheduler, device, arch_signature
        )
        start_epoch = resumed_epoch + 1
        print(f"Resuming from epoch {start_epoch} (best_val so far: {best_val:.4f})")
    elif os.path.exists(latest_ckpt) and args.fresh_start:
        print(f"--fresh_start given: ignoring existing checkpoint at {latest_ckpt}.")

    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}/{args.epochs} — nothing left "
              f"to train. Raise --epochs to continue training, or pass --fresh_start to retrain.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        for use_emb, style_feat, aid in train_loader:
            use_emb, style_feat, aid = use_emb.to(device), style_feat.to(device), aid.to(device)
            neg_aids, neg_style = sample_negatives(aid, style_feat, num_authors, args.k_neg, device)

            loss, a_loss, s_loss = contrastive_cosine_loss(
                model, use_emb, style_feat, aid, neg_aids, neg_style,
                margin=args.margin, alpha=args.alpha
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for use_emb, style_feat, aid in val_loader:
                use_emb, style_feat, aid = use_emb.to(device), style_feat.to(device), aid.to(device)
                neg_aids, neg_style = sample_negatives(aid, style_feat, num_authors, args.k_neg, device)
                vloss, _, _ = contrastive_cosine_loss(
                    model, use_emb, style_feat, aid, neg_aids, neg_style,
                    margin=args.margin, alpha=args.alpha
                )
                val_losses.append(vloss.item())

        avg_train, avg_val = float(np.mean(train_losses)), float(np.mean(val_losses))
        loss_history["train"].append(avg_train)
        loss_history["val"].append(avg_val)
        print(f"[Epoch {epoch}/{args.epochs}] train={avg_train:.4f} val={avg_val:.4f}")
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> new best, saved {best_ckpt}")

        # Full resumable state, every epoch — this is what makes a crash or
        # interrupt partway through a long training run non-fatal.
        _save_training_checkpoint(latest_ckpt, epoch, model, optimizer, scheduler,
                                   best_val, loss_history, arch_signature)

    # ---------------------------------------------------------------
    # 7. Reload best checkpoint, quick sanity check before trusting these
    #    embeddings to build a recommender on
    # ---------------------------------------------------------------
    if not os.path.exists(best_ckpt):
        raise RuntimeError(
            f"No {best_ckpt} was ever saved (no epoch improved on the initial best_val=inf). "
            f"This shouldn't normally happen — check the training loop actually ran at least "
            f"one epoch (start_epoch/--epochs) and that val loss isn't NaN."
        )
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    _report_nearest_author_accuracy(model, train_loader, test_loader, num_authors, device)

    # ---------------------------------------------------------------
    # 8. Export embeddings for EVERY chunk (all splits) + build the FAISS
    #    store — this is "the database": same corpus used for training now
    #    backs retrieval.
    # ---------------------------------------------------------------
    all_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    all_use = np.vstack([train_embs, val_embs, test_embs])
    all_style = np.vstack([train_feats, val_feats, test_feats])

    all_doc_embeddings = _extract_embeddings(model, all_use, all_style, device)

    genre_map = load_genre_map(args.genre_csv) if args.genre_csv else {}
    metadata_df = build_chunk_metadata(
        book_ids=all_df["book_id"].values,
        titles=all_df["file_name"].values,
        authors=all_df["author"].values,
        texts=all_df["text"].values,
        genre_map=genre_map,
    )
    faiss_index = build_faiss_index(all_doc_embeddings, already_normalized=True)
    save_index(faiss_index, metadata_df, os.path.join(args.output_dir, "vades_faiss_store"))

    author_vectors = model.authors.weight.detach().cpu().numpy()
    np.save(os.path.join(args.output_dir, "author_vectors.npy"), author_vectors)
    with open(os.path.join(args.output_dir, "author2id.json"), "w") as f:
        json.dump(author2id, f, indent=2)

    print(f"\nDone. RAG artifacts written to {args.output_dir}:")
    print(f"  - vades_faiss_store/   (FAISS index + chunk metadata, {len(metadata_df)} chunks)")
    print(f"  - author_vectors.npy   {author_vectors.shape}")
    print(f"  - author2id.json       ({num_authors} authors)")
    print(f"  - checkpoints/best.pth")
    print(f"  - checkpoints/latest.pth (resumable training state)")
    print(f"  - style_scaler.joblib  (needed to featurize any NEW text at inference time)")
    if not args.genre_csv:
        print(f"\nNOTE: no --genre_csv was given, so every chunk was indexed with an empty "
              f"genre list. Genre-filtered search will return nothing until you tag books "
              f"(see book_genres_template.csv) and rebuild the index (Step 8 can be re-run "
              f"on its own — see README).")


def _save_training_checkpoint(path, epoch, model, optimizer, scheduler, best_val,
                               loss_history, arch_signature):
    """Full resumable training state — everything needed to pick up training
    at the next epoch as if it had never stopped. Atomic write (see
    checkpointing.py) so a crash mid-save can't leave a corrupt checkpoint
    that then fails to load on the next resume attempt."""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val": best_val,
        "loss_history": loss_history,
        "arch": arch_signature,
    }
    atomic_write(path, lambda tmp: torch.save(state, tmp))


def _load_training_checkpoint(path, model, optimizer, scheduler, device, arch_signature):
    """Loads a training checkpoint IN PLACE into model/optimizer/scheduler.
    Refuses to load if the checkpoint's architecture/data settings
    (use_dim, style_in_dim, out_dim, hidden, n_authors) don't match the
    current run's — silently loading mismatched weights would either crash
    deep inside torch with a confusing shape error, or worse, load
    successfully into the wrong-shaped layers and train garbage."""
    state = torch.load(path, map_location=device)
    saved_arch = state.get("arch", {})
    mismatches = {k: (saved_arch.get(k), v) for k, v in arch_signature.items()
                  if saved_arch.get(k) != v}
    if mismatches:
        raise ValueError(
            f"Checkpoint at {path} was trained with different settings than this run "
            f"(saved vs. current): {mismatches}. This usually means the corpus changed "
            f"(different n_authors) or a --hidden/--out_dim flag changed between runs. "
            f"Either match the original settings again, or pass --fresh_start to discard "
            f"the checkpoint and retrain from scratch."
        )
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    return state["epoch"], state["best_val"], state["loss_history"]


def _extract_embeddings(model, use_arr, style_arr, device, batch=256):
    model.eval()
    out_chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, len(use_arr), batch), desc="Extracting final embeddings"):
            use_b = torch.from_numpy(use_arr[i:i + batch]).float().to(device)
            style_b = torch.from_numpy(style_arr[i:i + batch]).float().to(device)
            out = model(use_b, style_b)
            out = F.normalize(out, dim=1)   # cosine-ready — style_faiss_index assumes this
            out_chunks.append(out.cpu().numpy())
    return np.vstack(out_chunks)


def _report_nearest_author_accuracy(model, train_loader, test_loader, num_authors, device):
    """Cheap sanity check: does nearest-centroid classification on held-out
    chunks beat random chance? Not a substitute for the full evaluation
    suite in the original notebook (T-SNE, confusion matrix, per-group MSE)
    — just a quick 'did training actually work' signal before these
    embeddings get used to build a recommender."""
    author_means = defaultdict(list)
    with torch.no_grad():
        for use_emb, style_feat, aid in train_loader:
            use_emb, style_feat = use_emb.to(device), style_feat.to(device)
            doc_out = model(use_emb, style_feat)
            for i, a in enumerate(aid.numpy()):
                author_means[a].append(doc_out[i].cpu().numpy())
    author_means = {a: np.mean(v, axis=0) for a, v in author_means.items()}

    correct, total = 0, 0
    with torch.no_grad():
        for use_emb, style_feat, aid in test_loader:
            use_emb, style_feat = use_emb.to(device), style_feat.to(device)
            doc_out = model(use_emb, style_feat).cpu().numpy()
            for i, true_a in enumerate(aid.numpy()):
                dists = {a: np.linalg.norm(doc_out[i] - c) for a, c in author_means.items()}
                pred_a = min(dists, key=dists.get)
                correct += int(pred_a == true_a)
                total += 1

    acc = 100 * correct / max(1, total)
    print(f"\nSanity check — nearest-author-centroid accuracy on test set: "
          f"{acc:.2f}% ({correct}/{total}), random baseline: {100 / num_authors:.2f}%")


if __name__ == "__main__":
    main()
