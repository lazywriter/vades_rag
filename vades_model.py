# -*- coding: utf-8 -*-
"""
vades_model.py

VADES_Fuse: the fusion encoder that combines a document's USE embedding
(semantic content) with its handcrafted style vector into a single 300-D
embedding, trained via contrastive metric learning so a document's fused
embedding sits close to (a) its author's learned embedding and (b) its own
style vector, and far from sampled negative authors/styles.

Flagging one naming thing: this is a deterministic encoder with a hinge/
contrastive loss — there's no stochastic latent or KL term, so it isn't
literally a Variational Information Bottleneck even though the project
description referenced "VIB." If you actually want a true VIB (reparameterized
stochastic bottleneck + KL regularizer against a prior), that's a model
change worth doing deliberately rather than folding in silently here — let me
know if that's what you meant and I'll add it as its own variant.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class VADES_Fuse(nn.Module):
    def __init__(self, use_dim: int = 512, style_in_dim: int = 300, out_dim: int = 300,
                 hidden: int = 1024, n_authors: int = 10, dropout: float = 0.5):
        """
        use_dim: dimension of USE input (512)
        style_in_dim: dimension of handcrafted style vector (300)
        out_dim: output embedding dimension (final fused doc/author embedding, 300)
        n_authors: size of the learned per-author embedding table
        """
        super().__init__()
        self.input_dim = use_dim + style_in_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
        self.authors = nn.Embedding(n_authors, out_dim)
        nn.init.xavier_uniform_(self.authors.weight)

    def forward(self, use_emb: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        if use_emb.dim() == 1:
            use_emb = use_emb.unsqueeze(0)
        if style_feat.dim() == 1:
            style_feat = style_feat.unsqueeze(0)
        x = torch.cat([use_emb, style_feat], dim=1)
        return self.encoder(x)

    def get_document_embedding(self, use_emb, style_feat):
        return self.forward(use_emb, style_feat)


class DocDataset(Dataset):
    """Pairs each chunk's (USE embedding, scaled style vector, author id)."""

    def __init__(self, embs: np.ndarray, feats: np.ndarray, author_ids: np.ndarray):
        self.embs = embs
        self.feats = feats
        self.author_ids = author_ids

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, i):
        return (
            torch.tensor(self.embs[i], dtype=torch.float32),
            torch.tensor(self.feats[i], dtype=torch.float32),
            torch.tensor(self.author_ids[i], dtype=torch.long)
        )


def sample_negatives(aid: torch.Tensor, style_feat: torch.Tensor, num_authors: int,
                      k_neg: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized negative sampling: K random authors != the true author per
    example, plus K batch-permuted style vectors as negative styles (cheap
    approximation — reuses other examples' real style vectors as negatives
    rather than synthesizing new ones)."""
    B = aid.size(0)
    neg_aids = torch.randint(0, num_authors, (B, k_neg), device=device)
    mask = (neg_aids == aid.unsqueeze(1))
    while mask.any():
        neg_aids[mask] = torch.randint(0, num_authors, (int(mask.sum().item()),), device=device)
        mask = (neg_aids == aid.unsqueeze(1))

    idx = torch.randperm(B, device=device)
    neg_style = torch.stack([style_feat[idx.roll(i + 1)] for i in range(k_neg)], dim=1)
    return neg_aids, neg_style


def contrastive_cosine_loss(model: VADES_Fuse, use_emb, style_feat, author_ids,
                             neg_author_ids, neg_style_feats,
                             margin: float = 0.2, alpha: float = 0.8):
    """
    Unchanged from vades_main.py. Hinge loss pulling a document's fused
    embedding toward (a) its author's learned embedding and (b) its own style
    vector, and pushing it at least `margin` away from K sampled negative
    authors/styles. `alpha` weights style-loss vs. author-loss.

    Returns: (total_loss, author_loss.detach(), style_loss.detach())
    """
    doc_out = model(use_emb, style_feat)            # [B, out_dim]
    doc_n = F.normalize(doc_out, dim=1)

    pos_author_emb = model.authors(author_ids)       # [B, out_dim]
    pos_author_n = F.normalize(pos_author_emb, dim=1)
    pos_style_n = F.normalize(style_feat, dim=1)

    pos_sim_author = (doc_n * pos_author_n).sum(dim=1)   # [B]
    pos_sim_style = (doc_n * pos_style_n).sum(dim=1)     # [B]

    B, K = neg_author_ids.shape
    neg_author_emb = model.authors(neg_author_ids.view(-1)).view(B, K, -1)   # [B,K,D]
    neg_author_n = F.normalize(neg_author_emb, dim=2)
    neg_style_n = F.normalize(neg_style_feats, dim=2)

    doc_n_exp = doc_n.unsqueeze(1)   # [B,1,D]
    neg_sim_author = (doc_n_exp * neg_author_n).sum(dim=2)   # [B,K]
    neg_sim_style = (doc_n_exp * neg_style_n).sum(dim=2)     # [B,K]

    author_loss = F.relu(neg_sim_author - pos_sim_author.unsqueeze(1) + margin).mean()
    style_loss = F.relu(neg_sim_style - pos_sim_style.unsqueeze(1) + margin).mean()

    total = (1 - alpha) * author_loss + alpha * style_loss
    return total, author_loss.detach(), style_loss.detach()
