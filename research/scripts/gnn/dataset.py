"""
dataset.py
──────────
Dataset classes and DataLoader factory for the GAT pose model.

Expected on-disk format (produced by the data-creation notebook):
  Each .pt chunk file contains a dict with keys:
    "images"  – float32 tensor (N, 3, 64, 64)   normalised sketch
    "coords"  – float32 tensor (N, 33, 3)        normalised x,y,z
    "vis"     – float32 tensor (N, 33)           MediaPipe visibility score

Alternatively, if your existing cache uses "labels" (heatmaps) instead of
"coords", point to a converted cache produced by convert_cache.py.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from constants import JOINT_WEIGHTS


class ChunkedGATDataset(Dataset):
    """
    Streams (sketch, coords, visibility) from pre-built .pt chunk files.

    Args:
        chunk_dir:      Directory containing chunk_*.pt files.
        max_chunks:     Limit the number of chunks loaded (None = all).
        shuffle_chunks: Randomise chunk order at construction time.
        seed:           RNG seed for chunk shuffling.
    """

    def __init__(
        self,
        chunk_dir: str,
        max_chunks: Optional[int] = None,
        shuffle_chunks: bool = True,
        seed: int = 42,
    ):
        print(f"Initializing dataset from {chunk_dir}…")
        files = sorted(Path(chunk_dir).glob("chunk_*.pt"))
        if not files:
            raise FileNotFoundError(f"No chunk_*.pt files found in {chunk_dir}")

        if max_chunks is not None:
            files = files[:max_chunks]

        if shuffle_chunks:
            random.Random(seed).shuffle(files)

        self.files = [str(f) for f in files]

        # Pre-scan sizes so __len__ and index-mapping work without loading data
        self.sizes: list[int] = []
        self.cumul: list[int] = []
        total = 0
        for fp in self.files:
            chunk = torch.load(fp, map_location="cpu", weights_only=True)
            n = len(chunk["images"])
            self.sizes.append(n)
            total += n
            self.cumul.append(total)

        self._cache_idx: int = -1
        self._cache: Optional[dict] = None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_chunk(self, chunk_idx: int) -> None:
        if self._cache_idx != chunk_idx:
            self._cache = torch.load(
                self.files[chunk_idx], map_location="cpu", weights_only=True
            )
            self._cache_idx = chunk_idx

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.cumul[-1]

    def __getitem__(self, idx: int):
        for ci, cum in enumerate(self.cumul):
            if idx < cum:
                self._load_chunk(ci)
                local = idx - (self.cumul[ci - 1] if ci > 0 else 0)
                images = self._cache["images"][local]          # (3, 64, 64)
                coords = self._cache["coords"][local]          # (33, 3)
                vis    = self._cache["vis"][local]             # (33,)
                return images, coords, vis

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_all_visibility(self) -> np.ndarray:
        """
        Load only the visibility tensors from every chunk and concatenate.
        Used once to build the weighted sampler; result is cached to disk.
        """
        parts = []
        for fp in self.files:
            chunk = torch.load(fp, map_location="cpu", weights_only=True)
            parts.append(chunk["vis"].numpy())
        return np.concatenate(parts, axis=0).astype(np.float32)


def build_loaders(
    chunk_dir: str,
    max_chunks: Optional[int] = None,
    batch_size: int = 16,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a chunked cache directory.

    The training loader uses a visibility-weighted sampler so that samples
    with rare/hard-to-detect joints are not under-represented.

    Returns:
        (train_loader, val_loader)
    """
    ds = ChunkedGATDataset(chunk_dir, max_chunks=max_chunks, seed=seed)

    n_train = int(len(ds) * train_ratio)
    n_val   = len(ds) - n_train
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    # ── Visibility-weighted sampler (built once, cached to disk) ──────────────
    vis_cache_path = Path(chunk_dir) / "vis_all.npy"
    if vis_cache_path.exists():
        all_vis = np.load(vis_cache_path).astype(np.float32)
    else:
        print("Building visibility cache (one-time cost)…")
        all_vis = ds.get_all_visibility()
        np.save(vis_cache_path, all_vis)
        print(f"  Saved → {vis_cache_path}")

    train_idx  = np.asarray(train_ds.indices, dtype=np.int64)
    train_vis  = all_vis[train_idx]               # (n_train, 33)

    jcounts    = train_vis.sum(axis=0)            # (33,) — how often each joint is visible
    jweights   = 1.0 / (jcounts + 1e-6)          # rare joint → high weight
    sw         = (train_vis * jweights).sum(axis=1)
    sw         = np.clip(sw, 1e-8, None)
    sampler    = WeightedRandomSampler(
        weights     = torch.tensor(sw, dtype=torch.double),
        num_samples = len(sw),
        replacement = True,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = pin,
        persistent_workers = (num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin,
        persistent_workers = (num_workers > 0),
    )

    print(
        f"Dataset | chunks={len(ds.files)} | "
        f"total={len(ds)} | train={n_train} | val={n_val}"
    )
    return train_loader, val_loader