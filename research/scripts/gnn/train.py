"""
train.py
────────
Training entry point for the GAT sketch-to-pose model.

Usage
─────
  # Train from scratch
  python train.py --chunk_dir /path/to/GAT_cache

  # Resume from a checkpoint
  python train.py --chunk_dir /path/to/GAT_cache --resume gat_pose_epoch_50.pth

  # Override any hyperparameter
  python train.py --chunk_dir /path/to/GAT_cache --lr 1e-4 --epochs 200 --batch_size 32

Run `python train.py --help` for all options.
"""

import argparse
import os
import random
import re

import numpy as np
import torch
import tqdm

from dataset import build_loaders
from loss import gat_loss
from model import SketchToCoordGAT


# Shared project data root
PATH = "C:/Users/leahz/Documents/ATC/pose-project/data"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _epoch_from_filename(filename: str) -> int:
    """Parse epoch number from a checkpoint filename, e.g. gat_pose_epoch_50.pth → 50."""
    m = re.search(r"epoch_(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else 0


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, val_loader, device, vis_threshold: float = 0.4) -> float:
    """
    Compute mean L2 distance on visible joints over the validation set.
    Returns the mean distance (lower is better).
    """
    model.eval()
    total_dist = 0.0
    total_n    = 0

    with torch.no_grad():
        for x, coords_gt, vis_gt in val_loader:
            x         = x.to(device, non_blocking=True)
            coords_gt = coords_gt.to(device, non_blocking=True)
            vis_gt    = vis_gt.to(device, non_blocking=True)

            coords_pred, _ = model(x)
            mask = vis_gt > vis_threshold

            if mask.any():
                d = torch.norm(coords_pred - coords_gt, dim=-1)   # (B, J)
                total_dist += (d * mask.float()).sum().item()
                total_n    += mask.sum().item()

    return total_dist / max(total_n, 1)


# ── Main training loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> SketchToCoordGAT:
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        chunk_dir   = args.chunk_dir,
        max_chunks  = args.max_chunks,
        batch_size  = args.batch_size,
        train_ratio = args.train_ratio,
        seed        = args.seed,
        num_workers = args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SketchToCoordGAT(
        img_feat_dim  = args.img_feat_dim,
        joint_emb_dim = args.joint_emb_dim,
        gat_hidden    = args.gat_hidden,
        gat_heads     = args.gat_heads,
        gat_layers    = args.gat_layers,
        dropout       = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Smart optimiser: separate LR groups ───────────────────────────────────
    #   CNN backbone learns slower (fine-tuning behaviour from scratch too)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.cnn.parameters(),       "lr": args.lr * 0.5, "name": "cnn"},
            {"params": model.joint_emb.parameters(), "lr": args.lr,       "name": "emb"},
            {"params": model.gat_list.parameters(),  "lr": args.lr,       "name": "gat"},
            {"params": model.head.parameters(),      "lr": args.lr,       "name": "head"},
        ],
        weight_decay = args.weight_decay,
    )

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0

    if args.resume:
        resume_path = os.path.join(args.save_dir, args.resume)
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            print(f"  Missing keys:    {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as e:
                print(f"  Optimizer not restored (architecture changed?): {e}")

        start_epoch = int(ckpt.get("epoch", 0)) or _epoch_from_filename(args.resume)
        print(f"  Resuming from epoch {start_epoch + 1} / {args.epochs}")

    # Force the requested LR onto all groups (useful when changing lr on resume)
    for pg in optimizer.param_groups:
        pg["lr"] = args.lr if pg.get("name") != "cnn" else args.lr * 0.5

    # ── LR scheduler ──────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    if start_epoch > 0:
        # Fast-forward scheduler state without running extra epochs
        scheduler.last_epoch = start_epoch - 1
        scheduler.base_lrs   = [pg["lr"] for pg in optimizer.param_groups]

    if start_epoch >= args.epochs:
        print("Checkpoint already at/past target epoch count. Nothing to train.")
        return model

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_dist = float("inf")

    for epoch in tqdm.tqdm(range(start_epoch, args.epochs), desc="Epochs"):
        model.train()
        totals = {"loss": 0.0, "coord": 0.0, "bone": 0.0, "vis": 0.0}

        for bi, (x, coords_gt, vis_gt) in enumerate(train_loader):
            x         = x.to(device, non_blocking=True)
            coords_gt = coords_gt.to(device, non_blocking=True)
            vis_gt    = vis_gt.to(device, non_blocking=True)

            coords_pred, vis_pred = model(x)
            loss, bd = gat_loss(
                coords_pred, coords_gt, vis_pred, vis_gt,
                coord_w = args.coord_w,
                bone_w  = args.bone_w,
                vis_w   = args.vis_w,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            totals["loss"]  += loss.item()
            totals["coord"] += bd["coord"]
            totals["bone"]  += bd["bone"]
            totals["vis"]   += bd["vis"]

            if bi % args.log_every == 0:
                lr_gat = optimizer.param_groups[2]["lr"]
                print(
                    f"  E[{epoch+1}/{args.epochs}] "
                    f"B[{bi}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  "
                    f"coord={bd['coord']:.4f}  "
                    f"bone={bd['bone']:.4f}  "
                    f"vis={bd['vis']:.4f}  "
                    f"lr={lr_gat:.2e}"
                )

        scheduler.step()

        nb  = max(len(train_loader), 1)
        ep  = epoch + 1
        avg = {k: v / nb for k, v in totals.items()}

        print(
            f"\nEpoch {ep} summary | "
            f"loss={avg['loss']:.4f}  "
            f"coord={avg['coord']:.4f}  "
            f"bone={avg['bone']:.4f}  "
            f"vis={avg['vis']:.4f}"
        )

        # ── Validation ────────────────────────────────────────────────────────
        val_dist = validate(model, val_loader, device)
        print(f"  Val mean L2 (visible joints): {val_dist:.4f}")

        is_best = val_dist < best_val_dist
        if is_best:
            best_val_dist = val_dist

        # ── Checkpoint saving ─────────────────────────────────────────────────
        save_this_epoch = (ep % args.save_every == 0) or is_best

        if save_this_epoch:
            tag  = "best" if is_best else f"epoch_{ep}"
            path = os.path.join(args.save_dir, f"gat_pose_{tag}.pth")
            torch.save(
                {
                    "epoch":               ep,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dist":            val_dist,
                    "avg_loss":            avg["loss"],
                    "args":                vars(args),
                },
                path,
            )
            label = " ★ best" if is_best else ""
            print(f"  Saved checkpoint: {path}{label}")

        print()   # blank line between epochs

    print(f"\nTraining complete. Best val L2: {best_val_dist:.4f}")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GAT sketch-to-3D-pose model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--chunk_dir",  default=f"{PATH}/GAT_cache",
                   help="Directory containing chunk_*.pt dataset files.")
    p.add_argument("--save_dir",   default=f"{PATH}/models",
                   help="Directory to save checkpoints.")
    p.add_argument("--resume",     default="",
                   help="Checkpoint filename (inside save_dir) to resume from.")

    # Training
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--max_chunks",  type=int,   default=None,
                   help="Limit number of chunks used (None = all).")

    # Loss weights
    p.add_argument("--coord_w", type=float, default=1.0)
    p.add_argument("--bone_w",  type=float, default=0.5)
    p.add_argument("--vis_w",   type=float, default=0.25)

    # Logging / saving
    p.add_argument("--log_every",  type=int, default=10,
                   help="Print batch log every N batches.")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save a checkpoint every N epochs (best is always saved).")

    # Model architecture
    p.add_argument("--img_feat_dim",  type=int,   default=256)
    p.add_argument("--joint_emb_dim", type=int,   default=64)
    p.add_argument("--gat_hidden",    type=int,   default=128)
    p.add_argument("--gat_heads",     type=int,   default=4)
    p.add_argument("--gat_layers",    type=int,   default=3)
    p.add_argument("--dropout",       type=float, default=0.1)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())