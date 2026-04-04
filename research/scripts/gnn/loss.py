"""
loss.py
───────
Loss function for the GAT pose estimator.

Three terms
───────────
  coord_loss  — visibility-masked L2 distance between predicted and GT
                joint coordinates, weighted by JOINT_WEIGHTS.
  bone_loss   — penalises deviation of predicted bone lengths from GT.
  vis_loss    — binary cross-entropy between predicted and GT visibility.

Visibility masking
──────────────────
  Invisible joints still participate fully in GAT message-passing.
  They are excluded from the coord and bone loss terms via an
  effective visibility mask:

      eff_vis = min(vis_pred, vis_gt)

  This mirrors the pose_distance metric used in evaluation: both the
  model and the ground truth must agree a joint is visible before its
  error contributes to the loss.
"""

import torch
import torch.nn.functional as F

from constants import BONES, JOINT_WEIGHTS


def gat_loss(
    coords_pred: torch.Tensor,
    coords_gt:   torch.Tensor,
    vis_pred:    torch.Tensor,
    vis_gt:      torch.Tensor,
    coord_w:     float = 1.0,
    bone_w:      float = 0.5,
    vis_w:       float = 0.25,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the combined pose estimation loss.

    Args:
        coords_pred : (B, J, 3) — model output (raw, no sigmoid).
        coords_gt   : (B, J, 3) — ground-truth normalised coordinates.
        vis_pred    : (B, J)    — model visibility prediction in (0, 1).
        vis_gt      : (B, J)    — ground-truth visibility scores in (0, 1).
        coord_w     : Weight for the coordinate L2 loss term.
        bone_w      : Weight for the bone length loss term.
        vis_w       : Weight for the visibility BCE loss term.

    Returns:
        total_loss  : Scalar tensor.
        breakdown   : Dict with keys "coord", "bone", "vis".
    """
    B, J, _ = coords_pred.shape
    device   = coords_pred.device
    jw       = JOINT_WEIGHTS.to(device)   # (J,)
    vis_gt_c = vis_gt.clamp(0.0, 1.0)

    # ── 1. Visibility BCE ──────────────────────────────────────────────────────
    vis_loss = F.binary_cross_entropy(vis_pred, vis_gt_c, reduction="mean")

    # ── 2. Effective visibility mask ──────────────────────────────────────────
    #   eff_vis[b, j] > 0 only when BOTH model and GT agree joint is visible.
    eff_vis = torch.min(vis_pred, vis_gt_c)   # (B, J)

    # ── 3. Coordinate L2 loss (visibility-masked) ─────────────────────────────
    dist       = torch.norm(coords_pred - coords_gt, dim=-1)   # (B, J)
    dist       = dist * jw.unsqueeze(0)                        # joint importance
    denom      = eff_vis.sum(dim=-1) + 1e-6                    # (B,)
    coord_loss = (dist * eff_vis).sum(dim=-1) / denom          # (B,)
    coord_loss = coord_loss.mean()

    # ── 4. Bone length loss ────────────────────────────────────────────────────
    bone_loss   = torch.zeros(1, device=device)
    valid_bones = 0

    for (a, b) in BONES:
        bone_vis = eff_vis[:, a] * eff_vis[:, b]   # (B,) — both endpoints visible
        if bone_vis.sum() < 1e-6:
            continue

        pred_len   = torch.norm(coords_pred[:, a, :] - coords_pred[:, b, :], dim=-1)
        target_len = torch.norm(coords_gt[:, a, :]   - coords_gt[:, b, :],   dim=-1)

        bone_loss += (
            ((pred_len - target_len) ** 2) * bone_vis
        ).sum() / (bone_vis.sum() + 1e-6)
        valid_bones += 1

    if valid_bones:
        bone_loss = bone_loss / valid_bones

    bone_loss = bone_loss.squeeze()

    # ── 5. Total ───────────────────────────────────────────────────────────────
    total = coord_w * coord_loss + bone_w * bone_loss + vis_w * vis_loss

    return total, {
        "coord": coord_loss.item(),
        "bone":  bone_loss.item(),
        "vis":   vis_loss.item(),
    }