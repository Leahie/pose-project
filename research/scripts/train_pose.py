import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import re

from dataset import build_chunked_loaders

EDGES = [(11,12),(11,23),(12,24),(23,24),  # torso
         (23,25),(25,27),(24,26),(26,28),   # legs
         (11,13),(13,15),(12,14),(14,16)]   # arms

class SkeletonGNN(nn.Module):
    def __init__(self, num_joints=33, feat_dim=64):
        super().__init__()
        # Build adjacency list
        self.neighbors = {i: [i] for i in range(num_joints)}  # self-loops
        for a, b in EDGES:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
        
        self.msg = nn.Linear(feat_dim, feat_dim)
        self.update = nn.GRUCell(feat_dim, feat_dim)

    def forward(self, joint_feats, visibility=None):
        # visibility: (B, J) — 0/1 or soft confidence
        B, J, D = joint_feats.shape
        out = joint_feats.clone()
        for j in range(J):
            neighbors = self.neighbors[j]
            neighbor_feats = joint_feats[:, neighbors, :]  # (B, N, D)
            if visibility is not None:
                # mask out invisible neighbor contributions
                vis_weights = visibility[:, neighbors].unsqueeze(-1)  # (B, N, 1)
                neighbor_feats = neighbor_feats * vis_weights
            msgs = self.msg(neighbor_feats).mean(1)
            out[:, j, :] = self.update(msgs, joint_feats[:, j, :])
        return out
    
    
class PoseCNN(nn.Module):
    def __init__(self, num_keypoints=33):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        # Produce a rich feature map (not collapsed to heatmaps yet)
        self.pre_heatmap = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
        )  # output: (B, 64, 64, 64)

        # Initial per-joint heatmap prediction (before GNN refinement)
        self.initial_heatmap_head = nn.Conv2d(64, num_keypoints, 1)

        # Per-joint feature extractor: pool a small region around each joint
        self.joint_feat_dim = 64
        self.joint_feat_proj = nn.Linear(64, self.joint_feat_dim)

        # GNN refinement
        self.gnn = SkeletonGNN(num_joints=num_keypoints, feat_dim=self.joint_feat_dim)

        # Final heatmap refinement after GNN
        self.refine_head = nn.Conv2d(num_keypoints, num_keypoints, 3, padding=1)

        # Visibility head (unchanged)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_keypoints),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)                          # (B, 256, 16, 16)
        feat_map = self.pre_heatmap(enc)               # (B, 64, 64, 64)

        # Initial heatmaps
        init_heatmaps = torch.sigmoid(
            self.initial_heatmap_head(feat_map)
        )                                              # (B, J, 64, 64)

        # Extract per-joint features by sampling feat_map
        # at each joint's predicted location (soft-argmax)
        B, J, H, W = init_heatmaps.shape
        h_idx = torch.linspace(0, 1, H, device=x.device)
        w_idx = torch.linspace(0, 1, W, device=x.device)
        sm = F.softmax(init_heatmaps.view(B, J, -1) / 0.1, dim=-1).view(B, J, H, W)
        pred_x = (sm.sum(-2) * w_idx).sum(-1)         # (B, J)
        pred_y = (sm.sum(-1) * h_idx).sum(-1)         # (B, J)

        # Sample feat_map at predicted joint locations using grid_sample
        # grid_sample expects coords in [-1, 1]
        grid_x = pred_x * 2 - 1                       # (B, J)
        grid_y = pred_y * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, J, 1, 2)
        # feat_map: (B, 64, 64, 64) — sample for each joint
        joint_feats = F.grid_sample(
            feat_map, grid, align_corners=True
        ).squeeze(-1).permute(0, 2, 1)               # (B, J, 64)

        joint_feats = self.joint_feat_proj(joint_feats)  # (B, J, feat_dim)

        # GNN refinement — joints now talk to anatomical neighbors
        refined_feats = self.gnn(joint_feats)            # (B, J, feat_dim)

        # Use refined features to adjust heatmaps
        # Project back: each joint's refined feature biases its heatmap channel
        refined_bias = refined_feats.mean(-1).unsqueeze(-1).unsqueeze(-1)  # (B, J, 1, 1)
        refined_heatmaps = torch.sigmoid(
            self.refine_head(init_heatmaps) + refined_bias
        )                                                # (B, J, 64, 64)

        vis_pred = self.visibility_head(enc)             # (B, J)

        return refined_heatmaps, vis_pred
  
   # Joint weights — emphasize torso joints
JOINT_WEIGHTS = torch.ones(33)
# Left Arm
JOINT_WEIGHTS[12] = 3.0  # left shoulder
JOINT_WEIGHTS[14] = 3.0  # right shoulder
JOINT_WEIGHTS[16] = 3.0  # left hip

# Bone connections for structural loss
BONES = [
    # torso
    (12, 11), (24, 23), (12, 24), (11, 23),
    # left leg
    (24, 26), (26, 28),
    # right leg
    (23, 25), (25, 27),
    # left arm
    (12, 14), (14, 16),
    # right arm
    (11, 13), (13, 15),
    # head
    (8, 7), (10, 9),
]

def masked_per_joint_loss(
    pred, target, visibility,
    vis_pred,                        # (B, J) — model's predicted visibility
    coord_weight=0.25,
    bone_weight=0.5,
    vis_weight=0.125,                  # weight for visibility BCE loss
    use_pred_vis_for_coord=True,     # mirror pose_distance: weight by min(pred, gt)
):
    b, j, h, w = pred.shape
    jw = JOINT_WEIGHTS.to(pred.device)  # (J,)

    # ------------------------------------------------------------------
    # 1. Visibility supervision (BCE)
    # ------------------------------------------------------------------
    vis_loss = F.binary_cross_entropy(vis_pred, visibility, reduction='none')
    vis_loss = vis_loss.mean(dim=0)  # per joint
    num_visible = visibility.sum()
    vis_loss = vis_loss.sum() / (num_visible + 1e-6)

    # ------------------------------------------------------------------
    # 2. Effective per-joint visibility weight
    #    Mirrors pose_distance: vv = min(vis1[i], vis2[i])
    #    Here: "predicted confidence" vs "ground truth presence"
    # ------------------------------------------------------------------
    if use_pred_vis_for_coord:
        # Soft weight: joint matters only if both GT says visible
        # AND model is confident — mirrors the min() in pose_distance
        eff_vis = torch.min(vis_pred, visibility)          # (B, J)
    else:
        eff_vis = visibility                               # original behaviour

    # ------------------------------------------------------------------
    # 3. Heatmap MSE loss (unchanged, uses GT visibility for masking)
    # ------------------------------------------------------------------
    mse = ((pred - target) ** 2 * (1 + 5 * target)).view(b, j, -1).mean(-1)  # (B, J)
    mse = mse * jw.unsqueeze(0)
    vis_denom = visibility.sum(-1) + 1e-6                 # (B,) — keep GT denom here
    heatmap_loss = (mse * visibility).sum(-1) / vis_denom
    heatmap_loss = heatmap_loss.mean()

    # ------------------------------------------------------------------
    # 4. Coordinate extraction
    # ------------------------------------------------------------------
    def soft_argmax(heatmap):
        h_idx = torch.linspace(0, 1, h, device=heatmap.device)
        w_idx = torch.linspace(0, 1, w, device=heatmap.device)
        sm = F.softmax(heatmap.view(b, j, -1) / 0.1, dim=-1).view(b, j, h, w)
        x = (sm.sum(-2) * w_idx).sum(-1)
        y = (sm.sum(-1) * h_idx).sum(-1)
        return torch.stack([x, y], dim=-1)  # (B, J, 2)


    pred_coords   = soft_argmax(pred)      # (B, J, 2)
    target_coords = soft_argmax(target)    # (B, J, 2)

    # ------------------------------------------------------------------
    # 5. Weighted coordinate loss — uses eff_vis (mirrors pose_distance)
    #    Distance is L2; weight = min(pred_vis, gt_vis) * joint_weight
    # ------------------------------------------------------------------
    coord_dist = torch.norm(pred_coords - target_coords, dim=-1)   # (B, J)  ← L2, like pose_distance
    coord_dist = coord_dist * jw.unsqueeze(0)                      # joint importance

    eff_denom  = eff_vis.sum(-1) + 1e-6                            # (B,)
    coord_loss = (coord_dist * eff_vis).sum(-1) / eff_denom        # visibility-weighted mean
    coord_loss = coord_loss.mean()

    # ------------------------------------------------------------------
    # 6. Bone structural loss — also uses eff_vis for both endpoints
    #    Mirrors: bone_vis = vis_a * vis_b (but soft now)
    # ------------------------------------------------------------------
    bone_loss = torch.tensor(0.0, device=pred.device)
    num_valid_bones = 0

    for (a, b_idx) in BONES:
        # Soft bone visibility: both endpoints must be confidently visible
        bone_vis = eff_vis[:, a] * eff_vis[:, b_idx]   # (B,) — product mirrors min() chained

        if bone_vis.sum() < 1e-6:
            continue

        pred_vec   = pred_coords[:, a, :]   - pred_coords[:, b_idx, :]
        target_vec = target_coords[:, a, :] - target_coords[:, b_idx, :]

        pred_len   = torch.norm(pred_vec,   dim=-1)
        target_len = torch.norm(target_vec, dim=-1)

        length_diff = ((pred_len - target_len) ** 2) * bone_vis
        bone_loss  += length_diff.sum() / (bone_vis.sum() + 1e-6)
        num_valid_bones += 1

    if num_valid_bones > 0:
        bone_loss = bone_loss / num_valid_bones

    # ------------------------------------------------------------------
    # 7. Total loss
    # ------------------------------------------------------------------
    total = (
        heatmap_loss * .125
        + coord_weight * coord_loss
        + bone_weight  * bone_loss
        + vis_weight   * vis_loss
    )
    return total, {
        "heatmap": heatmap_loss.item(),
        "coord":   coord_loss.item(),
        "bone":    bone_loss.item(),
        "vis":     vis_loss.item(),
    }  
    
PATH = "C:/Users/leahz/Documents/ATC/pose-project/data"

def _extract_epoch_from_filename(filename):
    match = re.search(r"epoch_(\d+)", os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0


def train_pose(
    filename="",
    num_epochs=100,
    seed=42,
    learning_rate=5e-4,
    save_every=10,
    log_every=10,
    max_chunks=10,
    batch_size=8,
    train_ratio=0.8,
):
    # --- Seed ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, train_loader, _ = build_chunked_loaders(
        chunk_path=f"{PATH}/dataset_cache",
        max_chunks=max_chunks,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        shuffle_chunks=True,
    )


    model = PoseCNN(num_keypoints=33)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = f"{PATH}/models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    resume_path = ""
    if isinstance(filename, str) and filename.strip():
        resume_path = f"{checkpoint_dir}/{filename.strip()}"

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )
        print(f"Missing keys when loading checkpoint: {missing_keys}")
        print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as exc:
                print(
                    "Optimizer state is incompatible with current model parameters; "
                    "starting with a fresh optimizer state."
                )
                print(f"Optimizer load warning: {exc}")

        # Prefer the epoch stored in checkpoint; fallback to parsing from filename.
        start_epoch = int(checkpoint.get("epoch", 0))
        if start_epoch <= 0:
            start_epoch = _extract_epoch_from_filename(resume_path)

        print(f"Resuming training from checkpoint: {resume_path}")
        print(f"Starting from epoch {start_epoch + 1} of {num_epochs}")

    # Force requested LR even when optimizer state was loaded from checkpoint.
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5,
    )

    # Align scheduler epoch position when resuming.
    if start_epoch > 0:
        scheduler.last_epoch = start_epoch - 1
        scheduler.base_lrs = [learning_rate for _ in optimizer.param_groups]

    if start_epoch >= num_epochs:
        print(f"Checkpoint is already at/after target num_epochs={num_epochs}. Nothing to train.")
        return model

    for epoch in tqdm.tqdm(range(start_epoch, num_epochs)):
        model.train()
        total_loss = 0.0
        total_heatmap = 0.0
        total_coord = 0.0
        total_bone = 0.0
        total_vis = 0.0

        for batch_idx, (x, y, visibility) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            visibility = visibility.to(device, non_blocking=True)

            heatmaps, vis_pred = model(x)
            loss, breakdown = masked_per_joint_loss(
                heatmaps, y, visibility, vis_pred
            )

            optimizer.zero_grad()
            loss.backward()

            # --- Gradient norm (debugging) ---
            total_grad_norm = 0
            if batch_idx % log_every == 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')).item()

            optimizer.step()

            total_loss += loss.item()
            total_heatmap += breakdown["heatmap"]
            total_coord += breakdown["coord"]
            total_bone += breakdown["bone"]
            total_vis += breakdown["vis"]

            # --- Batch logging ---
            if batch_idx % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] ",
                    f"Batch [{batch_idx}/{len(train_loader)}] ",
                    f"Loss: {loss.item():.6f} ",
                    f"HM: {breakdown['heatmap']:.6f} ",
                    f"Coord: {breakdown['coord']:.6f} ",
                    f"Bone: {breakdown['bone']:.6f} ",
                    f"Vis: {breakdown['vis']:.6f} ",
                    f"GradNorm: {total_grad_norm:.4f} ",
                    f"LR: {current_lr}"
                )

        scheduler.step()

        num_batches = max(len(train_loader), 1)
        avg_loss = total_loss / num_batches
        avg_heatmap = total_heatmap / num_batches
        avg_coord = total_coord / num_batches
        avg_bone = total_bone / num_batches
        avg_vis = total_vis / num_batches
        current_epoch = epoch + 1

        print(f"\nEpoch {current_epoch} Summary:")
        print(f"Avg Loss: {avg_loss:.6f}")
        print(
            f"Avg Breakdown -> HM: {avg_heatmap:.6f}, "
            f"Coord: {avg_coord:.6f}, "
            f"Bone: {avg_bone:.6f}, "
            f"Vis: {avg_vis:.6f}\n"
        )
        print(f"Scheduler LR after epoch {current_epoch}: {scheduler.get_last_lr()[0]:.8f}")

        if current_epoch % save_every == 0:
            checkpoint_path = f"{checkpoint_dir}/pose_model_added_visibility_epoch_{current_epoch}.pth"
            torch.save({
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_keypoints": 33,
                "avg_loss": avg_loss,
                "avg_heatmap": avg_heatmap,
                "avg_coord": avg_coord,
                "avg_bone": avg_bone,
                "avg_vis": avg_vis,
                "seed": seed,
            }, checkpoint_path)

            print(f"Saved checkpoint: {checkpoint_path}")

    return model


if __name__ == "__main__":
    # Leave blank to train from scratch, or provide a checkpoint filename to resume.
    train_pose(filename="")