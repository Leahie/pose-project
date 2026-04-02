import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from dataset import build_chunked_loaders


class PoseCNN(nn.Module):
    def __init__(self, num_keypoints=33):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 64x64
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding=1),  # 64x64
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),                  # 32x32

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(2),                  # 16x16

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 32x32
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),   # 64x64
            nn.ReLU(),

            nn.Conv2d(64, num_keypoints, 1),             # 64x64xK
            nn.Sigmoid()                                 # bound output to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    # import your model class
    
# Joint weights — emphasize torso joints
JOINT_WEIGHTS = torch.ones(33)
JOINT_WEIGHTS[11] = 3.0  # left shoulder
JOINT_WEIGHTS[12] = 3.0  # right shoulder
JOINT_WEIGHTS[23] = 3.0  # left hip
JOINT_WEIGHTS[24] = 3.0  # right hip

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

def masked_per_joint_loss(pred, target, visibility, coord_weight=0.2, bone_weight=0.3):
    # pred, target: (B, J, H, W)
    # visibility: (B, J)
    b, j, h, w = pred.shape

    # --- Joint weights (emphasize torso) ---
    jw = JOINT_WEIGHTS.to(pred.device)  # (J,)

    # --- Weighted heatmap MSE loss ---
    mse = ((pred - target) ** 2 * (1 + 5 * target)).view(b, j, -1).mean(-1)  # (B, J)
    mse = mse * jw.unsqueeze(0)  # apply joint weights
    vis_denom = visibility.sum(-1) + 1e-6  # (B,)
    loss = (mse * visibility).sum(-1) / vis_denom
    loss = loss.mean()

    # --- Coordinate extraction ---
    def soft_argmax(heatmap):
        h_idx = torch.linspace(0, 1, h, device=heatmap.device)
        w_idx = torch.linspace(0, 1, w, device=heatmap.device)
        sm = F.softmax(heatmap.view(b, j, -1) / 0.1, dim=-1).view(b, j, h, w)
        x = (sm.sum(-2) * w_idx).sum(-1)
        y = (sm.sum(-1) * h_idx).sum(-1)
        return torch.stack([x, y], dim=-1)  # (B, J, 2)

    def hard_coords(heatmap):
        flat = heatmap.view(b, j, -1).argmax(-1)
        y = (flat // w).float() / (h - 1)
        x = (flat % w).float() / (w - 1)
        return torch.stack([x, y], dim=-1)  # (B, J, 2)

    pred_coords = soft_argmax(pred)    # (B, J, 2)
    target_coords = hard_coords(target)

    # --- Weighted coordinate loss ---
    coord_mse = ((pred_coords - target_coords) ** 2).sum(-1)  # (B, J)
    coord_mse = coord_mse * jw.unsqueeze(0)  # apply joint weights here too
    coord_loss = (coord_mse * visibility).sum(-1) / vis_denom
    coord_loss = coord_loss.mean()

    # --- Bone structural loss ---
    # For each bone (a, b): penalize if predicted distance differs from target distance
    bone_loss = torch.tensor(0.0, device=pred.device)
    num_valid_bones = 0

    for (a, b_idx) in BONES:
        # Both joints must be visible
        vis_a = visibility[:, a]   # (B,)
        vis_b = visibility[:, b_idx]  # (B,)
        bone_vis = vis_a * vis_b   # (B,) — 1 only if both visible

        if bone_vis.sum() < 1e-6:
            continue

        # Predicted and target bone vectors
        pred_vec = pred_coords[:, a, :] - pred_coords[:, b_idx, :]    # (B, 2)
        target_vec = target_coords[:, a, :] - target_coords[:, b_idx, :]  # (B, 2)

        pred_len = torch.norm(pred_vec, dim=-1)    # (B,)
        target_len = torch.norm(target_vec, dim=-1)  # (B,)

        # Penalize difference in bone length
        length_diff = ((pred_len - target_len) ** 2) * bone_vis  # (B,)
        bone_loss = bone_loss + length_diff.sum() / (bone_vis.sum() + 1e-6)
        num_valid_bones += 1

    if num_valid_bones > 0:
        bone_loss = bone_loss / num_valid_bones

    return loss + coord_weight * coord_loss + bone_weight * bone_loss


PATH = "C:/Users/leahz/Documents/ATC/pose-project/data"


# --- Seed ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

_, train_loader, _ = build_chunked_loaders(
    chunk_path=f"{PATH}/dataset_cache",
    max_chunks=10,
    batch_size=8,
    train_ratio=0.8,
    seed=seed,
    shuffle_chunks=True,
)



model = PoseCNN(num_keypoints=33)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

num_epochs = 100
save_every = 10
log_every = 10   # print every N batches

checkpoint_dir = f"{PATH}/models"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in tqdm.tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0

    for batch_idx, (x, y, visibility) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        visibility = visibility.to(device, non_blocking=True)

        pred = model(x)
        loss = masked_per_joint_loss(pred, y, visibility)

        optimizer.zero_grad()
        loss.backward()

        # --- Gradient norm (debugging) ---
        total_grad_norm = 0
        if batch_idx % log_every == 0:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')).item()

        optimizer.step()

        total_loss += loss.item()

        # --- Batch logging ---
        if batch_idx % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{num_epochs}] ",
                f"Batch [{batch_idx}/{len(train_loader)}] ",
                f"Loss: {loss.item():.6f} ",
                f"GradNorm: {total_grad_norm:.4f} ",
                f"LR: {current_lr}"
            )

    avg_loss = total_loss / len(train_loader)
    current_epoch = epoch + 1


    print(f"\nEpoch {current_epoch} Summary:")
    print(f"Avg Loss: {avg_loss:.6f}\n")

    if current_epoch % save_every == 0:
        checkpoint_path = f"{checkpoint_dir}/pose_model_epoch_{current_epoch}.pth"
        torch.save({
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_keypoints": 33,
            "avg_loss": avg_loss,
            "seed": seed,
        }, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")