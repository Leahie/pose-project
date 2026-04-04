"""
model.py
────────
CNN backbone + Graph Attention Network pose estimator.

Architecture overview
─────────────────────
  sketch (3×64×64)
      │
  CNNBackbone          → global image feature  (B, img_feat_dim)
      │
  broadcast to J nodes + concat joint_embedding[i]
      │
  GATv2Conv × gat_layers   (nodes talk to anatomical neighbours)
      │
  MLP head per node    → (x, y, z, visibility)  raw logits
      │
  sigmoid(vis)         → visibility in (0, 1)
  coords               → passed raw to loss (no activation needed for L2)

Output shapes:
  coords   : (B, J, 3)
  vis_pred : (B, J)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from constants import NUM_JOINTS, EDGE_INDEX


class CNNBackbone(nn.Module):
    """
    Encodes a 3×64×64 sketch image into a global feature vector.

    Architecture:
        3 × conv-block (Conv→ReLU→BN), two MaxPool strides,
        AdaptiveAvgPool → Flatten → Linear projection.

    Args:
        feat_dim: Dimension of the output feature vector.
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 64 → 64
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                              # 64 → 32
            # Block 2: 32 → 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                              # 32 → 16
            # Block 3: 16 → 16  (no pool — keep spatial resolution)
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4)),                 # (B, 256, 4, 4)
            nn.Flatten(),                                 # (B, 4096)
        )
        self.proj = nn.Sequential(
            nn.Linear(256 * 16, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 64, 64) normalised sketch tensor.
        Returns:
            (B, feat_dim) global image feature.
        """
        return self.proj(self.encoder(x))


class SketchToCoordGAT(nn.Module):
    """
    Full sketch-to-3D-pose model using Graph Attention Networks.

    Steps:
        1. CNN backbone extracts a global image feature per sample.
        2. Each of the 33 joints gets a node feature =
               concat(global_image_feat, joint_embedding[i])
        3. GATv2 layers propagate information along skeleton edges.
        4. Per-node MLP head outputs (x, y, z, visibility).

    Args:
        num_joints:    Number of body joints (33 for MediaPipe BlazePose).
        img_feat_dim:  CNN output feature dimension.
        joint_emb_dim: Learnable per-joint embedding dimension.
        gat_hidden:    Total hidden dimension across all GAT heads.
        gat_heads:     Number of attention heads per GAT layer.
        gat_layers:    Number of stacked GATv2 layers.
        dropout:       Attention dropout probability.
    """

    def __init__(
        self,
        num_joints:    int   = NUM_JOINTS,
        img_feat_dim:  int   = 256,
        joint_emb_dim: int   = 64,
        gat_hidden:    int   = 128,
        gat_heads:     int   = 4,
        gat_layers:    int   = 3,
        dropout:       float = 0.1,
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "torch-geometric is required.\n"
                "Install: pip install torch-geometric"
            )

        self.num_joints = num_joints

        # ── 1. CNN backbone ────────────────────────────────────────────────────
        self.cnn = CNNBackbone(feat_dim=img_feat_dim)

        # ── 2. Learnable joint identity embedding ──────────────────────────────
        self.joint_emb = nn.Embedding(num_joints, joint_emb_dim)

        node_in = img_feat_dim + joint_emb_dim

        # ── 3. GATv2 layers ───────────────────────────────────────────────────
        #   All intermediate layers: concat=True  → output dim = gat_hidden
        #   Final layer:             concat=False → output dim = out_per_head
        self.gat_list = nn.ModuleList()
        in_dim = node_in
        for i in range(gat_layers):
            out_per_head = gat_hidden // gat_heads
            is_last      = (i == gat_layers - 1)
            self.gat_list.append(
                GATv2Conv(
                    in_channels    = in_dim,
                    out_channels   = out_per_head,
                    heads          = gat_heads,
                    concat         = not is_last,   # last layer averages heads
                    dropout        = dropout,
                    add_self_loops = True,
                )
            )
            in_dim = gat_hidden if not is_last else out_per_head

        self._gat_out_dim = in_dim

        # ── 4. Per-joint MLP head ──────────────────────────────────────────────
        #   Output: (x, y, z) raw + visibility logit
        self.head = nn.Sequential(
            nn.Linear(self._gat_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),   # x, y, z, vis_logit
        )

        # Cached on-device edge index (lazy)
        self._edge_index_cache: dict = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_edge_index(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._edge_index_cache:
            self._edge_index_cache[key] = EDGE_INDEX.to(device)
        return self._edge_index_cache[key]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 64, 64) normalised sketch images.

        Returns:
            coords   : (B, J, 3) — predicted (x, y, z), raw (no sigmoid).
            vis_pred : (B, J)    — predicted visibility in (0, 1).
        """
        B      = x.shape[0]
        device = x.device
        ei     = self._get_edge_index(device)   # (2, E)

        # ── Image features → (B, img_feat_dim) ────────────────────────────────
        img_feat = self.cnn(x)

        # ── Node features: broadcast + concat joint embedding ─────────────────
        joint_ids  = torch.arange(self.num_joints, device=device)   # (J,)
        joint_embs = self.joint_emb(joint_ids)                       # (J, emb_dim)

        # Expand to (B*J, dim)
        img_exp  = img_feat.unsqueeze(1).expand(-1, self.num_joints, -1) \
                           .reshape(B * self.num_joints, -1)
        emb_exp  = joint_embs.unsqueeze(0).expand(B, -1, -1) \
                             .reshape(B * self.num_joints, -1)

        node_feats = torch.cat([img_exp, emb_exp], dim=-1)  # (B*J, node_in)

        # ── Batch the graph: offset edge indices by i * num_joints ────────────
        offsets    = torch.arange(B, device=device) * self.num_joints  # (B,)
        src = (ei[0].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst = (ei[1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batched_ei = torch.stack([src, dst], dim=0)                    # (2, B*E)

        # ── GAT message passing ───────────────────────────────────────────────
        h = node_feats
        for layer in self.gat_list:
            h = F.elu(layer(h, batched_ei))   # (B*J, gat_dim)

        h = h.reshape(B, self.num_joints, -1)   # (B, J, gat_out_dim)

        # ── Per-joint prediction ──────────────────────────────────────────────
        out      = self.head(h)                  # (B, J, 4)
        coords   = out[..., :3]                  # (B, J, 3)  — raw, no activation
        vis_pred = torch.sigmoid(out[..., 3])    # (B, J)     — in (0, 1)

        return coords, vis_pred