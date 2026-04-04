"""
constants.py
────────────
Shared constants: graph topology, joint metadata, and per-joint loss weights.
"""

import torch

NUM_JOINTS = 33
COORD_DIM  = 3   # x, y, z

# ── Skeleton edges ─────────────────────────────────────────────────────────────
EDGES_RAW = [
    # head
    (8,6),(6,5),(5,4),(4,0),(0,1),(1,2),(2,3),(3,7),(10,9),
    # neck
    (10,12),(9,11),(8,12),(7,11),
    # torso
    (12,11),(12,24),(24,23),(23,11),(12,23),(11,24),
    # left arm
    (11,13),(13,15),(11,15),(15,21),(15,19),(19,17),(15,17),
    # right arm
    (12,14),(14,16),(12,16),(16,22),(16,20),(20,18),(16,18),
    # left leg
    (24,26),(26,28),(24,28),(23,26),
    # right leg
    (23,25),(25,27),(23,27),(24,25),
    # left foot
    (28,30),(30,32),(32,28),
    # right foot
    (27,29),(29,31),(31,27),
]

# Bones used for structural length loss
BONES = [
    (12,11),(24,23),(12,24),(11,23),   # torso
    (24,26),(26,28),(23,25),(25,27),   # legs
    (12,14),(14,16),(11,13),(13,15),   # arms
    (28,30),(30,32),(27,29),(29,31),   # feet
]

# Joint category groups
JOINT_CATEGORIES = {
    "head":       [0,1,2,3,4,5,6,7,8,9,10],
    "torso":      [11,12,23,24],
    "left_arm":   [11,13,15,17,19,21],
    "right_arm":  [12,14,16,18,20,22],
    "left_leg":   [23,25,27],
    "right_leg":  [24,26,28],
    "left_hand":  [15,17,18,19,20,22],
    "right_hand": [16,18,19,20,21],
    "left_foot":  [28,30,32],
    "right_foot": [27,29,31],
}

# Reverse lookup: joint index → category name
JOINT_TO_CAT: dict = {}
for _cat, _idxs in JOINT_CATEGORIES.items():
    for _i in _idxs:
        JOINT_TO_CAT[_i] = _cat

# Per-joint loss weight tensor  (higher = penalised more)
JOINT_WEIGHTS = torch.ones(NUM_JOINTS)
for _j in [11, 12, 23, 24]:   # torso anchors
    JOINT_WEIGHTS[_j] = 3.0
for _j in [13, 14, 25, 26]:   # elbows / knees
    JOINT_WEIGHTS[_j] = 2.0


def build_edge_index() -> torch.Tensor:
    """
    Build an undirected COO edge-index tensor from EDGES_RAW.
    Returns shape (2, num_directed_edges).
    """
    src, dst = [], []
    seen = set()
    for a, b in EDGES_RAW:
        for u, v in [(a, b), (b, a)]:
            if (u, v) not in seen:
                src.append(u)
                dst.append(v)
                seen.add((u, v))
    return torch.tensor([src, dst], dtype=torch.long)


# Pre-built at import time; move to device inside model forward pass.
EDGE_INDEX: torch.Tensor = build_edge_index()