import os
import random

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np

class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, chunk_dir, shuffle_chunks=False, seed=42, max_chunks=None):
        self.chunk_files = sorted([
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".pt")
        ])

        if max_chunks is not None:
            if max_chunks <= 0:
                raise ValueError("max_chunks must be a positive integer or None")
            self.chunk_files = self.chunk_files[:max_chunks]

        if shuffle_chunks:
            rng = random.Random(seed)
            rng.shuffle(self.chunk_files)  # shuffle chunk order, not samples

        self.chunk_sizes = []
        self.cumulative_sizes = []

        total = 0
        for file in self.chunk_files:
            data = torch.load(file, map_location="cpu")
            size = len(data["images"])
            self.chunk_sizes.append(size)
            total += size
            self.cumulative_sizes.append(total)

        self.current_chunk_idx = -1
        self.current_chunk = None

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _load_chunk(self, chunk_idx):
        if self.current_chunk_idx != chunk_idx:
            self.current_chunk = torch.load(
                self.chunk_files[chunk_idx],
                map_location="cpu"
            )
            self.current_chunk_idx = chunk_idx

    def get_all_visibility(self):
        """Fast path for sampler weights: load only visibility tensors from each chunk."""
        vis_list = []
        for chunk_file in self.chunk_files:
            chunk = torch.load(chunk_file, map_location="cpu")
            vis = chunk["visibility"]
            if isinstance(vis, torch.Tensor):
                vis = vis.numpy()
            vis_list.append(np.asarray(vis, dtype=np.float32))
        return np.concatenate(vis_list, axis=0)

    def __getitem__(self, idx):
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                self._load_chunk(i)
                prev = 0 if i == 0 else self.cumulative_sizes[i - 1]
                local_idx = idx - prev
                return (
                    self.current_chunk["images"][local_idx],
                    self.current_chunk["labels"][local_idx],
                    self.current_chunk["visibility"][local_idx]
                )

def build_chunked_loaders(
	chunk_path,
	max_chunks=25,
	batch_size=8,
	train_ratio=0.8,
	seed=42,
	shuffle_chunks=True,
):
	dataset = ChunkedDataset(
		chunk_path,
		shuffle_chunks=shuffle_chunks,
		seed=seed,
		max_chunks=max_chunks,
	)

	train_size = int(train_ratio * len(dataset))
	test_size = len(dataset) - train_size
	split_gen = torch.Generator().manual_seed(seed)

	train_dataset, test_dataset = random_split(
		dataset,
		[train_size, test_size],
		generator=split_gen
	)

	# Fast visibility loading: use cache when available, otherwise build once from chunk tensors.
	vis_cache = os.path.join(chunk_path, "visibility_all.npy")
	if os.path.exists(vis_cache):
		all_visibility = np.load(vis_cache).astype(np.float32)
	else:
		print("Building visibility cache (one-time cost)...")
		all_visibility = dataset.get_all_visibility().astype(np.float32)
		np.save(vis_cache, all_visibility)
		print(f"Saved visibility cache to: {vis_cache}")

	train_indices = np.asarray(train_dataset.indices, dtype=np.int64)
	train_visibility = all_visibility[train_indices]  # (num_train_samples, num_joints)

	joint_counts = train_visibility.sum(axis=0)
	joint_weights = 1.0 / (joint_counts + 1e-6)
	sample_weights = (train_visibility * joint_weights).sum(axis=1)
	sample_weights = np.clip(sample_weights, a_min=1e-8, a_max=None)
	sample_weights_t = torch.tensor(sample_weights, dtype=torch.double)

	train_sampler = WeightedRandomSampler(
		weights=sample_weights_t,
		num_samples=len(sample_weights_t),
		replacement=True,
	)

	pin_memory = torch.cuda.is_available()

	train_loader = DataLoader(
		train_dataset,
		batch_size=8,
		sampler=train_sampler,
		num_workers=0,
		pin_memory=pin_memory,
		persistent_workers=False
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=8,
		shuffle=False,
		num_workers=0,
		pin_memory=pin_memory,
		persistent_workers=False
	)

	print(f"Chunks used: {len(dataset.chunk_files)}")
	print(f"Dataset size: {len(dataset)}")
	print(f"Train size: {len(train_dataset)}")
	print(f"Test size: {len(test_dataset)}")
	print(f"Weighted sampler active: {len(sample_weights_t)} training samples")
	print(f"Visibility cache: {vis_cache}")

	return dataset, train_loader, test_loader
