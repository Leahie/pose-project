import os
import random

import torch
from torch.utils.data import DataLoader, random_split


class ChunkedDataset(torch.utils.data.Dataset):
	def __init__(self, chunk_dir, shuffle_chunks=False, seed=42, max_chunks=None):
		self.chunk_files = sorted(
			[
				os.path.join(chunk_dir, f)
				for f in os.listdir(chunk_dir)
				if f.endswith(".pt")
			]
		)

		if max_chunks is not None:
			if max_chunks <= 0:
				raise ValueError("max_chunks must be a positive integer or None")
			self.chunk_files = self.chunk_files[:max_chunks]

		if not self.chunk_files:
			raise ValueError(f"No chunk files found in: {chunk_dir}")

		if shuffle_chunks:
			rng = random.Random(seed)
			rng.shuffle(self.chunk_files)

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
				map_location="cpu",
			)
			self.current_chunk_idx = chunk_idx

	def __getitem__(self, idx):
		for i, size in enumerate(self.cumulative_sizes):
			if idx < size:
				self._load_chunk(i)
				prev = 0 if i == 0 else self.cumulative_sizes[i - 1]
				local_idx = idx - prev
				return (
					self.current_chunk["images"][local_idx],
					self.current_chunk["labels"][local_idx],
					self.current_chunk["visibility"][local_idx],
				)
		raise IndexError("Index out of range")


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
		generator=split_gen,
	)

	pin_memory = torch.cuda.is_available()

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=pin_memory,
		persistent_workers=False,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=pin_memory,
		persistent_workers=False,
	)

	return dataset, train_loader, test_loader
