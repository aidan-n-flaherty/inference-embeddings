import os
import pickle
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class FactContrastiveDataset(Dataset):
	"""
	On-demand contrastive dataset for FactKG.
	Dynamically generates (positive, negative) pairs.
	"""

	def __init__(
		self,
		factkg,
		N=10,
		model_name="sentence-transformers/all-MiniLM-L6-v2",
		cache_path="fact_sbert_cache.pkl",
		seed=42,
	):
		self.factkg = factkg
		self.N = N
		self.current_length = 0
		self.model = SentenceTransformer(model_name)#, model_kwargs={"torch_dtype": "float16"})
		self.cache_path = cache_path
		self.cache = self._load_cache()
		self.random = random.Random()
		self.grouped = self._group_by_topic()
		self.samples = self._collect_topics()
		self.access = 0

	# -------------------------------
	# Internal setup helpers
	# -------------------------------

	def _load_cache(self):
		if os.path.exists(self.cache_path):
			with open(self.cache_path, "rb") as f:
				cache = pickle.load(f)
			print(f"[Cache] Loaded {len(cache)} embeddings from {self.cache_path}.")

			self.current_length = len(cache)
			return cache
		print("[Cache] Starting with empty embedding cache.")
		return {}

	def _save_cache(self):
		with open(self.cache_path, "wb") as f:
			pickle.dump(self.cache, f)
			self.current_length = len(self.cache)

	def _group_by_topic(self):
		grouped = defaultdict(list)
		for fact, details in self.factkg.items():
			keys = tuple(sorted(details["Entity_set"]))
			for key in keys:
				grouped[key].append((fact, details))
		return grouped

	def _collect_topics(self):
		topics = []
		for topic, facts in self.grouped.items():
			try:
				if float(topic.replace('"', '').replace('-', '')):
					continue
			except:
				pass

			true_facts = [f for f, d in facts if d["Label"] == [True] and "multi hop" in d["types"]]
			false_facts = [f for f, d in facts if d["Label"] == [False] and "multi hop" in d["types"]]

			if len(true_facts) >= self.N and false_facts:
				topics.append((topic, true_facts, false_facts))
		return topics

	def _get_embedding(self, fact):
		if fact in self.cache:
			return self.cache[fact]
		emb = torch.tensor(self.model.encode(fact, convert_to_numpy=True))
		self.cache[fact] = emb
		return emb

	def _make_triplet(self):
		topic, true_facts, false_facts = self.random.choice(self.samples)
		true_sample = self.random.sample(true_facts, self.N)
		false_fact = self.random.choice(false_facts)

		# flip coin: positive or negative pair
		i = self.random.randrange(self.N)
		alt = self.random.choice([f for f in true_facts if f != true_sample[i]])
		positive = true_sample[:i] + [alt] + true_sample[i + 1 :]
		
		i = self.random.randrange(self.N)
		negative = true_sample[:i] + [false_fact] + true_sample[i + 1 :]

		return true_sample, positive, negative

	def __len__(self):
		return len(self.samples) * 10  # arbitrary multiplier for effective epoch size

	def __getitem__(self, idx):
		original, positive, negative = self._make_triplet()

		orig_embs = [self._get_embedding(f) for f in original]
		pos_embs = [self._get_embedding(f) for f in positive]
		neg_embs = [self._get_embedding(f) for f in negative]

		self.access += 1
		if self.access % 500 == 0 and len(self.cache) != self.current_length:  # occasionally save cache
			print(f"Saving {len(self.cache) - self.current_length} new embeddings...")
			self._save_cache()
			print(f"New cache size: {self.current_length}")

		return {
			"original": original,
			"positive": positive,
			"negative": negative,
			"original_embeddings": torch.stack(orig_embs),
			"positive_embeddings": torch.stack(pos_embs),
			"negative_embeddings" : torch.stack(neg_embs)
		}