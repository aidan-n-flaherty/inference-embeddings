import os
import pickle
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

class FactContrastiveDataset(Dataset):
	def __init__(
		self,
		factkg,
		N=10,
		K=[5, 5],
		model_name="sentence-transformers/all-MiniLM-L6-v2",
		cache_path="fact_sbert_cache.pkl",
		seed=42,
		types=["multi hop", "multi claim", "negation", "existence"]
	):
		self.factkg = factkg
		self.types = types
		self.N = N
		self.K = K
		self.current_length = 0
		self.model = SentenceTransformer(model_name)#, model_kwargs={"torch_dtype": "float16"})
		self.cache_path = cache_path
		self.cache = self._load_cache()
		self.random = random.Random()
		self.grouped = self._group_by_topic()
		self.samples = self._collect_topics()
		self.access = 0

		"""self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)

		base_model = AutoModel.from_pretrained(model_name).to(self.device)

		try:
			self.model = PeftModel.from_pretrained(base_model, "lora_adapter/checkpoint-1")
		except:
			# Choose LoRA injection points (Gemma-style projection layers)
			config = LoraConfig(
				r=16,
				lora_alpha=32,
				lora_dropout=0.05,
				target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
				bias="none",
				task_type="FEATURE_EXTRACTION"
			)

			self.model = get_peft_model(base_model, config)"""

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

			true_facts = [f for f, d in facts if d["Label"] == [True] and any(t in d["types"] for t in self.types)]
			false_facts = [f for f, d in facts if d["Label"] == [False] and any(t in d["types"] for t in self.types)]

			if len(true_facts) >= min(self.K) and false_facts:
				topics.append((topic, true_facts, false_facts))
		
		return topics

	"""def _get_embedding(self, fact):
		inputs = self.tokenizer(fact, return_tensors="pt", truncation=True).to(self.device)
		outputs = self.model(**inputs)
		emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
		return emb.cpu()"""
	
	def _get_embedding(self, fact):
		if fact in self.cache:
			return self.cache[fact]
		emb = torch.tensor(self.model.encode(fact, convert_to_numpy=True))
		self.cache[fact] = emb
		return emb

	def _random_external(self, topic, n):
		selections = self.random.sample([t for t in self.samples if t[0] != topic], n)

		facts = [self.random.choice(true_facts) for (t, true_facts, false_facts) in selections]

		return facts

	def _make_triplet(self):
		topic, true_facts, false_facts = self.random.choice(self.samples)
		selection = min(max(self.K), len(true_facts))
		true_sample = self.random.sample(true_facts, selection) + self._random_external(topic, self.N - selection)
		false_fact = self.random.choice(false_facts)

		indices = list(range(self.N))

		self.random.shuffle(indices)

		i = self.random.randrange(selection)
		alt = self.random.choice([f for f in true_facts if f != true_sample[i]])
		
		positive = true_sample[:i] + [alt] + true_sample[i + 1 :]
		negative = true_sample[:i] + [false_fact] + true_sample[i + 1 :]

		positive = [positive[n] for n in indices]
		negative = [negative[n] for n in indices]

		true_sample = [true_sample[n] for n in indices]

		print(false_fact, "conflicts with", true_facts)

		return true_sample, positive, negative, indices.index(i), [indices.index(n) for n in range(selection)]

	def __len__(self):
		return len(self.samples) * 10

	def __getitem__(self, idx):
		original, positive, negative, flipped_index, positive_indices = self._make_triplet()

		orig_embs = [self._get_embedding(f) for f in original]
		pos_embs = [self._get_embedding(f) for f in positive]
		neg_embs = [self._get_embedding(f) for f in negative]

		self.access += 1
		if self.access % 500 == 0 and len(self.cache) != self.current_length:
			print(f"Saving {len(self.cache) - self.current_length} new embeddings...")
			self._save_cache()
			print(f"New cache size: {self.current_length}")

		return {
			"original": original,
			"positive": positive,
			"negative": negative,
			"original_embeddings": torch.stack(orig_embs),
			"positive_embeddings": torch.stack(pos_embs),
			"negative_embeddings" : torch.stack(neg_embs),
			"flipped_index": flipped_index,
			"positive_indices": positive_indices
		}
	
class SyntheticContrastiveDataset(Dataset):
	def __init__(
		self,
		synthetic_data,
		model_name="sentence-transformers/all-MiniLM-L6-v2",
		cache_path="fact_sbert_cache.pkl"
	):
		self.synthetic_data = synthetic_data
		self.current_length = 0
		self.model = SentenceTransformer(model_name)#, model_kwargs={"torch_dtype": "float16"})
		self.cache_path = cache_path
		self.cache = self._load_cache()
		self.random = random.Random()
		self.access = 0

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

	"""def _get_embedding(self, fact):
		inputs = self.tokenizer(fact, return_tensors="pt", truncation=True).to(self.device)
		outputs = self.model(**inputs)
		emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
		return emb.cpu()"""
	
	def _get_embedding(self, fact):
		if fact in self.cache:
			return self.cache[fact]
		emb = torch.tensor(self.model.encode(fact, convert_to_numpy=True))
		self.cache[fact] = emb
		return emb

	def _random_external(self, index, n):
		selections = self.random.sample(self.synthetic_data[:index] + self.synthetic_data[index + 1:], n)
		
		facts = [self.random.choice(item["supported"] + item["refuted"]) for item in selections]

		return facts

	def _make_triplet(self):
		selection = 5
		index = self.random.randint(0, len(self.synthetic_data) - 1)
		item = self.synthetic_data[index]
		true_sample = item["supported"][:selection] + self._random_external(index, 5)
		false_fact = self.random.choice(item["refuted"])
		alt = item["statement"]


		i = self.random.randint(0, 4)
		
		positive = true_sample[:i] + [alt] + true_sample[i + 1 :]
		negative = true_sample[:i] + [false_fact] + true_sample[i + 1 :]

		indices = list(range(len(true_sample)))

		self.random.shuffle(indices)

		positive = [positive[n] for n in indices]
		negative = [negative[n] for n in indices]

		true_sample = [true_sample[n] for n in indices]

		#print(false_fact, "conflicts with", true_sample)

		return true_sample, positive, negative, indices.index(i), [indices.index(n) for n in range(selection)]

	def __len__(self):
		return len(self.synthetic_data)

	def __getitem__(self, idx):
		original, positive, negative, flipped_index, positive_indices = self._make_triplet()

		orig_embs = [self._get_embedding(f) for f in original]
		pos_embs = [self._get_embedding(f) for f in positive]
		neg_embs = [self._get_embedding(f) for f in negative]

		self.access += 1
		if self.access % 500 == 0 and len(self.cache) != self.current_length:
			print(f"Saving {len(self.cache) - self.current_length} new embeddings...")
			self._save_cache()
			print(f"New cache size: {self.current_length}")

		return {
			"original": original,
			"positive": positive,
			"negative": negative,
			"original_embeddings": torch.stack(orig_embs),
			"positive_embeddings": torch.stack(pos_embs),
			"negative_embeddings" : torch.stack(neg_embs),
			"flipped_index": flipped_index,
			"positive_indices": positive_indices
		}