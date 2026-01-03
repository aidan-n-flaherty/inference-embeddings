import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

from model import fully_connected_edges

class LinearProbe(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Sequential(
			nn.Linear(3, 1),
		)
		self.flatten = nn.Flatten()

	def forward(self, x):
		#return self.fc(self.flatten(x))
		return self.fc(x.sum(dim=1))

def run_linear_probe(dataset, test_dataset, device, epochs=10, lr=1e-3):
	from sentence_transformers import CrossEncoder

	model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

	probe = LinearProbe().to(device)
	optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
	criterion = nn.BCEWithLogitsLoss()

	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

	train_acc, val_acc, train_loss, val_loss = 0.0, 0.0, 0.0, 0.0

	cache = {}

	import os
	import pickle

	if os.path.exists(f"nli_cache.pkl"):	
		with open("nli_cache.pkl", "rb") as f:
			cache = pickle.load(f)

	for epoch in range(epochs):
		total_loss = 0.0
		n = 0
		for batch in loader:
			with torch.no_grad():
				positive = batch["original"]
				negative = batch["negative"]

				pairs_positive = []
				pairs_negative = []
				for i in range(len(positive)):
					for j in range(len(positive)):
						if i == j:
							continue

						pairs_positive.append((positive[i][0], positive[j][0]))
						pairs_negative.append((negative[i][0], negative[j][0]))

				pos_out = []
				for pair in pairs_positive:
					if pair in cache:
						pos_out.append(cache[pair])
					else:
						cache[pair] = model.predict(pair)
						pos_out.append(cache[pair])

				pos_out = torch.tensor(pos_out).to(device)

				neg_out = []
				for pair in pairs_negative:
					if pair in cache:
						neg_out.append(cache[pair])
					else:
						cache[pair] = model.predict(pair)
						neg_out.append(cache[pair])
				neg_out = torch.tensor(neg_out).to(device)

			x = torch.stack([pos_out, neg_out])
			y = torch.tensor([0., 1.], device=device).unsqueeze(1)

			logits = probe(x)
			loss = criterion(logits, y)
			
			n += 1

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			print(len(cache))
			
			if n % 100 == 0:
				print(f"[Linear Probe Train] Epoch {epoch+1}/{epochs} {n}/{len(loader)} | Loss: {total_loss/n:.4f}")

				with open(f"nli_cache.pkl", "wb") as f:
					pickle.dump(cache, f)
		
		for mode in ["train", "test"]:
			total_acc = 0
			total_loss = 0
			differentiation = 0
			n = 0

			tp, fp, tn, fn = 0, 0, 0, 0

			for batch in (test_loader if mode == "test" else loader):
				with torch.no_grad():
					#anchor = batch["original_embeddings"].squeeze(0).to(device).float()
					positive = batch["original"]
					negative = batch["negative"]

					pairs_positive = []
					pairs_negative = []
					for i in range(len(positive)):
						for j in range(len(positive)):
							if i == j:
								continue

							pairs_positive.append((positive[i][0], positive[j][0]))
							pairs_negative.append((negative[i][0], negative[j][0]))

					pos_out = []
					for pair in pairs_positive:
						if pair in cache:
							pos_out.append(cache[pair])
						else:
							cache[pair] = model.predict(pair)
							pos_out.append(cache[pair])

					pos_out = torch.tensor(pos_out).to(device)

					neg_out = []
					for pair in pairs_negative:
						if pair in cache:
							neg_out.append(cache[pair])
						else:
							cache[pair] = model.predict(pair)
							neg_out.append(cache[pair])
					neg_out = torch.tensor(neg_out).to(device)

				x = torch.stack([pos_out, neg_out])
				y = torch.tensor([0., 1.], device=device).unsqueeze(1)

				logits = probe(x)
				loss = criterion(logits, y)

				n += 1

				preds = (torch.sigmoid(logits) > 0.5).float().squeeze(1)

				if torch.sum(preds) == 1:
					differentiation += 1

				acc = (preds == y.squeeze(1)).float().mean().item()

				total_loss += loss.item()
				total_acc += acc

				tp += ((preds == 1) & (y.squeeze(1) == 1)).sum().item()
				fp += ((preds == 1) & (y.squeeze(1) == 0)).sum().item()
				tn += ((preds == 0) & (y.squeeze(1) == 0)).sum().item()
				fn += ((preds == 0) & (y.squeeze(1) == 1)).sum().item()

				#print(f"[Linear Probe Test] Epoch {epoch+1}/{epochs} {n}/{len(test_loader)} | Loss: {total_loss/n:.4f} | Acc: {total_acc/n:.4f} | Differentiation: {differentiation/n:.4f}")

			if mode == "train":
				train_acc = total_acc/n
				train_loss = total_loss/n
			else:
				val_acc = total_acc/n
				val_loss = total_loss/n

			f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
			print(f"[Linear Probe {mode.capitalize()}] Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} | Acc: {total_acc/n:.4f} | Differentiation: {differentiation/n:.4f} | F1: {f1:.4f}")

			with open(f"nli_cache.pkl", "wb") as f:
				pickle.dump(cache, f)

	print("Linear probe evaluation complete.")
	model.train()
	return train_acc, train_loss, val_acc, val_loss

"""import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

from model import fully_connected_edges

class LinearProbe(nn.Module):
	def __init__(self, embedding_dim, hidden_dim=128):
		super().__init__()
		self.fc = nn.Sequential(
			nn.Linear(embedding_dim, 1),
			nn.Sigmoid()
		)
		self.flatten = nn.Flatten()

	def forward(self, x):
		#return self.fc(self.flatten(x))
		#return torch.sum(self.fc(x), dim=1)
		#return self.fc(torch.sum(x, dim=1))
		return self.fc(x)

def run_linear_probe(model, dataset, test_dataset, device, epochs=10, embedding_dim=128, lr=1e-3):
	model.eval()
	probe = LinearProbe(embedding_dim * 2).to(device)
	optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
	criterion = nn.BCELoss()

	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

	train_acc, val_acc, train_loss, val_loss = 0.0, 0.0, 0.0, 0.0

	for epoch in range(epochs):
		total_loss = 0.0
		n = 0
		for batch in loader:
			with torch.no_grad():
				#anchor = batch["original_embeddings"].squeeze(0).to(device).float()
				#positive = batch["positive_embeddings"].squeeze(0).to(device).float()
				negative = batch["negative_embeddings"].squeeze(0).to(device).float()
				flipped_index = batch["flipped_index"]
				positive_indices = batch["positive_indices"]

				#edge_positive = fully_connected_edges(positive.size(0), device)
				edge_negative = fully_connected_edges(negative.size(0), device)

				#pos_out = model(positive, edge_positive)
				neg_out = model(negative, edge_negative)

				#print([i for i in positive_indices if i != flipped_index][0], [i for i in range(neg_out.shape[0]) if i not in positive_indices][0])
				#print(torch.linalg.norm(neg_out[[i for i in positive_indices if i != flipped_index][0]] - neg_out[[i for i in range(neg_out.shape[0]) if i not in positive_indices][0]]))

			#x = torch.stack([pos_out, neg_out])
			#y = torch.tensor([0., 1.], device=device).unsqueeze(1)
			x = torch.stack([torch.concat((neg_out[flipped_index].squeeze(), neg_out[[i for i in positive_indices if i != flipped_index][0]].squeeze())), torch.concat((neg_out[flipped_index].squeeze(), neg_out[[i for i in range(neg_out.shape[0]) if i not in positive_indices][0]].squeeze()))])
			y = torch.tensor([1., 0.], device=device).unsqueeze(1)

			logits = probe(x)
			loss = criterion(logits, y)
			
			n += 1

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			
			if n % 100 == 0:
				print(f"[Linear Probe Train] Epoch {epoch+1}/{epochs} {n}/{len(loader)} | Loss: {total_loss/n:.4f}")
		
		for mode in ["train", "test"]:
			total_acc = 0
			total_loss = 0
			differentiation = 0
			n = 0

			all_logits = []
			average = 0

			for batch in (test_loader if mode == "test" else loader):
				with torch.no_grad():
					#anchor = batch["original_embeddings"].squeeze(0).to(device).float()
					#positive = batch["positive_embeddings"].squeeze(0).to(device).float()
					negative = batch["negative_embeddings"].squeeze(0).to(device).float()
					flipped_index = batch["flipped_index"]
					positive_indices = batch["positive_indices"]

					#edge_positive = fully_connected_edges(positive.size(0), device)
					edge_negative = fully_connected_edges(negative.size(0), device)

					#pos_out = model(positive, edge_positive)
					neg_out = model(negative, edge_negative)

				#x = torch.stack([pos_out, neg_out])
				#y = torch.tensor([0., 1.], device=device).unsqueeze(1)
				x = torch.stack([torch.concat((neg_out[flipped_index].squeeze(), neg_out[[i for i in positive_indices if i != flipped_index][0]].squeeze())), torch.concat((neg_out[flipped_index].squeeze(), neg_out[[i for i in range(neg_out.shape[0]) if i not in positive_indices][0]].squeeze()))])
				y = torch.tensor([1., 0.], device=device).unsqueeze(1)

				logits = probe(x)
				loss = criterion(logits, y)

				all_logits.append(logits)
				average += torch.sum(logits)/2
				
				n += 1
			
				#preds = (logits > 0.5).float().squeeze(1)

				#if torch.sum(preds) == 1:
				#	differentiation += 1

				#acc = (preds == y.squeeze(1)).float().mean().item()

				total_loss += loss.item()
				#total_acc += acc

				#print(f"[Linear Probe Test] Epoch {epoch+1}/{epochs} {n}/{len(test_loader)} | Loss: {total_loss/n:.4f} | Acc: {total_acc/n:.4f} | Differentiation: {differentiation/n:.4f}")
			
			average /= n

			for logits in all_logits:
				y = torch.tensor([1., 0.], device=device).unsqueeze(1)

				preds = (logits > average).float().squeeze(1)

				if torch.sum(preds) == 1:
					differentiation += 1

				acc = (preds == y.squeeze(1)).float().mean().item()

				total_acc += acc

			if mode == "train":
				train_acc = total_acc/n
				train_loss = total_loss/n
			else:
				val_acc = total_acc/n
				val_loss = total_loss/n

			print(f"[Linear Probe {mode.capitalize()}] Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} | Acc: {total_acc/n:.4f} | Differentiation: {differentiation/n:.4f}")

	print("Linear probe evaluation complete.")
	model.train()
	return train_acc, train_loss, val_acc, val_loss"""