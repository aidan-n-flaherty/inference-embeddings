import torch
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
		)
		self.flatten = nn.Flatten()

	def forward(self, x):
		#return self.fc(self.flatten(x))
		return torch.sum(self.fc(x), dim=1)

def run_linear_probe(model, dataset, test_dataset, validation_dataset, device, epochs=10, embedding_dim=128, lr=1e-3):
	model.eval()
	probe = LinearProbe(embedding_dim).to(device)
	optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
	criterion = nn.BCEWithLogitsLoss()

	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
	val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

	train_acc, test_acc, val_acc, train_loss, test_loss, val_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

	for epoch in range(epochs):
		total_loss = 0.0
		n = 0
		for batch in loader:
			with torch.no_grad():
				#anchor = batch["original_embeddings"].squeeze(0).to(device).float()
				positive = batch["positive_embeddings"].squeeze(0).to(device).float()
				negative = batch["negative_embeddings"].squeeze(0).to(device).float()

				edge_positive = fully_connected_edges(positive.size(0), device)
				edge_negative = fully_connected_edges(negative.size(0), device)

				pos_out, _ = model(positive, edge_positive)
				neg_out, _ = model(negative, edge_negative)

			x = torch.stack([pos_out, neg_out])
			y = torch.tensor([0., 1.], device=device).unsqueeze(1)

			logits = probe(x)
			loss = criterion(logits, y)
			
			n += 1

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			
			if n % 100 == 0:
				print(f"[Linear Probe Train] Epoch {epoch+1}/{epochs} {n}/{len(loader)} | Loss: {total_loss/n:.4f}")
		
		for mode in ["train", "test", "validation"]:
			total_acc = 0
			total_loss = 0
			differentiation = 0
			n = 0

			tp, fp, tn, fn = 0, 0, 0, 0

			for batch in (test_loader if mode == "test" else val_loader if mode == "validation" else loader):
				with torch.no_grad():
					#anchor = batch["original_embeddings"].squeeze(0).to(device).float()
					positive = batch["positive_embeddings"].squeeze(0).to(device).float()
					negative = batch["negative_embeddings"].squeeze(0).to(device).float()

					edge_positive = fully_connected_edges(positive.size(0), device)
					edge_negative = fully_connected_edges(negative.size(0), device)

					pos_out, _ = model(positive, edge_positive)
					neg_out, _ = model(negative, edge_negative)

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
			elif mode == "test":
				test_acc = total_acc/n
				test_loss = total_loss/n
			else:
				val_acc = total_acc/n
				val_loss = total_loss/n

			f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
			print(f"[Linear Probe {mode.capitalize()}] Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} | Acc: {total_acc/n:.4f} | Differentiation: {differentiation/n:.4f} | F1: {f1:.4f}")

	print("Linear probe evaluation complete.")
	model.train()
	return train_acc, train_loss, test_acc, test_loss, val_acc, val_loss

import pickle
from preprocess import SyntheticContrastiveDataset
from model import CustomGNN

if __name__ == "__main__":
	with open("synthetic_data_train.pkl", "rb") as f:
		synthetic_train = pickle.load(f)
		dataset = SyntheticContrastiveDataset(synthetic_train, model_name="sentence-transformers/nli-mpnet-base-v2", cache_path="fact_nli_mpnet_cache.pkl")

	with open("synthetic_data_test.pkl", "rb") as f:
		synthetic_test = pickle.load(f)
		test_dataset = SyntheticContrastiveDataset(synthetic_test, model_name="sentence-transformers/nli-mpnet-base-v2", cache_path="fact_nli_mpnet_cache.pkl")
	
	with open("synthetic_data_validation.pkl", "rb") as f:
		synthetic_validation = pickle.load(f)
		validation_dataset = SyntheticContrastiveDataset(synthetic_validation, model_name="sentence-transformers/nli-mpnet-base-v2", cache_path="fact_nli_mpnet_cache.pkl")
	
	out_dim = 256
	model = CustomGNN(768, 1024, out_dim)

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
	model = model.to(device)

	try:
		d = torch.load(f"models/checkpoint-100.pth", map_location=device)
		model.load_state_dict(d, strict=True)
	except:
		import traceback
		print(traceback.format_exc())
		print("Could not load model")
		pass

	run_linear_probe(model, dataset, test_dataset, validation_dataset, device, 2, out_dim, 0.0001)
