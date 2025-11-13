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
		return self.fc(self.flatten(x))
		#return torch.sum(self.fc(x), dim=1)

# -------------------------------------------------------
# LINEAR PROBE TRAINING LOOP
# -------------------------------------------------------
def run_linear_probe(model, dataset, test_dataset, device, epochs=10, embedding_dim=128, lr=1e-3):
	model.eval()  # freeze encoder
	probe = LinearProbe(embedding_dim).to(device)  # match your model's output dim
	optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
	criterion = nn.BCEWithLogitsLoss()

	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

	train_acc, val_acc, train_loss, val_loss = 0.0, 0.0, 0.0, 0.0

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

				pos_out = model(positive, edge_positive)
				neg_out = model(negative, edge_negative)

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
		
		for mode in ["train", "test"]:
			total_acc = 0
			total_loss = 0
			differentiation = 0
			n = 0

			for batch in (test_loader if mode == "test" else loader):
				with torch.no_grad():
					#anchor = batch["original_embeddings"].squeeze(0).to(device).float()
					positive = batch["positive_embeddings"].squeeze(0).to(device).float()
					negative = batch["negative_embeddings"].squeeze(0).to(device).float()

					edge_positive = fully_connected_edges(positive.size(0), device)
					edge_negative = fully_connected_edges(negative.size(0), device)

					pos_out = model(positive, edge_positive)
					neg_out = model(negative, edge_negative)

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

				#print(f"[Linear Probe Test] Epoch {epoch+1}/{epochs} {n}/{len(test_loader)} | Loss: {total_loss/n:.4f} | Acc: {total_acc/n:.4f} | Differentiation: {differentiation/n:.4f}")
			
			if mode == "train":
				train_acc = total_acc/n
				train_loss = total_loss/n
			else:
				val_acc = total_acc/n
				val_loss = total_loss/n

			print(f"[Linear Probe {mode.capitalize()}] Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} | Acc: {total_acc/n:.4f} | Differentiation: {differentiation/n:.4f}")

	print("Linear probe evaluation complete.")
	model.train()
	return train_acc, train_loss, val_acc, val_loss
