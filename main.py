from preprocess import FactContrastiveDataset
import pickle
from model import CustomGNN, fully_connected_edges
from linprobe import run_linear_probe

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

if __name__ == "__main__":
	with open("factkg_train.pickle", "rb") as f:
		factkg = pickle.load(f)
		dataset = FactContrastiveDataset(factkg, N=10, model_name="google/embeddinggemma-300m", cache_path="fact_gemma_cache.pkl")
	
	with open("factkg_test.pickle", "rb") as f:
		factkg_test = pickle.load(f)
		test_dataset = FactContrastiveDataset(factkg_test, N=10, model_name="google/embeddinggemma-300m", cache_path="fact_gemma_cache.pkl")

	writer = SummaryWriter("logs")
	
	model = CustomGNN(768, 512, 128)

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
	model = model.to(device)

	try:
		model.load_state_dict(torch.load(f"models/checkpoint-10.pth"))
	except:
		print("Could not load model")
		pass

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

	loader = DataLoader(dataset, batch_size=1, shuffle=True)
	triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2)
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	BATCH_SIZE = 8
	EPOCHS = 10

	for epoch in range(EPOCHS):
		model.train()
		total_loss = 0.0

		accum_loss = None
		n = 0
		for batch in loader:
			optimizer.zero_grad()

			anchor = batch["original_embeddings"].squeeze(0).to(device).float()
			positive = batch["positive_embeddings"].squeeze(0).to(device).float()
			negative = batch["negative_embeddings"].squeeze(0).to(device).float()

			# build fully connected edge lists
			edge_anchor = fully_connected_edges(anchor.size(0), device)
			edge_positive = fully_connected_edges(positive.size(0), device)
			edge_negative = fully_connected_edges(negative.size(0), device)

			# forward pass
			anchor_out = model(anchor, edge_anchor)
			positive_out = model(positive, edge_positive)
			negative_out = model(negative, edge_negative)

			loss = triplet_loss(anchor_out, positive_out, negative_out)
			#loss = torch.sum(torch.pow(5, cos(anchor_out, negative_out) - cos(anchor_out, positive_out)))
			#loss = F.relu(1 + torch.sum(cos(anchor_out, negative_out)) - torch.sum(cos(anchor_out, positive_out)))
			#loss = torch.sum(torch.pow(5, cos(anchor_out, negative_out) - cos(anchor_out, positive_out)))
			#loss = torch.exp(torch.linalg.norm(anchor_out - positive_out) - torch.linalg.norm(anchor_out - negative_out))
			#loss = torch.linalg.norm(anchor_out - positive_out) - torch.linalg.norm(anchor_out - negative_out)
			accum_loss = loss if accum_loss is None else loss + accum_loss

			if (n + 1) % BATCH_SIZE == 0:
				#accum_loss = torch.exp(accum_loss)
				accum_loss /= BATCH_SIZE
				accum_loss.backward()
				optimizer.step()
				
				print(f"Epoch {epoch+1}/{EPOCHS} {int(n/BATCH_SIZE)}/{int(len(loader)/BATCH_SIZE)} | Loss: {accum_loss.item():.4f} | Mag: {torch.linalg.norm(anchor_out):.4f} | Anchor+: {torch.linalg.norm(F.relu(anchor_out)):.4f} | Consistent+: {torch.linalg.norm(F.relu(positive_out)):.4f} | Contradiction+: {torch.linalg.norm(F.relu(negative_out)):.4f}")
				writer.add_scalar('Loss/train', accum_loss.item(), epoch * len(loader) + n)
				
				#print(cos(anchor_out, negative_out), cos(anchor_out, positive_out))
				accum_loss = None

				#run_linear_probe(model, dataset, dataset, device, 1, 128, 0.01)

			total_loss += loss.item()
			n += 1

		os.makedirs("models", exist_ok=True)
		torch.save(model.state_dict(), f"models/checkpoint-{epoch + 1}.pth")

		#run_linear_probe(model, dataset, test_dataset, device, 1, 128, 0.001)
		train_acc, train_loss, val_acc, val_loss = run_linear_probe(model, dataset, test_dataset, device, 1, 128, 0.001)
		writer.add_scalar('Probe_Accuracy/train', train_acc, epoch * len(loader))
		writer.add_scalar('Probe_Loss/train', train_loss, epoch * len(loader))
		writer.add_scalar('Probe_Accuracy/test', val_acc, epoch * len(loader))
		writer.add_scalar('Probe_Loss/test', val_loss, epoch * len(loader))

		avg_loss = total_loss / len(loader)
		print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

	print("Training complete.")

	writer.close()