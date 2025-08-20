# file: get_predictions.py

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from dataloader import get_dataloaders  # reuse your existing dataloader

# -----------------------------
# 1. Load trained DenseNet
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 14)  # 14 pathologies

# Load checkpoint (adjust path as needed)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 2. Load test data
# -----------------------------
# Assuming your get_dataloaders returns (train, val, test)
_, _, test_loader = get_dataloaders(
    img_dir="data/images",
    batch_size=32,   # batch for faster inference
    shuffle=False
)

# -----------------------------
# 3. Run inference on test set
# -----------------------------
results = []

PNEUMONIA_IDX = 14  # confirm your label mapping! (check train CSV columns)

sigmoid = nn.Sigmoid()

with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)              # raw logits
        probs = sigmoid(outputs)             # convert to probabilities

        pneumonia_probs = probs[:, PNEUMONIA_IDX].cpu().numpy()
        pneumonia_labels = labels[:, PNEUMONIA_IDX].cpu().numpy()

        # predicted label = threshold at 0.5
        predicted_labels = (pneumonia_probs >= 0.5).astype(int)

        # Save per image
        for i in range(len(inputs)):
            results.append({
                "filename": test_loader.dataset.df.iloc[i]["Image Index"],
                "true_pneumonia_label": int(pneumonia_labels[i]),
                "predicted_prob": float(pneumonia_probs[i]),
                "predicted_label": int(predicted_labels[i])
            })

# -----------------------------
# 4. Save to CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("results_pneumonia.csv", index=False)

print("✅ Saved predictions to results_pneumonia.csv")
