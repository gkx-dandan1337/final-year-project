# file: get_predictions.py

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from src.phase_1.dataloader import get_dataloaders

# -----------------------------
# 1. Load trained DenseNet
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 14)  # 14 pathologies

# Load checkpoint (adjust path as needed)
model.load_state_dict(torch.load("src/models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 2. Load test data
# -----------------------------
# Assuming your get_dataloaders returns (train, val, test)
_, _, test_loader = get_dataloaders(
    img_dir="data/images",
    batch_size=32,   # batch for faster inference
)

# -----------------------------
# 3. Run inference on test set
# -----------------------------
results = []

PNEUMONIA_IDX = 12  # confirm your label mapping! (check train CSV columns)

sigmoid = nn.Sigmoid()

results = []

with torch.no_grad():
    for inputs, labels, filenames in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = torch.sigmoid(outputs)

        pneumonia_probs = probs[:, PNEUMONIA_IDX].cpu().numpy()
        pneumonia_labels = labels[:, PNEUMONIA_IDX].cpu().numpy()
        predicted_labels = (pneumonia_probs >= 0.5).astype(int)

        for j, fname in enumerate(filenames):
            results.append({
                "filename": fname,  # now coming directly from dataset
                "true_pneumonia_label": int(pneumonia_labels[j]),
                "predicted_prob": float(pneumonia_probs[j]),
                "predicted_label": int(predicted_labels[j])
            })

df = pd.DataFrame(results)
df.to_csv("outputs/test_predictions.csv", index=False)
print("Saved predictions to outputs/test_predictions.csv")

# -----------------------------
# 4. Save to CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("src/phase2/results_pneumonia.csv", index=False)

print("✅ Saved predictions to results_pneumonia.csv")
