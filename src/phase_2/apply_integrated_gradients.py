import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load trained model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 14)
model.load_state_dict(torch.load("src/models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 2. Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# -----------------------------
# 3. Integrated Gradients function
# -----------------------------
def integrated_gradients(input_tensor, target_class, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)

    # Scale inputs between baseline and input
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    for scaled in scaled_inputs:
        scaled.requires_grad = True
        out = model(scaled)
        loss = out[0, target_class]
        model.zero_grad()
        loss.backward()
        grads.append(scaled.grad.detach().cpu().numpy())

    grads = np.array(grads)  # (steps+1, C, H, W)
    avg_grads = grads[:-1].mean(axis=0)  # average over steps
    delta = (input_tensor - baseline).detach().cpu().numpy()[0]  # (C,H,W)
    attributions = delta * avg_grads  # IG formula
    return attributions

# -----------------------------
# 4. Load predictions CSV
# -----------------------------
preds = pd.read_csv("outputs/get_predictions.csv")

# Example subset: True Positives (label=1 and prob>0.5)
subset = preds[(preds["Pneumonia_label"]==1) & (preds["Pneumonia_prob"]>=0.5)]

# -----------------------------
# 5. Run IG on subset
# -----------------------------
PNEUMONIA_IDX = 12  # confirm this is correct for your label mapping!
save_dir = "outputs/integrated_gradients"
os.makedirs(save_dir, exist_ok=True)

for idx, row in subset.iterrows():
    fname = row["Image Index"]
    img_path = os.path.join("data/images", fname)

    orig_img = Image.open(img_path).convert("RGB").resize((224,224))
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    attributions = integrated_gradients(input_tensor, PNEUMONIA_IDX, steps=50)

    # Convert attributions to heatmap
    attr_sum = np.abs(attributions).sum(axis=0)  # sum over channels
    attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)

    # Overlay heatmap on original image
    overlay = plt.cm.jet(attr_norm)[..., :3] * 255
    overlay = overlay.astype(np.uint8)
    blended = (0.5 * np.array(orig_img) + 0.5 * overlay).astype(np.uint8)

    # Save
    save_path = os.path.join(save_dir, f"{fname}_ig.jpg")
    Image.fromarray(blended).save(save_path)

    # Optional: preview a few
    if idx < 3:
        plt.imshow(blended)
        plt.title(f"IG: {fname} | Label={row['Pneumonia_label']} | Prob={row['Pneumonia_prob']:.2f}")
        plt.axis("off")
        plt.show()

print(f"Saved {len(subset)} Integrated Gradients explanations to {save_dir}")
