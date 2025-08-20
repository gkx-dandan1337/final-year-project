import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# -----------------------------
# 1. Load your trained DenseNet model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 14)   # 14 pathologies
model.load_state_dict(torch.load("src/models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 2. Dataset + Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

class PredictionSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, preds_csv, img_dir, transform=None, subset_filter=None):
        df = pd.read_csv(preds_csv)

        # Apply filtering if provided (example: only pneumonia false negatives)
        if subset_filter:
            df = subset_filter(df)

        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image Index"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = img

        # return tensor + metadata (filename, prob, label)
        return img_tensor, row["Image Index"], row["Pneumonia_label"], row["Pneumonia_prob"]

# Example subset filter: False Negatives (label=1 but prob<0.5)
def subset_false_negatives(df):
    return df[(df["Pneumonia_label"] == 1) & (df["Pneumonia_prob"] < 0.5)]

# -----------------------------
# 3. Load dataset
# -----------------------------
preds_csv = "outputs/get_predictions.csv"
img_dir = "data/images"

test_dataset = PredictionSubsetDataset(
    preds_csv, img_dir, transform=transform, subset_filter=subset_false_negatives
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -----------------------------
# 4. Grad-CAM implementation
# -----------------------------
gradients = []
activations = []

def save_gradient(module, grad_in, grad_out):
    gradients.append(grad_out[0])

def save_activation(module, input, output):
    activations.append(output)

# hook last conv layer
target_layer = model.features[-1]
target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)

def grad_cam(input_tensor, class_idx):
    gradients.clear()
    activations.clear()

    output = model(input_tensor)
    loss = output[0, class_idx]
    model.zero_grad()
    loss.backward()

    grad = gradients[0].detach().cpu().numpy()
    act = activations[0].detach().cpu().numpy()

    weights = np.mean(grad, axis=(2, 3))[0, :]
    cam = np.zeros(act.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam

# -----------------------------
# 5. Run Grad-CAM on Pneumonia class
# -----------------------------
PNEUMONIA_IDX = 12  # double-check mapping!

save_dir = "outputs/gradcam"
os.makedirs(save_dir, exist_ok=True)

for i, (img_tensor, fname, true_label, pred_prob) in enumerate(tqdm(test_loader)):
    img_tensor = img_tensor.to(device)

    heatmap = grad_cam(img_tensor, class_idx=PNEUMONIA_IDX)
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    # Reload original image for overlay
    orig_img = Image.open(os.path.join(img_dir, fname[0])).convert("RGB")
    orig_resized = np.array(orig_img.resize((224,224)))
    overlay = cv2.addWeighted(orig_resized, 0.5, heatmap, 0.5, 0)

    save_path = f"{save_dir}/{fname[0]}_gradcam.jpg"
    cv2.imwrite(save_path, overlay)

    if i < 5:
        plt.imshow(overlay)
        plt.title(f"{fname[0]} | Label={true_label.item()} | Prob={pred_prob.item():.2f}")
        plt.axis("off")
        plt.show()
