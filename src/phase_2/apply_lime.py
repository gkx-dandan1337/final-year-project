import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries

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
# 3. Helper: prediction function for LIME
# -----------------------------
def predict_fn(images_np):
    model.eval()
    batch = []
    for img in images_np:
        pil = Image.fromarray(img.astype("uint8")).convert("RGB")
        batch.append(transform(pil))
    batch = torch.stack(batch).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)  # multi-label
    return probs.cpu().numpy()

# -----------------------------
# 4. Load predictions CSV
# -----------------------------
preds = pd.read_csv("outputs/get_predictions.csv")

# Example subset: False Negatives
subset = preds[(preds["Pneumonia_label"]==1) & (preds["Pneumonia_prob"]<0.5)]

# -----------------------------
# 5. LIME on subset and save
# -----------------------------
explainer = lime_image.LimeImageExplainer()
PNEUMONIA_IDX = 12  # confirm mapping!

save_dir = "outputs/lime"
os.makedirs(save_dir, exist_ok=True)

for idx, row in subset.iterrows():
    fname = row["Image Index"]
    img_path = os.path.join("data/images", fname)

    orig_img = Image.open(img_path).convert("RGB").resize((224,224))
    img_np = np.array(orig_img)

    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        label=PNEUMONIA_IDX,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    overlay = mark_boundaries(temp/255.0, mask)

    # Save to file
    save_path = os.path.join(save_dir, f"{fname}_lime.jpg")
    plt.imsave(save_path, overlay)

    # Optional: preview first few
    if idx < 3:
        plt.imshow(overlay)
        plt.title(f"LIME: {fname} | Label={row['Pneumonia_label']} | Prob={row['Pneumonia_prob']:.2f}")
        plt.axis("off")
        plt.show()

print(f"Saved {len(subset)} LIME explanations to {save_dir}")
