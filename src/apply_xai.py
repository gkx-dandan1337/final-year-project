import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from dataloader import get_dataloaders
import numpy as np
import cv2
import pandas as pd
import os


# -----------------------
# 1. Custom forward wrapper
# -----------------------
def densenet_no_inplace(num_classes=14):
    model = models.densenet121(weights="IMAGENET1K_V1")
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    # Monkey-patch forward to disable inplace ReLU
    def forward_no_inplace(x):
        features = model.features(x)
        out = F.relu(features, inplace=False)  # ✅ force out-of-place
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = model.classifier(out)
        return out

    model.forward = forward_no_inplace
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # 2. Load model checkpoint
    # -----------------------
    model = densenet_no_inplace(num_classes=14)
    checkpoint = torch.load("src/models/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval().to(device)

    # -----------------------
    # 3. Grad-CAM helper
    # -----------------------
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None

            target_layer.register_forward_hook(self.forward_hook)
            target_layer.register_full_backward_hook(self.backward_hook)

        def forward_hook(self, module, input, output):
            self.activations = output.clone().detach().requires_grad_(True)

        def backward_hook(self, module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def generate(self, input_tensor, class_idx):
            output = self.model(input_tensor)
            score = output[0, class_idx]
            self.model.zero_grad()
            score.backward()
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activations = self.activations[0]
            for i in range(len(pooled_gradients)):
                activations[i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activations, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() != 0:
                heatmap /= heatmap.max()
            return heatmap

    # -----------------------
    # 4. Data loader
    # -----------------------
    _, _, test_loader = get_dataloaders(img_dir="data/images", batch_size=1)

    os.makedirs("src/outputs/gradcam", exist_ok=True)
    results = []

    gradcam = GradCAM(model, model.features[-1])  # last conv layer
    pneumonia_idx = 12  # Pneumonia column in your CSV

    # -----------------------
    # 5. Run Grad-CAM
    # -----------------------
    for i, (images, labels, paths) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)  # [1, 14]
        probs_all = torch.sigmoid(outputs).cpu().detach().numpy()[0]
        prob = probs_all[pneumonia_idx]
        pred = 1 if prob > 0.5 else 0

        heatmap = gradcam.generate(images, class_idx=pneumonia_idx)
        heatmap = cv2.resize(heatmap, (images.shape[2], images.shape[3]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = np.uint8(255 * img_np)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        fname = os.path.basename(paths if isinstance(paths, str) else paths[0])
        out_path = f"src/outputs/gradcam/{fname}"
        cv2.imwrite(out_path, overlay)

        results.append([fname, labels[0][pneumonia_idx].item(), prob, pred])

    # -----------------------
    # 6. Save predictions
    # -----------------------
    df = pd.DataFrame(results, columns=["image", "true_label", "probability", "pred_label"])
    df.to_csv("src/outputs/predictions.csv", index=False)

    print("✅ Grad-CAM explanations saved in src/outputs/gradcam/")
    print("✅ Predictions saved in src/outputs/predictions.csv")


if __name__ == "__main__":
    main()
