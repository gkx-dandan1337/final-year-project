# Baseline Evaluation for Pneumonia Detection
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import get_dataloaders
from torchvision import models


def main():
    # -----------------------
    # 1. Load Data
    # -----------------------
    _, _, test_loader = get_dataloaders(
        img_dir="data/images",
        batch_size=32,
    )

    # -----------------------
    # 2. Load Model
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 14)  # trained with 14 outputs
    model = model.to(device)

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Pneumonia index in CheXNet dataset (verify this matches your CSV order)
    pneumonia_idx = 12

    # -----------------------
    # 3. Run Inference
    # -----------------------
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)               # [batch, 14]
            probs = torch.sigmoid(outputs)        # [batch, 14]

            y_true.extend(labels[:, pneumonia_idx].cpu().numpy())
            y_pred.extend(probs[:, pneumonia_idx].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # -----------------------
    # 4. Metrics
    # -----------------------
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_binary)
    acc = accuracy_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)

    print("=== Baseline Evaluation (Pneumonia) ===")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # -----------------------
    # 5. ROC Curve
    # -----------------------
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Pneumonia)")
    plt.legend()
    plt.show()

    # -----------------------
    # 6. PR Curve
    # -----------------------
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.plot(rec, prec, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Pneumonia)")
    plt.legend()
    plt.show()

    # -----------------------
    # 7. Confusion Matrix Plot
    # -----------------------
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Pneumonia", "Pneumonia"],
                yticklabels=["No Pneumonia", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Pneumonia)")
    plt.show()


    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Pneumonia)")
    plt.legend()
    plt.savefig("roc_curve.png", dpi=300)   # <--- saves image
    plt.close()

    # Save PR curve
    plt.figure(figsize=(6,6))
    plt.plot(rec, prec, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Pneumonia)")
    plt.legend()
    plt.savefig("pr_curve.png", dpi=300)
    plt.close()

    # Save confusion matrix
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Pneumonia", "Pneumonia"],
                yticklabels=["No Pneumonia", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Pneumonia)")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()