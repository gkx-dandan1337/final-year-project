# file: train_chexnet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader import get_dataloaders
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


def main():
    # -----------------------
    # 1. Load data
    # -----------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        img_dir="data/images",
        batch_size=16          
    )

    # -----------------------
    # 2. Define model
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(weights="IMAGENET1K_V1")  # pretrained on ImageNet
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 14)       # 14 pathologies
    model = model.to(device)

    # -----------------------
    # 3. Loss + Optimizer
    # -----------------------
    criterion = nn.BCEWithLogitsLoss()   # multi-label
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # CheXNet used 1e-3

    # -----------------------
    # 4. Training loop
    # -----------------------
    EPOCHS = 15
    best_auc = 0.0

    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loader_tqdm.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        all_labels, all_outputs = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_labels.append(labels.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Concatenate all predictions
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)

        # Compute AUROC per class
        aucs = []
        for i in range(14):
            try:
                auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            except ValueError:
                auc = float("nan")  # if class not present in val set
            aucs.append(auc)

        mean_auc = np.nanmean(aucs)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f} "
            f"Mean AUROC: {mean_auc:.4f}")

        for i, auc in enumerate(aucs):
            print(f"  Class {i}: AUROC = {auc:.4f}")

        # ---- Save checkpoints ----
        torch.save(model.state_dict(), f"chexnet_epoch{epoch+1}.pth")

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), "chexnet_best.pth")
            print(f"✅ Saved new best model (AUROC = {best_auc:.4f})")

    print("🎉 Training finished. Model checkpoints saved.")

if __name__ == "__main__":
    main()