# file: train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader import get_dataloaders
from tqdm import tqdm   # progress bar

# -----------------------
# 1. Load data
# -----------------------
train_loader, val_loader, test_loader = get_dataloaders(
    img_dir="data/images",
    batch_size=32
)

# -----------------------
# 2. Define model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained DenseNet-121
model = models.densenet121(pretrained=True)

# Replace final classifier (DenseNet has 1024 features -> 1 output for binary)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)

model = model.to(device)

# -----------------------
# 3. Loss + Optimizer
# -----------------------
criterion = nn.BCEWithLogitsLoss()   # binary classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------
# 4. Training loop
# -----------------------
EPOCHS = 10   # start with 10; increase if val_loss keeps improving

for epoch in range(EPOCHS):
    # -------- Training --------
    model.train()
    running_loss = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loader_tqdm.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    avg_train_loss = running_loss / len(train_loader)

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    print(f"\nEpoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Val Acc: {val_acc:.4f}")

# -----------------------
# 5. Save checkpoint
# -----------------------
torch.save(model.state_dict(), "baseline_densenet121.pth")
print("✅ Model saved as baseline_densenet121.pth")
