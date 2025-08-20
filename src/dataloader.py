# file: dataloader.py
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------------
# Custom Dataset
# -----------------------
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # all disease columns (skip first 2: Image Index + Patient ID)
        self.label_columns = self.data.columns[2:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]   # column 0 = filename
        labels   = self.data.iloc[idx, 2:].values.astype("float32")  # columns 2..end = one-hot labels
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(labels, dtype=torch.float32)  # shape [14]
        
        # return image, labels, and the filename (not the whole path if you prefer clean names)
        return image, labels, img_name


# -----------------------
# Transforms
# -----------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------
# Dataloader function
# -----------------------
def get_dataloaders(img_dir="data\images", batch_size=32,
                    train_csv="data/train.csv", val_csv="data/val.csv", test_csv="data/test.csv"):
    train_dataset = ChestXrayDataset(train_csv, img_dir, transform=train_transform)
    val_dataset   = ChestXrayDataset(val_csv, img_dir, transform=val_test_transform)
    test_dataset  = ChestXrayDataset(test_csv, img_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
