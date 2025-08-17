import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with image names + labels
            img_dir (str): Directory with all images
            transform (callable, optional): Optional transform to apply to images
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image Index"]
        label = self.data.iloc[idx]["label"]

        # Full path
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if given
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(batch_size=32, img_dir="data/images"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ChestXrayDataset("data/train.csv", img_dir, transform)
    val_dataset   = ChestXrayDataset("data/val.csv", img_dir, transform)
    test_dataset  = ChestXrayDataset("data/test.csv", img_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


from dataloader import get_dataloaders

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        img_dir="data/images",
        batch_size=32
    )


    # Sanity check: print one batch
    for images, labels in train_loader:
        print("Batch image tensor shape:", images.shape)
        print("Batch label tensor shape:", labels.shape)
        print("Labels:", labels[:10])  # print first 10 labels
        break
