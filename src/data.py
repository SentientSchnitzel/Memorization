import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, transform=None, label_cols=None, frac=None):
        self.data = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.transform = transform
        
        self.label_cols = label_cols or [
            # "Frontal/Lateral", 
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"]
        
        self.data = self.data.dropna(subset=["Path"])  # Drop rows with no path

        if frac is not None:
            self.data = self.data.sample(frac=frac)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_root_dir, os.path.relpath(row["Path"], "CheXpert-v1.0-small/train"))
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        # Process labels: convert NaN to 0 and -1 to 0
        labels = row[self.label_cols].apply(pd.to_numeric, errors="coerce")
        labels = labels.fillna(0).replace(-1, 0).values.astype("float32")
        
        return image, labels


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    data_root = "../s194323/data/"
    
    train_dataset = CheXpertDataset(
        csv_file = f"{data_root}/train.csv",
        img_root_dir = f"{data_root}/train",
        transform = transform,
        frac=None)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print('Number of samples in train dataset:', len(train_dataset))
    
    val_dataset = CheXpertDataset(
        csv_file = f"{data_root}/valid.csv",
        img_root_dir = f"{data_root}/valid",
        transform = transform)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print('Number of samples in validation dataset:', len(val_dataset))
    
    for x, y in train_loader:
        print("x (images) shape:", x.shape)  # Should be [batch_size, 3, 224, 224]
        print("y (labels) shape:", y.shape)  # Should be [batch_size, num_labels]
        print("y (labels):", y)
        break  



