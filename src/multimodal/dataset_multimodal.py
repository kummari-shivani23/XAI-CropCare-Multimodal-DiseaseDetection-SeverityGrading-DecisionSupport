import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, image_dir, metadata_csv, transform=None):
        self.image_ds = datasets.ImageFolder(image_dir, transform=transform)
        self.meta = pd.read_csv(metadata_csv)
        self.class_names = self.image_ds.classes

    def __len__(self):
        return len(self.image_ds)

    def __getitem__(self, idx):
        image, label = self.image_ds[idx]

        # weather features
        weather = self.meta.iloc[idx][
            ["temperature", "humidity", "rainfall", "wind_speed"]
        ].values.astype("float32")

        return image, torch.tensor(weather), label
