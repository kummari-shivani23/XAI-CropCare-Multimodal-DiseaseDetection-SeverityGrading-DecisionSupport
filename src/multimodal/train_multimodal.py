import os
import csv


import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from multimodal.dataset_multimodal import MultimodalDataset
from multimodal.fusion_modal import MultimodalFusionModel


# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# DATASET & LOADER
# --------------------------------------------------
dataset = MultimodalDataset(
    image_dir="data/processed/grapes/train",
    metadata_csv="data/metadata/grape_weather.csv",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0   # safe for Windows
)

print("Total batches:", len(loader))


# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = MultimodalFusionModel(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


# --------------------------------------------------
# CSV LOGGING SETUP
# --------------------------------------------------
os.makedirs("outputs/logs", exist_ok=True)
log_path = "outputs/logs/multimodal_train_log.csv"

log_file = open(log_path, "w", newline="")
logger = csv.writer(log_file)
logger.writerow(["epoch", "loss", "accuracy"])


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (img, weather, label) in enumerate(loader):
        img = img.to(device)
        weather = weather.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        out = model(img, weather)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        # 🔹 batch-level log (lightweight)
        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = total_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    # 🔹 epoch summary
    print(
        f"Epoch {epoch+1} Summary → "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_acc:.4f}"
    )

    # 🔹 write CSV
    logger.writerow([epoch+1, epoch_loss, epoch_acc])


# --------------------------------------------------
# CLEANUP & SAVE
# --------------------------------------------------
log_file.close()

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/multimodal_model.pth")

print("✅ Multimodal training complete")
print("📄 Logs saved to:", log_path)
print("💾 Model saved to: models/multimodal_model.pth")
