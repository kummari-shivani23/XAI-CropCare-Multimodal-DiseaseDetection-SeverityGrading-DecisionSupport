# import torch
# import numpy as np
# from torch import nn

# print("🚀 train_baseline.py started")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Class counts (ORDER MATTERS!)
# class_counts = np.array([1180, 1383, 1076, 423])

# # Inverse frequency
# class_weights = 1.0 / class_counts

# # Normalize (optional but recommended)
# class_weights = class_weights / class_weights.sum()

# # Convert to tensor
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# criterion = nn.CrossEntropyLoss(weight=class_weights)



import os
import torch
import numpy as np
import csv

from torch import nn, optim
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score

import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_loader import get_dataloaders



# --------------------------------------------------
# DEBUG START
# --------------------------------------------------
print("🚀 train_baseline.py started")

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# CLASS WEIGHTS (STEP-9)
# ORDER MUST MATCH class_to_idx
# --------------------------------------------------
# black_rot, esca, leaf_blight, healthy
class_counts = np.array([1180, 1383, 1076, 423])

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------
DATA_DIR = "data/processed/grapes"
train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR)

print("Class mapping:", class_names)
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

# --------------------------------------------------
# MODEL (EfficientNet-B0)
# --------------------------------------------------
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 4)

# 3️⃣ 🔥 FREEZE BACKBONE (ADD THIS)

for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------
optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4)

# --------------------------------------------------
# TRAIN FUNCTION (STEP-10)
# --------------------------------------------------
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):

    best_val_f1 = 0.0
    os.makedirs("models", exist_ok=True)

    #csv logs

    os.makedirs("outputs/logs", exist_ok=True)
    log_file = open("outputs/logs/train_log.csv", "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow([
    "Epoch",
    "Train_Loss",
    "Train_Acc",
    "Train_F1",
    "Val_Loss",
    "Val_Acc",
    "Val_F1"
    ])


    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            if batch_idx % 10 == 0:
                print(
            f"Epoch {epoch+1} | "
            f"Batch {batch_idx}/{len(train_loader)} | "
            f"Loss: {loss.item():.4f}"
            )

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro")

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        # ---------- LOGS ----------
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

        print(f"Val   Loss: {val_loss/len(val_loader):.4f} | "
              f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        

        logger.writerow([
             epoch + 1,
              train_loss / len(train_loader),
                train_acc,
                train_f1,
            val_loss / len(val_loader),
            val_acc,
                val_f1
        ])


        # ---------- CHECKPOINT ----------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "models/best_baseline_model.pth")
            print("✅ Best model saved!")

    print("\n🎯 Training completed")
    print(f"Best Validation Macro-F1: {best_val_f1:.4f}")

    log_file.close()

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=5
    )


