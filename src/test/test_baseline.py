import torch
import numpy as np
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import get_dataloaders

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_DIR = "data/processed/grapes"
_, _, test_loader, class_names = get_dataloaders(DATA_DIR)

print("Classes:", class_names)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 4)

model.load_state_dict(torch.load("models/best_baseline_model.pth", map_location=device))
model.to(device)
model.eval()

# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --------------------------------------------------
# METRICS
# --------------------------------------------------
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

print("\n📊 TEST RESULTS")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro-F1 : {f1:.4f}")

print("\n📄 Classification Report:")


class_names = test_loader.dataset.classes
print(classification_report(all_labels, all_preds, target_names=class_names))

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()
