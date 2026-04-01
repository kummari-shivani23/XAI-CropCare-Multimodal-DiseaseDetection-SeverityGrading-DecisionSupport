import os
import sys
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# --------------------------------------------------
# DATASET & LOADER
# --------------------------------------------------
dataset = MultimodalDataset(
    image_dir="data/processed/grapes/test",
    metadata_csv="data/metadata/grape_weather.csv",
    transform=transform
)

loader = DataLoader(dataset, batch_size=8, shuffle=False)

class_names = dataset.class_names
print("Classes:", class_names)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = MultimodalFusionModel(num_classes=len(class_names))
model.load_state_dict(torch.load("models/multimodal_model.pth", map_location=device))
model.to(device)
model.eval()

# --------------------------------------------------
# TEST LOOP
# --------------------------------------------------
all_preds, all_labels = [], []

with torch.no_grad():
    for img, weather, label in loader:
        img, weather, label = img.to(device), weather.to(device), label.to(device)
        outputs = model(img, weather)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

# --------------------------------------------------
# METRICS
# --------------------------------------------------
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro")
rec = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

print("\n📊 MULTIMODAL TEST RESULTS")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"Macro-F1 : {f1:.4f}")

print("\n📄 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


cm=confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()
