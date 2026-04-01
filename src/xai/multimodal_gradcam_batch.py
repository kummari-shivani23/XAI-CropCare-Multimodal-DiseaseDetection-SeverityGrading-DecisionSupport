import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import cv2
import numpy as np
import random
from PIL import Image
from torchvision import transforms, datasets

from multimodal.fusion_modal import MultimodalFusionModel
from xai.gradcam_multimodal import GradCAM

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = "data/processed/grapes/test"
OUT_DIR = "outputs/gradcam_multimodal"
SAMPLES_PER_CLASS = 5

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# MODEL
# -------------------------------
model = MultimodalFusionModel(num_classes=4)
model.load_state_dict(torch.load("models/multimodal_model.pth", map_location=device))
model.to(device)


#unfreeze the modal for gradcam generation only the target layer
for param in model.cnn.features[-1][0].parameters():
    param.requires_grad = True

model.eval()



target_layer = model.cnn.features[-1][0]
gradcam = GradCAM(model, target_layer)

# -------------------------------
# TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------------
# DATASET
# -------------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

# -------------------------------
# DUMMY WEATHER (consistent)
# -------------------------------
default_weather = torch.tensor([[28.0, 80.0, 10.0, 5.0]], dtype=torch.float32).to(device)

# -------------------------------
# GENERATE CAMs
# -------------------------------
for class_idx, class_name in enumerate(class_names):
    print(f"🔍 Processing {class_name}")

    class_dir = os.path.join(OUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # get all indices of this class
    indices = [i for i, (_, y) in enumerate(dataset.samples) if y == class_idx]
    sampled = random.sample(indices, min(SAMPLES_PER_CLASS, len(indices)))

    for i, idx in enumerate(sampled):
        img_tensor, _ = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        cam = gradcam.generate(img_tensor, default_weather)

        # load original image for overlay
        img_path, _ = dataset.samples[idx]
        original = Image.open(img_path).convert("RGB")
        original = original.resize((224,224))
        img_np = np.array(original)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        save_path = os.path.join(class_dir, f"cam_{i+1}.jpg")
        cv2.imwrite(save_path, overlay)

    print(f"✅ Saved {len(sampled)} CAMs for {class_name}")

print("\n🎯 Multimodal Grad-CAM batch generation complete")
