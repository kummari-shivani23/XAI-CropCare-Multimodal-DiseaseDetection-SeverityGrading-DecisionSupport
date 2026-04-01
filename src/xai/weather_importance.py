import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from multimodal.fusion_modal import MultimodalFusionModel

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "data/processed/grapes/test/Black_rot/0aff8add-93ad-4099-97ae-23515744e620___FAM_B.Rot 0748.JPG"  # any valid image
MODEL_PATH = "models/multimodal_model.pth"
OUTPUT_PATH = "outputs/xai_weather/weather_importance.png"

FEATURE_NAMES = ["Temperature", "Humidity", "Rainfall", "Wind Speed"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LOAD MODEL
# -----------------------------
model = MultimodalFusionModel(num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# IMAGE PREPROCESS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)  # ✅ FIXED

# -----------------------------
# DEFAULT WEATHER (normalized)
# -----------------------------
weather = torch.tensor(
    [[0.6, 0.7, 0.4, 0.3]],  # temp, humidity, rainfall, wind
    dtype=torch.float32,
    requires_grad=True
).to(device)

# -----------------------------
# FORWARD + BACKWARD
# -----------------------------
output = model(image_tensor, weather)
pred_class = output.argmax(dim=1)

score = output[0, pred_class]
score.backward()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importance = weather.grad.abs().detach().cpu().numpy()[0]
importance = importance / importance.sum()  # normalize

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(7, 4))
plt.bar(FEATURE_NAMES, importance)
plt.ylabel("Relative Importance")
plt.title("Weather Feature Importance (Gradient-based)")
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print("✅ Weather feature importance saved to:", OUTPUT_PATH)
