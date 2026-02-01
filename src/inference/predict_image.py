import torch
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
#IMAGE_PATH = "data/raw/grapes/Esca/0ab276b8-1342-4b3b-a87d-620880112e9c___FAM_B.Msls 1009.JPG"  
IMAGE_PATH="data/processed/grapes/train/Healthy/fed74aa9-511b-4958-824b-41066b2e5406___Mt.N.V_HL 8918.JPG "# <-- change this
MODEL_PATH = "models/best_baseline_model.pth"

class_names = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# TRANSFORMS (same as val/test)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, len(class_names))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # add batch dim

# -----------------------------
# PREDICTION
# -----------------------------
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, dim=1)

predicted_class = class_names[pred.item()]
confidence = conf.item()

print("🧠 Prediction Result")
print("--------------------")
print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence:.4f}")
