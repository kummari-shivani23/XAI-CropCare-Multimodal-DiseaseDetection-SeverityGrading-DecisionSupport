import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision import models
from gradcam import GradCAM
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("models/best_baseline_model.pth", map_location=device))
model.eval().to(device)

target_layer = model.features[-1]
cam_generator = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def apply_gradcam(img_path, class_idx, save_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    cam = cam_generator.generate(input_tensor, class_idx)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(img.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlay)
