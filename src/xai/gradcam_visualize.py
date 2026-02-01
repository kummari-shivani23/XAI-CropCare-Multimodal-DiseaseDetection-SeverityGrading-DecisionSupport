import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image

# ---- GradCAM Class ----
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward()

        grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (grads * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # avoid division by zero

        return cam.detach().cpu().numpy()[0]

# ---- Device & Model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("models/best_baseline_model.pth", map_location=device))
model.eval().to(device)

target_layer = model.features[-3]
cam_generator = GradCAM(model, target_layer)

# ---- Image Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---- Grad-CAM Function ----
def apply_gradcam(img_path, class_idx, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠ Could not open image {img_path}: {e}")
        return

    input_tensor = transform(img).unsqueeze(0).to(device)
    cam = cam_generator.generate(input_tensor, class_idx)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(img.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlay)

# ---- Main ----
if __name__ == "__main__":
    os.makedirs("outputs/gradcam", exist_ok=True)

    test_images = {
        "black_rot": "data/processed/grapes/test/black_rot",
        "esca": "data/processed/grapes/test/esca",
        "leaf_blight": "data/processed/grapes/test/leaf_blight",
        "healthy": "data/processed/grapes/test/healthy"
    }

    class_to_idx = {"black_rot": 0, "esca": 1, "leaf_blight": 2, "healthy": 3}

    for cls, folder in test_images.items():
        save_dir = f"outputs/gradcam/{cls}"
        os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(folder):
            print(f"⚠ Folder not found: {folder}")
            continue

        images = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))][:5]

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            save_path = os.path.join(save_dir, img_name)
            apply_gradcam(img_path, class_to_idx[cls], save_path)

    print("✅ Grad-CAM images saved in outputs/gradcam/")
