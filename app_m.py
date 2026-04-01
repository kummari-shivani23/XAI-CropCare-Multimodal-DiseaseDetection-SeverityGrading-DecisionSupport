# import torch
# import torch.nn.functional as F
# from torchvision import models, transforms
# import gradio as gr
# from PIL import Image
# import numpy as np
# import cv2

# # -----------------------------
# # CONFIG
# # -----------------------------
# MODEL_PATH = "models/best_baseline_model.pth"

# class_names = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -----------------------------
# # TRANSFORM
# # -----------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = models.efficientnet_b0(weights=None)
# num_features = model.classifier[1].in_features
# model.classifier[1] = torch.nn.Linear(num_features, len(class_names))

# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

# # -----------------------------
# # GRAD-CAM (simple version)
# # -----------------------------
# def generate_gradcam(model, image_tensor):
#     image_tensor.requires_grad = True

#     outputs = model(image_tensor)
#     pred_class = outputs.argmax()

#     outputs[0, pred_class].backward()

#     gradients = image_tensor.grad[0].cpu().numpy()
#     heatmap = np.mean(gradients, axis=0)

#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= (np.max(heatmap) + 1e-8)

#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     return heatmap


# def generate_gradcam_overlay(model, image_tensor, original_image):
#     image_tensor.requires_grad = True

#     outputs = model(image_tensor)
#     pred_class = outputs.argmax()

#     outputs[0, pred_class].backward()

#     gradients = image_tensor.grad[0].cpu().numpy()
#     heatmap = np.mean(gradients, axis=0)

#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= (np.max(heatmap) + 1e-8)

#     # Resize heatmap
#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap = np.uint8(255 * heatmap)

#     # Apply color map
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     # Convert original image to numpy
#     original = original_image.resize((224, 224))
#     original = np.array(original)

#     # Overlay heatmap on original image
#     overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

#     return overlay
# # -----------------------------
# # LOGIC
# # -----------------------------
# def get_severity(conf):
#     if conf > 0.85:
#         return "Severe"
#     elif conf > 0.65:
#         return "Moderate"
#     else:
#         return "Mild"

# def get_recommendation(disease):
#     return {
#         "Black_rot": "Apply Mancozeb fungicide and remove infected leaves.",
#         "Esca": "Prune affected parts and apply fungicide spray.",
#         "Leaf_blight": "Use copper-based fungicide and maintain airflow.",
#         "Healthy": "No disease detected. Maintain proper crop care."
#     }.get(disease, "Consult agricultural expert.")

# # -----------------------------
# # PREDICTION FUNCTION
# # -----------------------------
# def predict(image):
#     img = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(img)
#         probs = F.softmax(outputs, dim=1)
#         conf, pred = torch.max(probs, dim=1)

#     disease = class_names[pred.item()]
#     confidence = conf.item()

#     # Grad-CAM
#     #heatmap = generate_gradcam(model, img.clone())

#     heatmap = generate_gradcam_overlay(model, img.clone(), image)
#     severity = get_severity(confidence)
#     recommendation = get_recommendation(disease)

#     result = f"""
#     🔍 Prediction: {disease}
#     📊 Confidence: {confidence:.4f}
#     ⚠ Severity: {severity}

#     💊 Recommendation:
#     {recommendation}
#     """

#     return result, heatmap

# # -----------------------------
# # UI
# # -----------------------------
# with gr.Blocks() as demo:

#     gr.Markdown("# 🌿 Crop Disease Detection (Baseline Model)")
#     gr.Markdown("Upload a leaf image to detect disease with Grad-CAM explanation")

#     image_input = gr.Image(type="pil", label="Upload Leaf Image")

#     btn = gr.Button("Analyze")

#     output_text = gr.Textbox(label="Prediction Result")
#     output_image = gr.Image(label="Grad-CAM Heatmap")

#     btn.click(predict,
#               inputs=image_input,
#               outputs=[output_text, output_image])

# demo.launch()



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from torchvision import transforms, models

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/best_baseline_model.pth"
CLASS_NAMES = ["Black_rot", "Esca","Healthy", "Leaf_blight"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# GRAD-CAM CLASS
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)

        score = output[:, class_idx]
        score.backward()

        grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (grads * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()[0]

# ✅ correct target layer for EfficientNet
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(image):

    image = image.convert("RGB")
    img_resized = image.resize((224, 224))

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ---- Prediction ----
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    class_idx = pred.item()
    predicted_class = CLASS_NAMES[class_idx]
    confidence = conf.item()

    # ---- Grad-CAM ----
    cam = gradcam.generate(input_tensor, class_idx)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(img_resized)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # ---- Severity Logic ----
    if predicted_class == "Healthy":
        severity = "🟢 None"
        recommendation = "Leaf is healthy. Maintain regular care."
    else:
        if confidence > 0.9:
            severity = "🔴 Severe"
        elif confidence > 0.75:
            severity = "🟠 Moderate"
        else:
            severity = "🟡 Mild"

        recommendation = {
            "Black_rot": "Use Mancozeb or Copper fungicide.",
            "Esca": "Remove infected parts and improve drainage.",
            "Leaf_blight": "Apply Chlorothalonil-based fungicide."
        }.get(predicted_class, "Consult agricultural expert.")

    result = f"""
🌿 Prediction: {predicted_class}  
📊 Confidence: {confidence:.4f}  

🔥 Severity: {severity}  

💊 Recommendation: {recommendation}
"""

    return overlay, result

# -----------------------------
# UI DESIGN (Premium Green)
# -----------------------------
css = """
body {
    background: linear-gradient(135deg, #e8f5e9, #ffffff);
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}
button {
    background: linear-gradient(90deg, #2e7d32, #66bb6a);
    color: white;
    border-radius: 8px;
}
"""

with gr.Blocks(css=css) as app:

    gr.Markdown("""
    # 🌿 XAI-CropCare
    ### Explainable Crop Disease Detection using Grad-CAM
    """)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Leaf Image")

    btn = gr.Button("🔍 Analyze Leaf")

    with gr.Row():
        gradcam_output = gr.Image(label="Grad-CAM Output")
        result_output = gr.Textbox(label="Prediction Details")

    btn.click(predict, inputs=image_input, outputs=[gradcam_output, result_output])

# -----------------------------
# RUN
# -----------------------------
app.launch()