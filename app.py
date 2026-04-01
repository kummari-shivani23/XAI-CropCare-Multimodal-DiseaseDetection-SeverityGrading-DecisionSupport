import torch
import torch.nn.functional as F
from torchvision import models, transforms
import gradio as gr
from PIL import Image
import numpy as np
import cv2

# -----------------------------
# CONFIG
# -----------------------------
BASELINE_MODEL_PATH = "models/best_baseline_model.pth"
MULTIMODAL_MODEL_PATH = "models/multimodal_model.pth"

class_names = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# LOAD BASELINE MODEL
# -----------------------------
baseline_model = models.efficientnet_b0(weights=None)
num_features = baseline_model.classifier[1].in_features
baseline_model.classifier[1] = torch.nn.Linear(num_features, len(class_names))
baseline_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))
baseline_model.to(device)
baseline_model.eval()

# -----------------------------
# MULTIMODAL MODEL
# -----------------------------
class MultimodalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.efficientnet_b0(weights=None)
        num_features = self.cnn.classifier[1].in_features
        self.cnn.classifier[1] = torch.nn.Identity()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16)
        )

        self.fc = torch.nn.Linear(num_features + 16, len(class_names))

    def forward(self, image, weather):
        img_feat = self.cnn(image)
        weather_feat = self.mlp(weather)
        combined = torch.cat((img_feat, weather_feat), dim=1)
        return self.fc(combined)

multimodal_model = MultimodalModel()
multimodal_model.load_state_dict(torch.load(MULTIMODAL_MODEL_PATH, map_location=device))
multimodal_model.to(device)
multimodal_model.eval()

# -----------------------------
# GRAD-CAM
# -----------------------------
def generate_gradcam(model, image_tensor):
    image_tensor.requires_grad = True
    outputs = model(image_tensor)
    pred_class = outputs.argmax()

    outputs[0, pred_class].backward()

    gradients = image_tensor.grad[0].cpu().numpy()
    heatmap = np.mean(gradients, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# -----------------------------
# LOGIC FUNCTIONS
# -----------------------------
def get_severity(conf):
    if conf > 0.85:
        return "🔴 Severe"
    elif conf > 0.65:
        return "🟠 Moderate"
    else:
        return "🟢 Mild"

def get_recommendation(disease):
    return {
        "Black_rot": "Apply Mancozeb. Remove infected leaves.",
        "Esca": "Prune affected vines. Use fungicide spray.",
        "Leaf_blight": "Use Copper fungicide. Improve airflow.",
        "Healthy": "No treatment needed."
    }.get(disease, "Consult expert.")

# -----------------------------
# BASELINE
# -----------------------------
def predict_baseline(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = baseline_model(img)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    disease = class_names[pred.item()]
    confidence = f"{conf.item():.2f}"

    heatmap = generate_gradcam(baseline_model, img.clone())

    severity = get_severity(conf.item())
    recommendation = get_recommendation(disease)

    result = f"""
    ### 🧠 Prediction: **{disease}**
    - Confidence: **{confidence}**
    - Severity: **{severity}**

    ### 💊 Recommendation:
    {recommendation}
    """

    return result, heatmap

# -----------------------------
# MULTIMODAL
# -----------------------------
def predict_multimodal(image, temp, humidity, rainfall):
    img = transform(image).unsqueeze(0).to(device)
    weather = torch.tensor([[temp, humidity, rainfall]], dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = multimodal_model(img, weather)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    disease = class_names[pred.item()]
    confidence = f"{conf.item():.2f}"

    heatmap = generate_gradcam(multimodal_model.cnn, img.clone())

    severity = get_severity(conf.item())
    recommendation = get_recommendation(disease)

    result = f"""
    ### 🌍 Prediction: **{disease}**
    - Confidence: **{confidence}**
    - Severity: **{severity}**

    ### 🌱 Environmental Inputs:
    - Temperature: {temp}
    - Humidity: {humidity}
    - Rainfall: {rainfall}

    ### 💊 Recommendation:
    {recommendation}
    """

    return result, heatmap

# -----------------------------
# UI DESIGN
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🌿 XAI-CropCare
    ### Explainable Multimodal Crop Disease Detection System
    """)

    with gr.Tabs():

        # -------- BASELINE --------
        with gr.Tab("📷 Baseline Model"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Leaf Image")

            btn1 = gr.Button("🔍 Analyze", variant="primary")

            with gr.Row():
                output_text = gr.Markdown()
                output_img = gr.Image(label="Grad-CAM Explanation")

            btn1.click(predict_baseline,
                       inputs=img_input,
                       outputs=[output_text, output_img])

        # -------- MULTIMODAL --------
        with gr.Tab("🌍 Multimodal Model"):
            with gr.Row():
                img_input2 = gr.Image(type="pil", label="Upload Leaf Image")

            with gr.Row():
                temp = gr.Slider(0, 50, label="Temperature (°C)")
                humidity = gr.Slider(0, 100, label="Humidity (%)")
                rainfall = gr.Slider(0, 200, label="Rainfall (mm)")

            btn2 = gr.Button("🚀 Analyze with Context", variant="primary")

            with gr.Row():
                output_text2 = gr.Markdown()
                output_img2 = gr.Image(label="Grad-CAM Explanation")

            btn2.click(predict_multimodal,
                       inputs=[img_input2, temp, humidity, rainfall],
                       outputs=[output_text2, output_img2])

demo.launch()