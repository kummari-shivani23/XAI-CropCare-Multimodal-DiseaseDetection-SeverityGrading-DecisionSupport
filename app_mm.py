# import gradio as gr
# import torch
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms

# from src.multimodal.fusion_modal import MultimodalFusionModel
# from src.xai.gradcam_multimodal import GradCAM   # your gradcam file

# # -----------------------------
# # CONFIG
# # -----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class_names = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = MultimodalFusionModel(num_classes=4).to(device)
# model.load_state_dict(torch.load("models/multimodal_model.pth", map_location=device))
# model.eval()

# target_layer = model.cnn.features[-3]
# gradcam = GradCAM(model, target_layer)

# # -----------------------------
# # TRANSFORM
# # -----------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # -----------------------------
# # SEVERITY LOGIC
# # -----------------------------
# def get_severity(conf):
#     if conf > 0.9:
#         return "🔴 Severe"
#     elif conf > 0.75:
#         return "🟠 Moderate"
#     else:
#         return "🟢 Mild"

# # -----------------------------
# # RECOMMENDATIONS
# # -----------------------------
# def get_recommendation(disease):
#     rec = {
#         "Black_rot": "Apply fungicides like Mancozeb. Remove infected leaves.",
#         "Esca": "Prune infected vines and apply protective fungicide.",
#         "Leaf_blight": "Use copper-based sprays and improve air circulation.",
#         "Healthy": "No disease detected. Maintain regular care."
#     }
#     return rec[disease]

# # -----------------------------
# # MAIN FUNCTION
# # -----------------------------
# def analyze(image, temp, humidity, rainfall, wind):

#     image = Image.fromarray(image).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0).to(device)

#     weather = torch.tensor([[temp, humidity, rainfall, wind]],
#                            dtype=torch.float32).to(device)
#     weather.requires_grad = True

#     # Prediction
#     output = model(input_tensor, weather)
#     probs = torch.softmax(output, dim=1)
#     conf, pred = torch.max(probs, dim=1)

#     disease = class_names[pred.item()]
#     confidence = conf.item()

#     # -----------------------------
#     # Grad-CAM
#     # -----------------------------
#     cam = gradcam.generate(input_tensor, weather, pred.item())

#     cam = cv2.resize(cam, (224, 224))
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

#     img_np = np.array(image.resize((224, 224)))
#     overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

#     # -----------------------------
#     # Weather Importance
#     # -----------------------------
#     model.zero_grad()
#     score = output[0, pred]
#     score.backward()

#     importance = weather.grad.abs().detach().cpu().numpy()[0]
#     importance = importance / importance.sum()

#     features = ["Temp", "Humidity", "Rainfall", "Wind"]

#     plt.figure(figsize=(5,3))
#     plt.bar(features, importance)
#     plt.title("Weather Importance")
#     plt.tight_layout()

#     chart_path = "temp_chart.png"
#     plt.savefig(chart_path)
#     plt.close()

#     severity = get_severity(confidence)
#     recommendation = get_recommendation(disease)

#     result_text = f"""
# 🌿 Prediction: {disease}  
# 📊 Confidence: {confidence:.4f}  

# ⚠ Severity: {severity}  

# 💊 Recommendation: {recommendation}
# """

#     return overlay, chart_path, result_text

# # -----------------------------
# # UI DESIGN
# # -----------------------------
# with gr.Blocks(
#     theme=gr.themes.Soft(),
#     css="""
#     body {background: linear-gradient(to right, #e8f5e9, #ffffff);}
#     h1 {text-align:center; color:#2e7d32;}
#     """
# ) as demo:

#     gr.Markdown("# 🌿 XAI-CropCare (Multimodal)")
#     gr.Markdown("### Explainable Crop Disease Detection with Weather Intelligence")

#     with gr.Row():
#         image_input = gr.Image(type="numpy", label="Upload Leaf Image")

#         with gr.Column():
#             temp = gr.Slider(0, 1, value=0.6, label="Temperature")
#             humidity = gr.Slider(0, 1, value=0.7, label="Humidity")
#             rainfall = gr.Slider(0, 1, value=0.4, label="Rainfall")
#             wind = gr.Slider(0, 1, value=0.3, label="Wind Speed")

#             btn = gr.Button("🔍 Analyze", variant="primary")

#     with gr.Row():
#         output_img = gr.Image(label="Grad-CAM Output")
#         chart = gr.Image(label="Weather Importance")

#     output_text = gr.Markdown()

#     btn.click(
#         analyze,
#         inputs=[image_input, temp, humidity, rainfall, wind],
#         outputs=[output_img, chart, output_text]
#     )

# # -----------------------------
# # RUN
# # -----------------------------
# if __name__ == "__main__":
#     demo.launch()


import gradio as gr
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.multimodal.fusion_modal import MultimodalFusionModel
from src.xai.gradcam_multimodal import GradCAM

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Black_rot", "Esca", "Healthy", "Leaf_blight"]

# -----------------------------
# LOAD MODEL
# -----------------------------
model = MultimodalFusionModel(num_classes=4).to(device)
model.load_state_dict(torch.load("models/multimodal_model.pth", map_location=device))
model.eval()

# ✅ BEST LAYER FOR GRADCAM
target_layer = model.cnn.features[-1]

gradcam = GradCAM(model, target_layer)

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# HELPERS
# -----------------------------
def get_severity(disease,conf):
    if disease == "Healthy":
        return "🟢 None"

    if conf > 0.9:
        return "🔴 Severe"
    elif conf > 0.75:
        return "🟠 Moderate"
    else:
        return "🟢 Mild"

def get_recommendation(disease):
    return {
        "Black_rot": "Apply fungicide like Mancozeb.",
        "Esca": "Prune infected vines.",
        "Leaf_blight": "Use copper-based sprays.",
        "Healthy": "No disease detected."
    }[disease]

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def analyze(image, temp, humidity, rainfall, wind):

    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    weather = torch.tensor([[temp, humidity, rainfall, wind]],
                           dtype=torch.float32,
                           requires_grad=True).to(device)

    # Prediction
    output = model(img_tensor, weather)
    probs = torch.softmax(output, dim=1)
    conf, pred = torch.max(probs, dim=1)

    disease = class_names[pred.item()]
    confidence = conf.item()

    # GradCAM
    cam = gradcam.generate(img_tensor, weather, pred.item())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Weather importance
    model.zero_grad()
    output[0, pred].backward()

    importance = weather.grad.abs().cpu().numpy()[0]
    importance = importance / importance.sum()

    features = ["Temp", "Humidity", "Rainfall", "Wind"]

    plt.figure(figsize=(5,3))
    plt.bar(features, importance)
    plt.title("Weather Importance")
    plt.tight_layout()

    chart_path = "temp_chart.png"
    plt.savefig(chart_path)
    plt.close()

    severity = get_severity(disease,confidence)
    recommendation = get_recommendation(disease)

    result = f"""
🌿 Prediction: {disease}  
📊 Confidence: {confidence:.4f}  

⚠ Severity: {severity}  

💊 Recommendation: {recommendation}
"""

    return overlay, chart_path, result

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="body {background: linear-gradient(to right, #e8f5e9, #ffffff);}"
) as demo:

    gr.Markdown("# 🌿 XAI-CropCare Multimodal System")

    with gr.Row():
        image_input = gr.Image(type="numpy")

        with gr.Column():
            temp = gr.Slider(0,1,0.6,label="Temperature")
            humidity = gr.Slider(0,1,0.7,label="Humidity")
            rainfall = gr.Slider(0,1,0.4,label="Rainfall")
            wind = gr.Slider(0,1,0.3,label="Wind Speed")

            btn = gr.Button("Analyze")

    with gr.Row():
        out_img = gr.Image(label="GradCAM")
        chart = gr.Image(label="Weather Importance")

    out_text = gr.Markdown()

    btn.click(analyze,
              inputs=[image_input, temp, humidity, rainfall, wind],
              outputs=[out_img, chart, out_text])

# -----------------------------
# RUN
# -----------------------------
demo.launch()