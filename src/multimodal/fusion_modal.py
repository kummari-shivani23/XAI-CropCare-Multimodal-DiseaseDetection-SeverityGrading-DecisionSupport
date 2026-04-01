# import torch
# import torch.nn as nn
# from torchvision import models

# class MultimodalFusionModel(nn.Module):
#     def __init__(self, num_classes=4):
#         super().__init__()

#         # Image encoder
#         self.cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")

#         # 2️⃣ 🔒 FREEZE CNN BACKBONE (ADD THIS BLOCK)
#         for param in self.cnn.parameters():
#             param.requires_grad = False

        
#         self.cnn.classifier = nn.Identity()  # remove final layer

#         image_dim = 1280

#         # Weather encoder (MLP)
#         self.weather_mlp = nn.Sequential(
#             nn.Linear(4, 32),
#             nn.ReLU(),
#             nn.Linear(32, 32)
#         )

#         # Fusion head
#         self.classifier = nn.Sequential(
#             nn.Linear(image_dim + 32, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, image, weather):
#         img_feat = self.cnn(image)
#         weather_feat = self.weather_mlp(weather)

#         fused = torch.cat([img_feat, weather_feat], dim=1)
#         return self.classifier(fused)




import torch
import torch.nn as nn
from torchvision import models

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # CNN
        self.cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # ✅ PARTIAL FREEZE (IMPORTANT FOR GRADCAM)
        for param in self.cnn.features[:-2].parameters():
            param.requires_grad = False

        for param in self.cnn.features[-2:].parameters():
            param.requires_grad = True

        self.cnn.classifier = nn.Identity()
        image_dim = 1280

        # Weather MLP
        self.weather_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, weather):
        img_feat = self.cnn(image)
        weather_feat = self.weather_mlp(weather)
        fused = torch.cat([img_feat, weather_feat], dim=1)
        return self.classifier(fused)
