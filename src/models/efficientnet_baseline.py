import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    model = models.efficientnet_b0(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model
