# import torch
# import cv2
# import numpy as np

# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self._register_hooks()

#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output

#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0]

#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)

#     def generate(self, image, weather, class_idx=None):
#         self.model.zero_grad()
#         output = self.model(image, weather)

#         if class_idx is None:
#             class_idx = output.argmax(dim=1).item()

#         output[:, class_idx].backward(retain_graph=True)

#         assert self.gradients is not None, "❌ Gradients not captured"

#         weights = self.gradients.mean(dim=(2, 3), keepdim=True)
#         cam = (weights * self.activations).sum(dim=1)

#         cam = torch.relu(cam)
#         cam = cam.squeeze().detach().cpu().numpy()

#         cam = cv2.resize(cam, (224, 224))
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

#         return cam


import torch
import cv2
import numpy as np

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

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)

        # ✅ FIX for PyTorch 2+
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image, weather, class_idx=None):
        self.model.zero_grad()

        output = self.model(image, weather)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        # ✅ SAFE CHECK
        if self.gradients is None:
            print("⚠ Gradients not captured")
            return np.zeros((224, 224))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam