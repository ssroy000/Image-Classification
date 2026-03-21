import torch
import torch.nn.functional as F
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    # -----------------------------
    # HOOKS (capture activations + gradients)
    # -----------------------------
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]      # (C,H,W)
        activations = self.activations[0]  # (C,H,W)

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    # -----------------------------
    # GRAD-CAM++
    # -----------------------------
    def generate_cam_pp(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3

        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)

        alpha = grads_power_2 / (
            2 * grads_power_2 +
            sum_activations * grads_power_3 + 1e-8
        )

        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()


# -----------------------------
# UTILITY: Overlay heatmap
# -----------------------------
def overlay_cam(image, cam):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + image * 0.6
    return np.uint8(overlay)