import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

from model import LungDiseaseModel
from phase4_explainability import GradCAM, overlay_cam
import cv2

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["covid", "pneumonia", "normal"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = LungDiseaseModel(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return image, image_tensor, CLASS_NAMES[pred.item()], conf.item()


# -----------------------------
# GRAD-CAM FUNCTION
# -----------------------------
def generate_explainability(image_tensor):
    target_layer = model.backbone.layers[-1]
    cam = GradCAM(model, target_layer)

    heatmap = cam.generate_cam(image_tensor)
    heatmap_pp = cam.generate_cam_pp(image_tensor)

    return heatmap, heatmap_pp


# -----------------------------
# MAIN TEST
# -----------------------------
if __name__ == "__main__":
    img_path = "test_image.jpg"

    image, image_tensor, label, confidence = predict_image(img_path)

    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.4f}")

    # Convert image for OpenCV
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Generate heatmaps
    heatmap, heatmap_pp = generate_explainability(image_tensor)

    overlay1 = overlay_cam(img_np, heatmap)
    overlay2 = overlay_cam(img_np, heatmap_pp)

    # Show results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(overlay1, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay2, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM++")

    plt.show()