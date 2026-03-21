import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from phase1_baseline import get_dataloaders
from model import LungDiseaseModel   
from metrics import evaluate
from phase4_explainability import GradCAM, overlay_cam
import cv2

def train_pipeline(base_path="dataset_split"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_dataset = get_dataloaders(base_path)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Class imbalance handling
    weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_dataset.targets),
        y=train_dataset.targets
    )
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    model = LungDiseaseModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    best_acc = 0
    patience, counter = 3, 0

    print(f"Training on {device}")

    for epoch in range(20):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc, cm = evaluate(model, val_loader, device, class_names)
        scheduler.step(acc)

        print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
        print("Confusion Matrix:\n", cm)

        # Early Stopping
        if acc > best_acc:
            best_acc = acc
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    print("Training Complete!")

    # =========================
    # PHASE 4: GRAD-CAM VISUALIZATION
    # =========================
    print("\nGenerating Grad-CAM visualization...")
    
    # Load BEST model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    # Sample image
    sample_img, label = next(iter(val_loader))
    sample_img = sample_img[0].unsqueeze(0).to(device)
    
    # Denormalize image
    img_np = sample_img[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std + mean)
    img_np = (img_np * 255).clip(0, 255).astype("uint8")
    
    # Target layer
    target_layer = model.backbone.layers[-1]
    
    # Grad-CAM
    cam = GradCAM(model, target_layer)
    
    heatmap = cam.generate_cam(sample_img)
    heatmap_pp = cam.generate_cam_pp(sample_img)
    
    overlay1 = overlay_cam(img_np, heatmap)
    overlay2 = overlay_cam(img_np, heatmap_pp)

    cv2.imwrite("gradcam.jpg", overlay1)
    cv2.imwrite("gradcam_pp.jpg", overlay2)

    print("Grad-CAM saved as gradcam.jpg")
    print("Grad-CAM++ saved as gradcam_pp.jpg")


# RUN
if __name__ == "__main__":
    train_pipeline()