# =========================
# EVALUATION
# =========================
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, loader, device, class_names):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, preds = torch.max(out, 1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    acc = np.mean(np.array(preds_all) == np.array(labels_all))
    cm = confusion_matrix(labels_all, preds_all)

    print("\nClassification Report:")
    print(classification_report(labels_all, preds_all, target_names=class_names))

    return acc, cm
