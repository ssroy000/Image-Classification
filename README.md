# 🧠 Lung Disease Classification using Deep Learning (COVID-19 / Pneumonia / Normal)

---

## 📌 Project Overview

This project is a **modular deep learning pipeline** for classifying **chest X-ray images** into:

* COVID-19
* Pneumonia
* Normal

It focuses on building a **lightweight, efficient, and explainable AI system** suitable for medical imaging applications.

---

## 🎯 Key Highlights

* Lightweight **MobileNet-inspired architecture**
* **CBAM attention mechanism** for better focus
* **Explainable AI (Grad-CAM & Grad-CAM++)**
* Handles **class imbalance**
* Uses **medical-grade evaluation metrics**

---

## 🧩 Project Structure

```
project/
│
├── phase1_baseline.py            # Data loading & preprocessing
├── phase2_efficient.py        # Feature extraction (MobileNet-like)
├── phase3_attention.py       # CBAM attention module
├── phase4_explainability.py  # Grad-CAM & Grad-CAM++
├── model.py                  # Final model (Phase 2 + Phase 3)
├── metrics.py                # Evaluation metrics
├── train.py                  # Training pipeline
├── inference.py              # Test new images
└── dataset_split/
    ├── train/
    └── test/
```

---

## 🔬 Project Phases

---

### 🔹 Phase 1: Data Pipeline (`phase1_baseline.py`)

* Loads dataset using folder structure
* Applies preprocessing:

  * Resize (224×224)
  * Normalization
  * Data augmentation (flip, rotation)
* Converts images into tensors

👉 Output:

* `train_loader`, `val_loader`
* Class labels

---

### 🔹 Phase 2: Feature Extraction (`phase2_efficient.py`)

* Uses **Depthwise Separable Convolutions**
* Inspired by MobileNet architecture

👉 Learns:

* Edges
* Textures
* Lung patterns

---

### 🔹 Phase 3: Attention (`phase3_attention.py`)

* Implements **CBAM (Convolutional Block Attention Module)**

👉 Enhances:

* Channel importance (**what to focus on**)
* Spatial importance (**where to focus**)

---

### 🔹 Phase 4: Explainability (`phase4_explainability.py`)

* Implements:

  * **Grad-CAM**
  * **Grad-CAM++**

👉 Purpose:

* Visualize **important regions in X-ray images**
* Convert model from **black-box → explainable system**

👉 Output:

* Heatmaps highlighting infection regions

---

### 🔹 Model (`model.py`)

* Combines:

  * Backbone (Phase 2)
  * Attention (Phase 3)

👉 Architecture:

```
Image → Backbone → Attention → Pooling → FC → Output
```

---

### 🔹 Metrics (`metrics.py`)

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

👉 Critical for medical evaluation

---

### 🔹 Training Pipeline (`train.py`)

Handles:

* Model training
* Loss calculation
* Backpropagation
* Validation

Includes:

* Class imbalance handling
* Learning rate scheduler
* Early stopping

👉 Saves:

```
best_model.pth
```

---

### 🔹 Inference (`inference.py`)

* Loads trained model
* Predicts new X-ray images

👉 Output:

* Predicted class
* Confidence score

---

## 📥 Input Requirements

### Dataset Format

```
dataset_split/
│
├── train/
│   ├── covid/
│   ├── pneumonia/
│   └── normal/
│
├── test/
│   ├── covid/
│   ├── pneumonia/
│   └── normal/
```

👉 Each folder contains corresponding X-ray images.

---

## 📤 Output

### 1. Model File

```
best_model.pth
```

### 2. Console Output

* Training loss
* Validation accuracy
* Confusion matrix
* Classification report

### 3. Explainability Output

```
gradcam.jpg
gradcam_pp.jpg
```

👉 Heatmaps showing model focus regions

---

## ⚙️ Installation

Install dependencies:

```bash
pip install torch torchvision scikit-learn numpy opencv-python pillow
```

---

## ▶️ How to Run

### Step 1: Navigate to project folder

```bash
cd Image-Classification
```

---

### Step 2: Train the model

```bash
python train.py
```

---

### Step 3: Test on new image

```bash
python inference.py
```

---

## 🧠 Model Workflow

```
Input Image
   ↓
Preprocessing
   ↓
Feature Extraction
   ↓
Attention Mechanism
   ↓
Classification
   ↓
Prediction
   ↓
Grad-CAM Visualization
```

---

## 🔍 Explainability (Important Feature)

* **Grad-CAM** → highlights important regions
* **Grad-CAM++** → better localization for multiple infection areas

👉 Helps doctors understand:

* Why model predicted a disease
* Which lung region is affected

---

## 🚀 Key Features

* Modular architecture (easy to extend)
* Lightweight model (fast inference)
* Attention-based learning
* Explainable AI integration
* Medical-grade metrics
* Training optimization (scheduler + early stopping)

---

## 📊 Use Cases

* COVID-19 detection
* Pneumonia screening
* AI-assisted radiology
* Medical research support

---

## 🔮 Future Improvements

* Pretrained models (EfficientNet)
* ROC-AUC analysis
* Web deployment (Flask / Streamlit)
* Real-time prediction system

---

## 🏁 Conclusion

This project delivers a **complete end-to-end deep learning pipeline** with:

* Efficient architecture
* Attention mechanism
* Explainability (Grad-CAM++)
* Robust evaluation

👉 Making it suitable for **academic research, healthcare AI, and real-world deployment**

---
