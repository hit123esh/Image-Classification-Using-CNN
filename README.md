# CIFAR-10 Image Classification — CNN + MobileNetV2

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A deep learning project that classifies images into **10 categories** using the CIFAR-10 dataset. Implements both a **custom CNN from scratch** and **MobileNetV2 transfer learning**, achieving up to **83.8% accuracy**.

---

## 📊 Results

| Model | Test Accuracy | Test Loss | Parameters |
|---|---|---|---|
| Custom CNN (2-block) | 66.84% | 0.9552 | 60,746 |
| **MobileNetV2 (fine-tuned)** | **83.80%** | **0.4823** | **2,423,242** |

### Class-Wise Performance (MobileNetV2)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| ✈️ Airplane | 83.8% | 88.6% | 86.1% |
| 🚗 Automobile | 90.6% | 94.0% | 92.3% |
| 🐦 Bird | 85.3% | 77.4% | 81.2% |
| 🐱 Cat | 77.1% | 62.3% | 68.9% |
| 🦌 Deer | 83.4% | 78.0% | 80.6% |
| 🐶 Dog | 78.5% | 76.4% | 77.5% |
| 🐸 Frog | 73.6% | 94.1% | 82.6% |
| 🐴 Horse | 85.3% | 86.8% | 86.0% |
| 🚢 Ship | 92.3% | 88.6% | 90.4% |
| 🚛 Truck | 89.5% | 91.8% | 90.6% |

---

## 🗂️ Project Structure

```
Image-Classification/
│
├── image-classification.py    # Main training pipeline
├── test_prediction.py         # Test on random images
├── requirements.txt           # Dependencies
│
├── cifar10_cnn.keras          # Saved custom CNN model
├── cifar10_mobilenetv2.keras  # Saved MobileNetV2 model
│
├── training_curves.png        # Loss & accuracy plots
├── confusion_matrix.png       # Confusion matrix heatmap
└── prediction_result.png      # Sample prediction output
```

---

## 🏗️ Model Architecture

### Part A — Custom CNN

```
Input (32×32×3)
    │
    ▼
Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
    │
    ▼
Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
    │
    ▼
Dropout(0.3) → Flatten → Dense(10, Softmax)
```

### Part B — MobileNetV2 Transfer Learning

```
Input (32×32×3)
    │
    ▼
Resizing (96×96×3)
    │
    ▼
MobileNetV2 (pretrained ImageNet, 154 layers)
    │
    ▼
GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → Dense(10, Softmax)
```

**Two-phase training:**
- **Phase 1** (5 epochs) — Base frozen, only custom head trains (`lr=1e-4`)
- **Phase 2** (10 epochs) — Top 54 layers unfrozen, fine-tuned (`lr=1e-5`)

---

## ⚙️ Training Configuration

| Parameter | Custom CNN | MobileNetV2 |
|---|---|---|
| Optimizer | Adam | Adam |
| Loss | Categorical Crossentropy | Categorical Crossentropy |
| Batch Size | 64 | 64 |
| Max Epochs | 20 | 5 + 10 |
| Early Stopping | patience=4 | patience=3 |
| Augmentation | Width/Height shift ±10% | Width/Height shift ±10% |

---

## 📦 Requirements

```
tensorflow-macos==2.16.2   # Apple Silicon
tensorflow-metal==1.1.0    # Apple Silicon GPU
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
```

---

## 📈 Key Features

- ✅ **Data Augmentation** — width & height shift to improve generalization
- ✅ **Batch Normalization** — stabilizes and speeds up training
- ✅ **Dropout Regularization** — prevents overfitting
- ✅ **Early Stopping** — automatically stops at best validation loss
- ✅ **Transfer Learning** — leverages ImageNet pretrained weights
- ✅ **Two-phase Fine-tuning** — gradual unfreezing for stable training
- ✅ **Confusion Matrix** — full class-wise evaluation
- ✅ **Reproducible** — fixed seeds for Python, NumPy, TensorFlow

---

## 🔍 Sample Prediction Output

Running `test_prediction.py` produces a 3-panel visualization:

```
=============================================
  True Label      : HORSE
  CNN Prediction  : HORSE  (71.3% confident)
  MobileNet Pred  : HORSE  (94.2% confident)
=============================================
```

---

## 📚 Dataset

**CIFAR-10** — 60,000 color images (32×32 px) across 10 classes
- 50,000 training images
- 10,000 test images
- Auto-downloaded via `keras.datasets.cifar10`

---

## 🙋 Author

**Hitesh Umesh**  
Feel free to ⭐ this repo if you found it useful!
