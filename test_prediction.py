"""
Test CIFAR-10 models on a random test image or your own image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow import keras

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─────────────────────────────────────────────
# Load saved models
# ─────────────────────────────────────────────
print("Loading models...")
cnn_model       = keras.models.load_model("cifar10_cnn.keras")
mobilenet_model = keras.models.load_model("cifar10_mobilenetv2.keras")
print("✔  Models loaded!")

# ─────────────────────────────────────────────
# Load CIFAR-10 test set
# ─────────────────────────────────────────────
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

# ─────────────────────────────────────────────
# Pick a random image
# ─────────────────────────────────────────────
idx        = np.random.randint(0, len(x_test))
image      = x_test[idx]
true_label = CLASS_NAMES[y_test[idx][0]]

# ─────────────────────────────────────────────
# Predict with both models
# ─────────────────────────────────────────────
img_input = np.expand_dims(image, axis=0)  # shape: (1, 32, 32, 3)

cnn_probs       = cnn_model.predict(img_input, verbose=0)[0]
mobilenet_probs = mobilenet_model.predict(img_input, verbose=0)[0]

cnn_pred       = CLASS_NAMES[np.argmax(cnn_probs)]
mobilenet_pred = CLASS_NAMES[np.argmax(mobilenet_probs)]

# ─────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"  True Label      : {true_label.upper()}")
print(f"  CNN Prediction  : {cnn_pred.upper()}  ({cnn_probs.max()*100:.1f}% confident)")
print(f"  MobileNet Pred  : {mobilenet_pred.upper()}  ({mobilenet_probs.max()*100:.1f}% confident)")
print(f"{'='*45}\n")

# ─────────────────────────────────────────────
# Visualise
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"True Label: {true_label.upper()}", fontsize=14, fontweight="bold")

# Original image
axes[0].imshow(image)
axes[0].set_title("Input Image (32×32)", fontsize=11)
axes[0].axis("off")

# CNN confidence bar chart
colors_cnn = ["#F44336" if c == cnn_pred else "#90CAF9" for c in CLASS_NAMES]
axes[1].barh(CLASS_NAMES, cnn_probs * 100, color=colors_cnn)
axes[1].set_title(f"Custom CNN → {cnn_pred.upper()}", fontsize=11)
axes[1].set_xlabel("Confidence (%)")
axes[1].set_xlim(0, 100)
axes[1].invert_yaxis()
axes[1].axvline(x=50, color="gray", linestyle="--", alpha=0.5)

# MobileNetV2 confidence bar chart
colors_mob = ["#4CAF50" if c == mobilenet_pred else "#A5D6A7" for c in CLASS_NAMES]
axes[2].barh(CLASS_NAMES, mobilenet_probs * 100, color=colors_mob)
axes[2].set_title(f"MobileNetV2 → {mobilenet_pred.upper()}", fontsize=11)
axes[2].set_xlabel("Confidence (%)")
axes[2].set_xlim(0, 100)
axes[2].invert_yaxis()
axes[2].axvline(x=50, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight")
plt.show()
print("✔  Saved: prediction_result.png")