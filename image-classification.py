"""
=============================================================
 CNN-Based Multi-Class Image Classification — CIFAR-10
 Author  : ML Engineer
 Dataset : CIFAR-10  (10 classes, 60 000 images 32×32 RGB)
 Framework: TensorFlow / Keras
=============================================================
"""

# ─────────────────────────────────────────────
# 0.  Imports & reproducibility
# ─────────────────────────────────────────────
import os, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
NUM_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────
# 1.  Data Loading
# ─────────────────────────────────────────────
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    print(f"Train : {x_train.shape}  |  Test : {x_test.shape}")
    return x_train, y_train, x_test, y_test


# ─────────────────────────────────────────────
# 2.  Preprocessing — One-Hot Encoding
# ─────────────────────────────────────────────
def preprocess_labels(y_train, y_test):
    y_train_ohe = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_ohe  = keras.utils.to_categorical(y_test,  NUM_CLASSES)
    return y_train_ohe, y_test_ohe


# ─────────────────────────────────────────────
# 3.  CNN Architecture
# ─────────────────────────────────────────────
def build_cnn(input_shape=(32, 32, 3)):
    # FIX: keras.Input as first layer — removes input_shape UserWarning
    model = models.Sequential([
        keras.Input(shape=input_shape),
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        # Head
        layers.Dropout(0.3, name="dropout"),
        layers.Flatten(name="flatten"),
        layers.Dense(NUM_CLASSES, activation="softmax", name="output"),
    ], name="CIFAR10_CNN")
    return model


# ─────────────────────────────────────────────
# 4.  Data Augmentation
# ─────────────────────────────────────────────
def get_data_augmentor():
    return ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
    )


# ─────────────────────────────────────────────
# 5.  Compile & Train
# ─────────────────────────────────────────────
def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train_model(model, x_train, y_train_ohe, x_test, y_test_ohe,
                epochs=20, batch_size=64):
    datagen = get_data_augmentor()
    datagen.fit(x_train)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )

    # FIX: assign generator to variable so it doesn't reset mid-epoch
    train_gen = datagen.flow(x_train, y_train_ohe, batch_size=batch_size)

    history = model.fit(
        train_gen,
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test_ohe),
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# 6.  Visualisation
# ─────────────────────────────────────────────
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — CIFAR-10 CNN", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.plot(history.history["loss"],     label="Train Loss",      color="#2196F3", linewidth=2)
    ax.plot(history.history["val_loss"], label="Validation Loss", color="#F44336", linewidth=2, linestyle="--")
    ax.set_title("Loss (Error Gradient)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history.history["accuracy"],     label="Train Accuracy",      color="#4CAF50", linewidth=2)
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#FF9800", linewidth=2, linestyle="--")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✔  Saved: training_curves.png")


# ─────────────────────────────────────────────
# 7.  Evaluation
# ─────────────────────────────────────────────
def evaluate_model(model, x_test, y_test, y_test_ohe):
    loss, accuracy = model.evaluate(x_test, y_test_ohe, verbose=0)
    print(f"\n{'='*45}")
    print(f"  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*45}\n")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    y_true = y_test.flatten()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor="gray")
    plt.title("Confusion Matrix — CIFAR-10 CNN", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✔  Saved: confusion_matrix.png")

    print("\nClass-Wise Performance:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    return y_pred, y_true


# ─────────────────────────────────────────────
# 8.  Transfer Learning — MobileNetV2
# ─────────────────────────────────────────────
def build_mobilenetv2_classifier(input_size=96):
    inp  = keras.Input(shape=(32, 32, 3), name="image_input")
    x    = layers.Resizing(input_size, input_size, name="upsample")(inp)

    base = MobileNetV2(input_shape=(input_size, input_size, 3),
                       include_top=False, weights="imagenet")
    base.trainable = False
    print(f"MobileNetV2 base: {len(base.layers)} layers (all frozen)")

    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dropout(0.3, name="dropout")(x)
    x       = layers.Dense(128, activation="relu", name="fc")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=outputs, name="MobileNetV2_CIFAR10")
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model, base


def fine_tune_mobilenetv2(model, base, x_train, y_train_ohe,
                           x_test, y_test_ohe, fine_tune_at=100, batch_size=64):
    datagen = get_data_augmentor()
    datagen.fit(x_train)

    early_stop = EarlyStopping(monitor="val_loss", patience=3,
                               restore_best_weights=True, verbose=1)

    # Phase 1 — head only
    print("\n── Phase 1: Training custom head (base frozen) ──")
    train_gen = datagen.flow(x_train, y_train_ohe, batch_size=batch_size)
    history_phase1 = model.fit(
        train_gen,
        steps_per_epoch=len(x_train) // batch_size,
        epochs=5,
        validation_data=(x_test, y_test_ohe),
        callbacks=[early_stop], verbose=1,
    )

    # Phase 2 — fine-tune top layers
    print(f"\n── Phase 2: Fine-tuning from layer {fine_tune_at} onwards ──")
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"  Unfrozen: {sum(l.trainable for l in base.layers)} / {len(base.layers)} base layers")

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    train_gen2 = datagen.flow(x_train, y_train_ohe, batch_size=batch_size)
    history_phase2 = model.fit(
        train_gen2,
        steps_per_epoch=len(x_train) // batch_size,
        epochs=10,
        validation_data=(x_test, y_test_ohe),
        callbacks=[early_stop], verbose=1,
    )
    return history_phase1, history_phase2


# ─────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  CIFAR-10 CNN Classifier — Full Pipeline")
    print("="*55 + "\n")

    x_train, y_train, x_test, y_test = load_data()
    y_train_ohe, y_test_ohe          = preprocess_labels(y_train, y_test)

    # ── Part A: Custom CNN ────────────────────────
    print("\n[Part A] Custom CNN\n")
    cnn_model = build_cnn()
    cnn_model = compile_model(cnn_model)
    history   = train_model(cnn_model, x_train, y_train_ohe,
                             x_test, y_test_ohe, epochs=20, batch_size=64)
    plot_training_curves(history)
    evaluate_model(cnn_model, x_test, y_test, y_test_ohe)
    cnn_model.save("cifar10_cnn.keras")
    print("\n✔  Custom CNN saved to cifar10_cnn.keras")

    # ── Part B: MobileNetV2 ───────────────────────
    print("\n[Part B] MobileNetV2 Transfer Learning\n")
    mobilenet_model, base = build_mobilenetv2_classifier(input_size=96)
    fine_tune_mobilenetv2(mobilenet_model, base,
                          x_train, y_train_ohe, x_test, y_test_ohe,
                          fine_tune_at=100, batch_size=64)

    print("\n[MobileNetV2] Final Evaluation")
    evaluate_model(mobilenet_model, x_test, y_test, y_test_ohe)
    mobilenet_model.save("cifar10_mobilenetv2.keras")
    print("\n✔  MobileNetV2 saved to cifar10_mobilenetv2.keras")
    print("\n✅  Pipeline complete.\n")


if __name__ == "__main__":
    main()