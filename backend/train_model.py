import tensorflow as tf
import json
import os
import numpy as np

# =====================================================
# ENVIRONMENT CHECK
# =====================================================
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# =====================================================
# KERAS ALIASES (for IDE / linter stability)
# =====================================================
MobileNetV2 = tf.keras.applications.MobileNetV2
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Model = tf.keras.Model

# =====================================================
# CONFIGURATION
# =====================================================
DATASET_DIR = "../dataset/Plant_leave"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Stage 1: train classifier head only
EPOCHS_STAGE_1 = 8

# Stage 2: fine-tune top layers of MobileNetV2
EPOCHS_STAGE_2 = 5

MODEL_PATH = "model/plant_disease_model.h5"
LABELS_PATH = "labels.json"

# =====================================================
# DATA GENERATOR (STRONGER AUGMENTATION)
# Helps reduce tomato bias
# =====================================================
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_data.num_classes
print(f"Detected {NUM_CLASSES} classes")

# =====================================================
# 1️⃣ CLASS WEIGHTING (FIXES TOMATO DOMINANCE)
# =====================================================
class_counts = np.bincount(train_data.classes)

class_weights = {
    i: (1.0 / count) * (len(train_data.classes) / NUM_CLASSES)
    for i, count in enumerate(class_counts)
}

print("Applied class weights:")
for k, v in class_weights.items():
    print(f"  Class {k}: {v:.2f}")

# =====================================================
# MOBILENETV2 BASE MODEL
# =====================================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze entire base model (Stage 1)
base_model.trainable = False

# =====================================================
# CUSTOM CLASSIFIER HEAD
# =====================================================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# =====================================================
# STAGE 1 — TRAIN CLASSIFIER HEAD ONLY
# =====================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🚀 Stage 1: Training classifier head")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE_1,
    class_weight=class_weights
)

# =====================================================
# 2️⃣ FINE-TUNING (UNFREEZE TOP MOBILENET LAYERS)
# =====================================================
print("\n🔓 Stage 2: Fine-tuning MobileNetV2")

# Unfreeze top N layers only
FINE_TUNE_AT = int(len(base_model.layers) * 0.7)

for layer in base_model.layers[FINE_TUNE_AT:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # lower LR is CRITICAL
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE_2,
    class_weight=class_weights
)

# =====================================================
# SAVE MODEL
# =====================================================
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)

# =====================================================
# SAVE LABELS (USED FOR TOP-3 INFERENCE)
# =====================================================
labels = {str(v): k for k, v in train_data.class_indices.items()}

with open(LABELS_PATH, "w") as f:
    json.dump(labels, f, indent=2)

print("✅ MobileNetV2 trained with class weighting + fine-tuning")
