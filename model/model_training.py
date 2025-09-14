import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# --- Configuration ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'data/dataset/' # Updated path to be relative to the script's location
SAVE_PATH = 'model/saved_model.h5'
VALIDATION_SPLIT = 0.2

# --- 1. Create a tf.data.Dataset ---
# This single function replaces the manual CSV parsing, path joining,
# label encoding, and train/test splitting.
print("Loading data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")

# --- 2. Configure Dataset for Performance ---
# .cache() keeps images in memory after the first epoch.
# .prefetch() overlaps data preprocessing and model execution.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Build the Model ---
# We now use Keras layers for data augmentation, which run on the GPU.
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ],
    name="data_augmentation"
)

# Load the VGG16 base model
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False # Freeze the base model

# Create the final model by chaining the layers using the Functional API
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.vgg16.preprocess_input(x) # VGG16-specific preprocessing
x = base_model(x, training=False)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# --- 4. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 5. Train the Model ---
# The .fit method can directly consume the tf.data.Dataset objects
print(f"Starting training for {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# --- 6. Save the Final Model ---
print(f"Training complete. Saving model to {SAVE_PATH}")
model.save(SAVE_PATH)
print("Model trained and saved successfully!")


model.save('../model/saved_model.h5')
print("Model trained and saved.")
