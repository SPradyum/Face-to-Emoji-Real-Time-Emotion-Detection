import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Dataset folders
train_dir = r"D:\Documents\Projects\Python\Face to Emoji\archive\train"
val_dir   = r"D:\Documents\Projects\Python\Face to Emoji\archive\test"

# Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1/255)
val_datagen   = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

# Model
emotion_model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")
])

emotion_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
emotion_model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

# Save full model (Keras 3 compatible)
emotion_model.save(r"D:\Documents\Projects\Python\Face to Emoji\model.h5")
print("\n✅ Model saved successfully as model.h5")
