"""CNN model definition for handwritten character recognition."""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape, num_classes, learning_rate=0.001):
    """
    Build and compile a beginner-friendly CNN model.

    Args:
        input_shape (tuple): Shape of one image, e.g. (28, 28, 1).
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for Adam optimizer.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = models.Sequential(
        [
            # Feature extraction block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            # Feature extraction block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Classification head
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="handwritten_character_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

