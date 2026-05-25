"""Prediction utilities for handwritten character recognition."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from model.preprocessing import preprocess_image_file, prepare_single_image_for_model


def load_class_names(class_names_path):
    """Load class names JSON generated during training."""
    class_names_path = Path(class_names_path)
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {class_names_path}\n"
            "Train the model first to generate this file."
        )

    with open(class_names_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not class_names:
        raise ValueError("Class names file is empty.")
    return class_names


def load_inference_assets(model_path, class_names_path):
    """Load trained model and class names for prediction."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train the model first or place a trained model in saved_model/."
        )

    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_names_path)
    return model, class_names


def predict_character(
    image_path,
    model=None,
    class_names=None,
    model_path="saved_model/handwritten_cnn.keras",
    class_names_path="saved_model/class_names.json",
    image_size=(28, 28),
):
    """
    Predict a handwritten character/digit from an image file.

    Returns:
        tuple: (predicted_label, confidence_percent, raw_probabilities)
    """
    if model is None or class_names is None:
        model, class_names = load_inference_assets(model_path, class_names_path)

    processed_image = preprocess_image_file(
        image_path=image_path,
        image_size=image_size,
        invert_if_needed=True,
    )
    model_input = prepare_single_image_for_model(processed_image)

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index]) * 100.0

    if predicted_index >= len(class_names):
        raise ValueError("Predicted index is out of class_names range.")

    predicted_label = class_names[predicted_index]
    return predicted_label, confidence, probabilities


def parse_arguments():
    """Command-line arguments for quick prediction testing."""
    parser = argparse.ArgumentParser(
        description="Predict handwritten character/digit from an image."
    )
    parser.add_argument("image_path", type=str, help="Path to image file for prediction.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_model/handwritten_cnn.keras",
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--class_names_path",
        type=str,
        default="saved_model/class_names.json",
        help="Path to class names JSON file.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[28, 28],
        metavar=("WIDTH", "HEIGHT"),
        help="Image size expected by model. Default: 28 28",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    image_size = tuple(args.image_size)

    predicted_label, confidence, probabilities = predict_character(
        image_path=args.image_path,
        model_path=args.model_path,
        class_names_path=args.class_names_path,
        image_size=image_size,
    )

    print("\nPrediction result")
    print("-----------------")
    print(f"Predicted Character: {predicted_label}")
    print(f"Confidence Score: {confidence:.2f}%")
    print(f"Raw Probabilities: {np.round(probabilities, 4)}")


if __name__ == "__main__":
    main()

