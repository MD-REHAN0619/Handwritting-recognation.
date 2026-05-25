"""Train CNN model for handwritten character recognition."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from model.cnn_model import build_cnn_model
from model.data_loader import load_dataset


def plot_training_history(history, output_path):
    """Save accuracy and loss curves as an image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_class_names(class_names, output_path):
    """Save class names so prediction scripts can map index -> character."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2, ensure_ascii=False)


def save_training_summary(summary_dict, output_path):
    """Save final training summary (loss/accuracy) as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary_dict, file, indent=2)


def parse_arguments():
    """Command-line arguments for flexible training."""
    parser = argparse.ArgumentParser(
        description="Train a CNN for handwritten character recognition."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="mnist",
        choices=["mnist", "emnist", "custom"],
        help="Choose dataset source: mnist | emnist | custom",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to dataset root folder (relative or absolute).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[28, 28],
        metavar=("WIDTH", "HEIGHT"),
        help="Input image size. Default: 28 28",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Test split ratio (used for custom dataset only).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_model/handwritten_cnn.keras",
        help="Path to save trained model (.keras recommended).",
    )
    parser.add_argument(
        "--class_names_path",
        type=str,
        default="saved_model/class_names.json",
        help="Path to save class names JSON.",
    )
    parser.add_argument(
        "--history_plot_path",
        type=str,
        default="static/training_curves.png",
        help="Path to save training curves image.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="saved_model/training_summary.json",
        help="Path to save final metrics summary JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_root = Path(__file__).resolve().parent

    dataset_dir = (project_root / args.dataset_dir).resolve() if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)
    model_path = (project_root / args.model_path).resolve() if not Path(args.model_path).is_absolute() else Path(args.model_path)
    class_names_path = (project_root / args.class_names_path).resolve() if not Path(args.class_names_path).is_absolute() else Path(args.class_names_path)
    history_plot_path = (project_root / args.history_plot_path).resolve() if not Path(args.history_plot_path).is_absolute() else Path(args.history_plot_path)
    summary_path = (project_root / args.summary_path).resolve() if not Path(args.summary_path).is_absolute() else Path(args.summary_path)

    image_size = tuple(args.image_size)

    print("\nLoading dataset...")
    x_train, y_train, x_test, y_test, class_names = load_dataset(
        dataset_type=args.dataset_type,
        dataset_dir=dataset_dir,
        image_size=image_size,
        test_split=args.test_split,
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples:  {len(x_test)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}\n")

    model = build_cnn_model(
        input_shape=(image_size[1], image_size[0], 1),
        num_classes=len(class_names),
        learning_rate=args.learning_rate,
    )
    model.summary()

    print("\nStarting training...")
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nSaving model and metadata...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    save_class_names(class_names, class_names_path)
    plot_training_history(history, history_plot_path)
    save_training_summary(
        {
            "dataset_type": args.dataset_type,
            "image_size": list(image_size),
            "num_classes": len(class_names),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "model_path": str(model_path),
            "class_names_path": str(class_names_path),
            "history_plot_path": str(history_plot_path),
        },
        summary_path,
    )

    print("\nTraining complete.")
    print(f"Model saved to: {model_path}")
    print(f"Class names saved to: {class_names_path}")
    print(f"Training curves saved to: {history_plot_path}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

