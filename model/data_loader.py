"""Dataset loading helpers for MNIST, EMNIST, and custom handwritten datasets."""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from .preprocessing import add_channel_dimension, preprocess_image_array


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_dataset(dataset_type, dataset_dir, image_size=(28, 28), test_split=0.2, random_seed=42):
    """
    Load one of the supported datasets and return train/test arrays.

    Args:
        dataset_type (str): One of: "mnist", "emnist", "custom".
        dataset_dir (str | Path): Root dataset directory.
        image_size (tuple): Image size used by the model.
        test_split (float): Test ratio for custom dataset split.
        random_seed (int): Random seed for reproducible custom split.

    Returns:
        tuple: x_train, y_train, x_test, y_test, class_names
    """
    dataset_type = dataset_type.lower().strip()
    dataset_dir = Path(dataset_dir)

    if dataset_type == "mnist":
        return load_mnist_dataset(image_size=image_size)
    if dataset_type == "emnist":
        return load_emnist_dataset(dataset_dir=dataset_dir, image_size=image_size)
    if dataset_type == "custom":
        return load_custom_dataset(
            dataset_dir=dataset_dir,
            image_size=image_size,
            test_split=test_split,
            random_seed=random_seed,
        )

    raise ValueError("dataset_type must be one of: mnist, emnist, custom")


def load_mnist_dataset(image_size=(28, 28)):
    """Load MNIST directly from TensorFlow/Keras."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = _prepare_batch_images(x_train, image_size=image_size)
    x_test = _prepare_batch_images(x_test, image_size=image_size)
    class_names = [str(i) for i in range(10)]
    return x_train, y_train.astype(np.int32), x_test, y_test.astype(np.int32), class_names


def load_emnist_dataset(dataset_dir, image_size=(28, 28)):
    """
    Load EMNIST from CSV files placed under:
        dataset/emnist/*train*.csv
        dataset/emnist/*test*.csv

    Optional mapping file:
        dataset/emnist/*mapping*.txt
    """
    emnist_dir = dataset_dir / "emnist"
    if not emnist_dir.exists():
        raise FileNotFoundError(
            f"EMNIST folder not found: {emnist_dir}\n"
            "Create it and add train/test CSV files."
        )

    train_csv, test_csv = _find_emnist_csv_files(emnist_dir)
    train_data = _read_numeric_csv(train_csv)
    test_data = _read_numeric_csv(test_csv)

    y_train_raw = train_data[:, 0].astype(np.int32)
    x_train_raw = train_data[:, 1:]
    y_test_raw = test_data[:, 0].astype(np.int32)
    x_test_raw = test_data[:, 1:]

    if x_train_raw.shape[1] != 784 or x_test_raw.shape[1] != 784:
        raise ValueError(
            "EMNIST CSV must contain 785 columns: 1 label + 784 pixel columns."
        )

    # EMNIST CSV images are often rotated/transposed; fix orientation.
    x_train_images = _fix_emnist_orientation(x_train_raw)
    x_test_images = _fix_emnist_orientation(x_test_raw)

    # Resize if needed and normalize.
    x_train = _prepare_batch_images(x_train_images, image_size=image_size)
    x_test = _prepare_batch_images(x_test_images, image_size=image_size)

    # Convert labels to continuous indices: 0,1,2...
    y_train, y_test, class_names = _encode_train_test_labels(
        y_train_raw, y_test_raw, mapping_path=_find_emnist_mapping_file(emnist_dir)
    )
    return x_train, y_train, x_test, y_test, class_names


def load_custom_dataset(dataset_dir, image_size=(28, 28), test_split=0.2, random_seed=42):
    """
    Load custom dataset from:
        dataset/custom/<class_name>/*.png
    """
    custom_dir = dataset_dir / "custom"
    if not custom_dir.exists():
        raise FileNotFoundError(
            f"Custom dataset folder not found: {custom_dir}\n"
            "Create class subfolders and add images."
        )

    class_dirs = sorted([d for d in custom_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found in {custom_dir}. "
            "Add folders such as 0, 1, A, B, etc."
        )

    images = []
    labels = []
    class_names = [d.name for d in class_dirs]

    for class_index, class_dir in enumerate(class_dirs):
        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            # For scanned paper images, auto-invert helps align with MNIST style.
            processed = preprocess_image_array(
                image_array=image,
                image_size=image_size,
                invert_if_needed=True,
            )
            images.append(add_channel_dimension(processed))
            labels.append(class_index)

    if not images:
        raise FileNotFoundError(
            f"No supported image files found under {custom_dir}. "
            f"Use extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    x_data = np.array(images, dtype=np.float32)
    y_data = np.array(labels, dtype=np.int32)

    x_train, y_train, x_test, y_test = _split_data(
        x_data, y_data, test_split=test_split, random_seed=random_seed
    )
    return x_train, y_train, x_test, y_test, class_names


def _prepare_batch_images(batch_images, image_size=(28, 28)):
    """Resize (if needed), normalize, and add channel dimension for a batch of images."""
    if batch_images.ndim != 3:
        raise ValueError(f"Expected image batch shape (N, H, W), got: {batch_images.shape}")

    prepared_images = []
    for image in batch_images:
        processed = preprocess_image_array(
            image_array=image,
            image_size=image_size,
            invert_if_needed=False,
        )
        prepared_images.append(add_channel_dimension(processed))

    return np.array(prepared_images, dtype=np.float32)


def _read_numeric_csv(csv_path):
    """Read a numeric CSV and automatically handle an optional header row."""
    data = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    # If there is a text header row, genfromtxt creates NaNs. Remove NaN rows.
    if np.isnan(data).any():
        data = data[~np.isnan(data).any(axis=1)]

    if data.size == 0:
        raise ValueError(f"CSV file appears empty or invalid: {csv_path}")
    return data


def _find_emnist_csv_files(emnist_dir):
    """Pick one train CSV and one test CSV from EMNIST directory."""
    train_candidates = sorted(emnist_dir.glob("*train*.csv"))
    test_candidates = sorted(emnist_dir.glob("*test*.csv"))

    if not train_candidates or not test_candidates:
        raise FileNotFoundError(
            f"Could not find EMNIST train/test CSV files in {emnist_dir}.\n"
            "Expected names like emnist-balanced-train.csv and emnist-balanced-test.csv."
        )

    return train_candidates[0], test_candidates[0]


def _find_emnist_mapping_file(emnist_dir):
    """Find optional mapping file (label -> ASCII code)."""
    mapping_files = sorted(emnist_dir.glob("*mapping*.txt"))
    return mapping_files[0] if mapping_files else None


def _fix_emnist_orientation(flat_images):
    """
    Convert flattened 784 columns to 28x28 and fix the orientation.
    EMNIST images are often rotated/transposed in CSV representation.
    """
    images = flat_images.reshape(-1, 28, 28)
    images = np.transpose(images, (0, 2, 1))
    images = np.flip(images, axis=2)
    return images


def _parse_emnist_mapping(mapping_path):
    """Parse mapping file where each line can be: index ascii [optional_ascii_lower]."""
    mapping = {}
    if mapping_path is None:
        return mapping

    with open(mapping_path, "r", encoding="utf-8") as file:
        for line in file:
            cleaned = line.strip().replace(",", " ")
            if not cleaned:
                continue
            parts = cleaned.split()
            if len(parts) < 2:
                continue
            try:
                label_id = int(parts[0])
                ascii_code = int(parts[1])
                mapping[label_id] = chr(ascii_code)
            except ValueError:
                continue
    return mapping


def _encode_train_test_labels(y_train_raw, y_test_raw, mapping_path=None):
    """Encode raw labels into 0..N-1 and build human-readable class names."""
    unique_labels = sorted(np.unique(np.concatenate([y_train_raw, y_test_raw])).tolist())
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    mapping = _parse_emnist_mapping(mapping_path)
    class_names = []
    for original_label in unique_labels:
        if original_label in mapping:
            class_names.append(mapping[original_label])
        else:
            class_names.append(str(original_label))

    y_train = np.array([label_to_index[label] for label in y_train_raw], dtype=np.int32)
    y_test = np.array([label_to_index[label] for label in y_test_raw], dtype=np.int32)
    return y_train, y_test, class_names


def _split_data(x_data, y_data, test_split=0.2, random_seed=42):
    """Simple train/test split implemented with NumPy only."""
    if not 0.0 < test_split < 1.0:
        raise ValueError("test_split must be between 0 and 1 (example: 0.2).")

    num_samples = len(x_data)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)

    x_data = x_data[indices]
    y_data = y_data[indices]

    split_index = int(num_samples * (1.0 - test_split))
    x_train = x_data[:split_index]
    y_train = y_data[:split_index]
    x_test = x_data[split_index:]
    y_test = y_data[split_index:]

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Dataset too small for the selected test_split.")

    return x_train, y_train, x_test, y_test

