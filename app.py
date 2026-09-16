"""Flask web app for handwritten character recognition."""

import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from ocr_service import OCRProcessingError, recognize_document_text
from predict import load_inference_assets, predict_character


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "saved_model" / "handwritten_cnn.keras"
CLASS_NAMES_PATH = BASE_DIR / "saved_model" / "class_names.json"
TRAINING_CURVE_PATH = BASE_DIR / "static" / "training_curves.png"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
RECOGNITION_MODES = {"character", "ocr"}

# Global cache (loaded once, reused for faster predictions)
MODEL = None
CLASS_NAMES = None


def create_app():
    """Application factory pattern."""
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET"])
    def index():
        model_ready = try_load_model()
        return _render_page(
            model_ready=model_ready,
            recognition_mode="character" if model_ready else "ocr",
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        model_ready = try_load_model()
        recognition_mode = request.form.get("recognition_mode", "character").lower().strip()
        if recognition_mode not in RECOGNITION_MODES:
            recognition_mode = "character"

        uploaded_file = request.files.get("image_file")
        if uploaded_file is None or uploaded_file.filename == "":
            return _render_page(
                model_ready=True,
                error_message="Please choose an image file before clicking Predict.",
                recognition_mode=recognition_mode,
            )

        if not is_allowed_file(uploaded_file.filename):
            return _render_page(
                model_ready=True,
                error_message=(
                    "Unsupported file type. Please upload PNG, JPG, JPEG, BMP, TIF, TIFF, or PDF."
                ),
                recognition_mode=recognition_mode,
            )

        safe_name = secure_filename(uploaded_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"{timestamp}_{safe_name}"
        file_path = UPLOAD_FOLDER / final_filename
        uploaded_file.save(file_path)
        file_is_image = is_image_file(file_path.name)
        should_use_ocr = recognition_mode == "ocr" or not file_is_image

        if should_use_ocr:
            try:
                ocr_result = recognize_document_text(file_path)
                return _render_page(
                    model_ready=model_ready,
                    recognition_mode="ocr",
                    recognized_text=ocr_result["text"],
                    recognized_confidence=ocr_result["confidence"],
                    ocr_candidates=ocr_result["candidates"],
                    uploaded_image=f"uploads/{final_filename}" if file_is_image else None,
                    uploaded_file_name=final_filename,
                )
            except OCRProcessingError as error:
                return _render_page(
                    model_ready=model_ready,
                    recognition_mode="ocr",
                    uploaded_image=f"uploads/{final_filename}" if file_is_image else None,
                    uploaded_file_name=final_filename,
                    error_message=f"OCR failed: {error}",
                )

        if not model_ready:
            return _render_page(
                model_ready=False,
                recognition_mode="character",
                uploaded_image=f"uploads/{final_filename}" if file_is_image else None,
                uploaded_file_name=final_filename,
                error_message=(
                    "Model file not found for character mode. Train with "
                    "`python train.py --dataset_type mnist` or switch to OCR mode."
                ),
            )

        try:
            predicted_label, confidence, _ = predict_character(
                image_path=file_path,
                model=MODEL,
                class_names=CLASS_NAMES,
            )
            return _render_page(
                model_ready=True,
                recognition_mode="character",
                predicted_label=predicted_label,
                confidence=round(confidence, 2),
                uploaded_image=f"uploads/{final_filename}" if file_is_image else None,
                uploaded_file_name=final_filename,
            )
        except Exception as error:  # pylint: disable=broad-except
            return _render_page(
                model_ready=True,
                recognition_mode="character",
                uploaded_image=f"uploads/{final_filename}" if file_is_image else None,
                uploaded_file_name=final_filename,
                error_message=f"Prediction failed: {error}",
            )

    return app


def _render_page(
    model_ready,
    recognition_mode="character",
    predicted_label=None,
    confidence=None,
    uploaded_image=None,
    uploaded_file_name=None,
    recognized_text=None,
    recognized_confidence=None,
    ocr_candidates=None,
    error_message=None,
):
    """Render index page with consistent context keys."""
    return render_template(
        "index.html",
        model_ready=model_ready,
        recognition_mode=recognition_mode,
        predicted_label=predicted_label,
        confidence=confidence,
        uploaded_image=uploaded_image,
        uploaded_file_name=uploaded_file_name,
        recognized_text=recognized_text,
        recognized_confidence=recognized_confidence,
        ocr_candidates=ocr_candidates or [],
        error_message=error_message,
        show_training_curve=TRAINING_CURVE_PATH.exists(),
    )


def is_allowed_file(filename):
    """Check whether uploaded file extension is supported."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image_file(filename):
    """Check whether uploaded file is an image format."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in IMAGE_EXTENSIONS


def try_load_model():
    """Load model and class names if not loaded already."""
    global MODEL, CLASS_NAMES  # pylint: disable=global-statement

    if MODEL is not None and CLASS_NAMES is not None:
        return True

    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        return False

    MODEL, CLASS_NAMES = load_inference_assets(MODEL_PATH, CLASS_NAMES_PATH)
    return True


app = create_app()


if __name__ == "__main__":
    # Keep Matplotlib cache writable in local project for stable startup on Windows.
    mpl_cache_dir = BASE_DIR / ".mplconfig"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    raw_port = os.environ.get("PORT", os.environ.get("FLASK_PORT", "5000"))
    try:
        port = int(raw_port)
    except ValueError as error:
        raise SystemExit(f"PORT or FLASK_PORT must be an integer, got {raw_port!r}.") from error
    if not 1 <= port <= 65535:
        raise SystemExit("PORT or FLASK_PORT must be between 1 and 65535.")
    debug = os.environ.get("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes"}

    # Disable reloader unless debug is explicitly enabled.
    app.run(host=host, port=port, debug=debug, use_reloader=debug)
