"""OCR utilities for recognizing handwritten text from images and PDFs."""

from pathlib import Path

import cv2
import numpy as np
import pymupdf


class OCRProcessingError(RuntimeError):
    """Raised when OCR processing cannot be completed."""


_OCR_ENGINE = None


try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover - optional enhancement dependency
    zipf_frequency = None


def recognize_document_text(document_path):
    """
    Recognize text from an image or PDF document.

    Returns:
        dict: {
            "text": str,
            "confidence": float | None,  # in percent (0-100)
            "candidates": list[dict],    # top OCR candidates for image mode
            "pages": int,
        }
    """
    document_path = Path(document_path)
    suffix = document_path.suffix.lower()

    if suffix == ".pdf":
        return _recognize_pdf_text(document_path)
    return _recognize_image_text(document_path)


def _get_ocr_engine():
    """Lazy-load OCR engine so app startup remains fast."""
    global _OCR_ENGINE  # pylint: disable=global-statement
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as error:
        raise OCRProcessingError(
            "OCR dependency missing. Install: pip install rapidocr-onnxruntime"
        ) from error

    _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _recognize_pdf_text(pdf_path):
    """Run OCR page-by-page on a PDF."""
    if not pdf_path.exists():
        raise OCRProcessingError(f"PDF file not found: {pdf_path}")

    text_chunks = []
    weighted_confidence_sum = 0.0
    total_weight = 0.0
    pages_with_text = 0

    try:
        document = pymupdf.open(pdf_path)
    except Exception as error:  # pylint: disable=broad-except
        raise OCRProcessingError(f"Could not open PDF: {error}") from error

    with document:
        for page in document:
            page_image = _render_pdf_page_to_bgr(page)
            page_result = _recognize_image_array(page_image)
            if not page_result["text"]:
                continue

            pages_with_text += 1
            text_chunks.append(page_result["text"])

            if page_result["confidence"] is not None:
                weight = max(len(page_result["text"]), 1)
                weighted_confidence_sum += page_result["confidence"] * weight
                total_weight += weight

    combined_text = "\n".join(chunk for chunk in text_chunks if chunk.strip()).strip()
    if not combined_text:
        raise OCRProcessingError("No readable text was detected in the PDF.")

    combined_confidence = None
    if total_weight > 0:
        combined_confidence = round((weighted_confidence_sum / total_weight) * 100.0, 2)

    return {
        "text": combined_text,
        "confidence": combined_confidence,
        "candidates": [],
        "pages": pages_with_text,
    }


def _recognize_image_text(image_path):
    """Run OCR on a single image file with multiple preprocessing variants."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise OCRProcessingError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OCRProcessingError(f"Could not read image file: {image_path}")

    result = _recognize_image_array(image)
    if not result["text"]:
        raise OCRProcessingError("No readable text was detected in the image.")

    return {
        "text": result["text"],
        "confidence": result["confidence_percent"],
        "candidates": result["candidates"],
        "pages": 1,
    }


def _render_pdf_page_to_bgr(page):
    """Render a PyMuPDF page into a BGR NumPy array."""
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )

    if pixmap.n == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if pixmap.n == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def _recognize_image_array(image_bgr):
    """Run OCR across multiple variants and choose the best result."""
    engine = _get_ocr_engine()
    candidates = []
    seen_text = set()
    raw_candidates = []

    for variant_name, variant_image in _build_image_variants(image_bgr):
        raw_result, _ = engine(variant_image)
        parsed = _parse_ocr_lines(raw_result)
        if not parsed["text"]:
            continue

        normalized_text = parsed["text"].lower().strip()
        if normalized_text in seen_text:
            continue
        seen_text.add(normalized_text)
        raw_candidates.append(parsed["text"])

        score = _candidate_score(
            text=parsed["text"],
            avg_confidence=parsed["confidence"],
            line_count=len(parsed["lines"]),
        )

        candidates.append(
            {
                "variant": variant_name,
                "text": parsed["text"],
                "confidence": parsed["confidence"],
                "confidence_percent": round(parsed["confidence"] * 100.0, 2)
                if parsed["confidence"] is not None
                else None,
                "score": score,
            }
        )

    lexicon_hints = _collect_hint_tokens(raw_candidates)

    for item in candidates:
        corrected_text, correction_gain = _apply_lexical_correction(
            item["text"], hint_tokens=lexicon_hints
        )
        item["corrected_text"] = corrected_text
        item["score"] += correction_gain

    if not candidates:
        return {"text": "", "confidence": None, "confidence_percent": None, "candidates": []}

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    top_candidates = [
        {
            "text": item["corrected_text"],
            "confidence": item["confidence_percent"],
            "variant": item["variant"],
        }
        for item in candidates[:3]
    ]

    return {
        "text": best["corrected_text"],
        "confidence": best["confidence"],
        "confidence_percent": best["confidence_percent"],
        "candidates": top_candidates,
    }


def _build_image_variants(image_bgr):
    """Create multiple image variants to improve OCR robustness."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cropped = _crop_foreground(gray)

    variants = []

    # Original color image.
    variants.append(("original", image_bgr))

    # Cropped region to remove large blank margins.
    variants.append(("cropped", cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)))

    # Adaptive threshold often helps with faint pen strokes.
    adaptive = cv2.adaptiveThreshold(
        cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    variants.append(("adaptive_threshold", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))

    # Slight upscale can improve recognition for small handwriting.
    upscaled = cv2.resize(
        cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR),
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )
    variants.append(("cropped_upscaled", upscaled))

    return variants


def _crop_foreground(gray_image):
    """Crop image to likely ink region while keeping a small margin."""
    inverted = cv2.threshold(gray_image, 245, 255, cv2.THRESH_BINARY_INV)[1]
    points = cv2.findNonZero(inverted)
    if points is None:
        return gray_image

    x, y, width, height = cv2.boundingRect(points)
    margin = 24
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(gray_image.shape[1], x + width + margin)
    y1 = min(gray_image.shape[0], y + height + margin)
    return gray_image[y0:y1, x0:x1]


def _parse_ocr_lines(raw_result):
    """Normalize raw RapidOCR output into text + confidence values."""
    lines = []
    for item in raw_result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue

        text = str(item[1]).strip()
        if not text:
            continue

        try:
            score = float(item[2])
        except (TypeError, ValueError):
            score = None

        lines.append({"text": text, "confidence": score})

    joined_text = " ".join(line["text"] for line in lines).strip()
    score_values = [line["confidence"] for line in lines if line["confidence"] is not None]
    average_confidence = sum(score_values) / len(score_values) if score_values else None

    return {"text": joined_text, "lines": lines, "confidence": average_confidence}


def _candidate_score(text, avg_confidence, line_count):
    """
    Rank OCR candidates by confidence + plausibility.

    Long multi-word text is lightly favored when confidence is similar.
    """
    confidence = avg_confidence if avg_confidence is not None else 0.0
    length_bonus = min(0.25, len(text) * 0.01)
    word_bonus = 0.10 if line_count > 1 or " " in text else 0.0
    return confidence + length_bonus + word_bonus


def _collect_hint_tokens(text_candidates):
    """Collect additional token hints from all OCR variants."""
    hints = set()
    for text in text_candidates:
        for token in text.split():
            clean = "".join(char for char in token if char.isalpha()).lower()
            if len(clean) >= 3:
                hints.add(clean)
    return hints


def _apply_lexical_correction(text, hint_tokens):
    """
    Correct likely OCR misspellings using word frequency + fuzzy hint matching.

    Returns:
        tuple[str, float]: corrected_text, score_gain
    """
    if not text or zipf_frequency is None:
        return text, 0.0

    corrected_words = []
    gain = 0.0

    for raw_word in text.split():
        word = raw_word.strip()
        if not word:
            continue

        prefix, core, suffix = _split_word_affixes(word)
        if not core or not core.isalpha():
            corrected_words.append(word)
            continue

        original_lower = core.lower()
        original_zipf = zipf_frequency(original_lower, "en")

        candidate_lower, candidate_zipf = _best_word_candidate(
            original_lower, original_zipf, hint_tokens
        )

        if candidate_lower != original_lower:
            corrected_core = _restore_case(core, candidate_lower)
            corrected_words.append(f"{prefix}{corrected_core}{suffix}")
            gain += min(0.25, max(0.0, candidate_zipf - original_zipf) * 0.07)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words), gain


def _split_word_affixes(word):
    """Split token into prefix punctuation, alpha core, suffix punctuation."""
    start = 0
    end = len(word)
    while start < end and not word[start].isalnum():
        start += 1
    while end > start and not word[end - 1].isalnum():
        end -= 1

    prefix = word[:start]
    core = word[start:end]
    suffix = word[end:]
    return prefix, core, suffix


def _best_word_candidate(word_lower, word_zipf, hint_tokens):
    """Pick best lexical candidate for a possibly misspelled word."""
    if len(word_lower) < 3:
        return word_lower, word_zipf

    best_word = word_lower
    best_score = _lexical_score(
        word_lower, word_zipf, distance=0, original_word=word_lower
    )
    candidates = set()

    # Hints from other OCR variants are strong candidates when text is similar.
    for hint in hint_tokens:
        distance = _levenshtein_distance(word_lower, hint)
        if distance <= 2:
            candidates.add((hint, distance))

    # Generate compact edit-distance-1 candidates.
    for generated in _generate_edit_candidates(word_lower):
        distance = _levenshtein_distance(word_lower, generated)
        if distance <= 2:
            candidates.add((generated, distance))

    for cand, distance in candidates:
        cand_zipf = zipf_frequency(cand, "en")
        if cand_zipf <= 0.0:
            continue
        cand_score = _lexical_score(
            cand, cand_zipf, distance=distance, original_word=word_lower
        )
        if cand_score > best_score:
            best_word = cand
            best_score = cand_score

    if best_word == word_lower:
        return word_lower, word_zipf

    best_zipf = zipf_frequency(best_word, "en")
    if (best_zipf - word_zipf) < 0.8:
        return word_lower, word_zipf
    return best_word, best_zipf


def _generate_edit_candidates(word):
    """Generate plausible nearby spellings for OCR errors."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    vowels = "aeiou"
    candidates = set()

    for index, char in enumerate(word):
        # OCR confusion-aware replacements.
        for repl in _confusable_replacements(char):
            if repl != char:
                candidates.add(word[:index] + repl + word[index + 1 :])

        # Vowel swaps are very common in handwriting OCR.
        if char in vowels:
            for vowel in vowels:
                if vowel != char:
                    candidates.add(word[:index] + vowel + word[index + 1 :])

        # Character deletion.
        candidates.add(word[:index] + word[index + 1 :])

        # Adjacent transpose.
        if index < len(word) - 1:
            transposed = (
                word[:index]
                + word[index + 1]
                + word[index]
                + word[index + 2 :]
            )
            candidates.add(transposed)

    # Insertion candidates using neighboring characters.
    for index in range(len(word) + 1):
        neighbor_chars = set()
        if index > 0:
            neighbor_chars.add(word[index - 1])
        if index < len(word):
            neighbor_chars.add(word[index])
        if not neighbor_chars:
            neighbor_chars = set(letters)
        for insert_char in neighbor_chars:
            candidates.add(word[:index] + insert_char + word[index:])

    # Keep search bounded.
    return {cand for cand in candidates if 2 <= len(cand) <= 20}


def _confusable_replacements(char):
    """Character-level OCR confusions."""
    table = {
        "a": "aoe",
        "c": "eo",
        "d": "cl",
        "e": "cao",
        "g": "q",
        "h": "bn",
        "i": "jl",
        "l": "id",
        "m": "nnw",
        "n": "mh",
        "o": "ace",
        "q": "g",
        "r": "n",
        "s": "a",
        "t": "l",
        "u": "v",
        "v": "u",
        "w": "m",
        "y": "v",
    }
    return table.get(char, "")


def _lexical_score(word, zipf_value, distance, original_word):
    """Higher score means better correction candidate."""
    base = zipf_value
    distance_penalty = distance * 0.45
    length_penalty = abs(len(word) - len(original_word)) * 0.45
    deletion_penalty = 0.25 if len(word) < len(original_word) else 0.0
    short_word_penalty = 0.35 if len(word) <= 3 else 0.0
    return base - distance_penalty - length_penalty - deletion_penalty - short_word_penalty


def _restore_case(template_word, new_word_lower):
    """Restore original token casing style."""
    if template_word.isupper():
        return new_word_lower.upper()
    if template_word.istitle():
        return new_word_lower.title()
    return new_word_lower


def _levenshtein_distance(left, right):
    """Compute Levenshtein edit distance."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        curr = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            curr.append(
                min(
                    prev[j] + 1,       # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = curr
    return prev[-1]
