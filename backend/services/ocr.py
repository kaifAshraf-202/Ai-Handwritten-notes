from typing import Dict, Any, List

import pytesseract

from PIL import Image, ImageEnhance


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# OCR CONFIGURATION
# ============================================================

DEFAULT_PSM = 3
DEFAULT_LANGUAGE = "eng"

# Prevent Tesseract from hanging indefinitely.
OCR_TIMEOUT = 60


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image
) -> Image.Image:
    """
    Prepare image for OCR.

    The original image is never modified.

    Processing:
        1. Convert to grayscale.
        2. Improve contrast slightly.

    We intentionally avoid aggressive sharpening because
    the complete HandNote AI pipeline already performs
    additional image processing.
    """

    if not isinstance(
        image,
        Image.Image
    ):
        raise TypeError(
            "image must be a PIL.Image.Image"
        )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    processed = image.convert(
        "L"
    )

    # --------------------------------------------------------
    # Improve contrast
    # --------------------------------------------------------

    processed = ImageEnhance.Contrast(
        processed
    ).enhance(
        1.35
    )

    return processed


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_text(
    text: str
) -> str:
    """
    Clean OCR output while preserving useful
    mathematical and symbolic content.

    Empty lines are removed.
    """

    if not text:
        return ""

    lines: List[str] = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0
) -> float:
    """
    Safely convert a value to float.
    """

    try:

        return float(
            value
        )

    except (
        ValueError,
        TypeError
    ):

        return default


def _safe_int(
    value: Any,
    default: int = 0
) -> int:
    """
    Safely convert a value to int.
    """

    try:

        return int(
            value
        )

    except (
        ValueError,
        TypeError
    ):

        return default


# ============================================================
# TESSERACT VALIDATION
# ============================================================

def _validate_tesseract() -> None:
    """
    Verify that Tesseract is available.
    """

    try:

        pytesseract.get_tesseract_version()

    except pytesseract.TesseractNotFoundError as exc:

        raise RuntimeError(
            "Tesseract OCR executable was not found. "
            f"Expected location: {TESSERACT_PATH}"
        ) from exc

    except Exception as exc:

        raise RuntimeError(
            "Unable to start Tesseract OCR. "
            f"Expected location: {TESSERACT_PATH}"
        ) from exc


# ============================================================
# RAW OCR DATA
# ============================================================

def extract_text_with_data(
    image: Image.Image,
    psm: int = 3,
    language: str = "eng"
) -> Dict[str, Any]:
    """
    Run Tesseract OCR and return detailed OCR information.

    Returns:

        {
            "text": str,

            "words": [
                {
                    "text": str,
                    "confidence": float,

                    "left": int,
                    "top": int,
                    "width": int,
                    "height": int,

                    "block_num": int,
                    "paragraph_num": int,
                    "line_num": int,
                    "word_num": int
                }
            ],

            "word_count": int,
            "average_confidence": float,

            "psm": int,
            "language": str,

            "image_width": int,
            "image_height": int
        }
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if not isinstance(
        image,
        Image.Image
    ):
        raise TypeError(
            "image must be a PIL.Image.Image"
        )

    try:

        psm = int(
            psm
        )

    except (
        TypeError,
        ValueError
    ) as exc:

        raise ValueError(
            f"Invalid PSM value: {psm}"
        ) from exc

    if psm < 0 or psm > 13:

        raise ValueError(
            f"Invalid PSM value: {psm}. "
            "Expected a value between 0 and 13."
        )

    if not language:

        language = DEFAULT_LANGUAGE

    language = str(
        language
    )

    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================

    processed_image = preprocess_image(
        image
    )

    # ========================================================
    # TESSERACT CONFIGURATION
    # ========================================================

    config = (
        f"--oem 3 --psm {psm}"
    )

    # ========================================================
    # RUN TESSERACT
    # ========================================================

    try:

        data = pytesseract.image_to_data(
            processed_image,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=OCR_TIMEOUT
        )

    except RuntimeError as exc:

        raise RuntimeError(
            "Tesseract OCR timed out or failed. "
            f"Timeout: {OCR_TIMEOUT} seconds. "
            f"Image size: "
            f"{image.width}x{image.height}. "
            f"PSM: {psm}. "
            f"Language: {language}."
        ) from exc

    except pytesseract.TesseractNotFoundError as exc:

        raise RuntimeError(
            "Tesseract executable was not found. "
            f"Expected: {TESSERACT_PATH}"
        ) from exc

    # ========================================================
    # EXTRACT WORDS
    # ========================================================

    words: List[Dict[str, Any]] = []

    text_data = data.get(
        "text",
        []
    )

    total_items = len(
        text_data
    )

    for index in range(
        total_items
    ):

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text = str(
            text_data[index]
        ).strip()

        if not text:
            continue

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence_data = data.get(
            "conf",
            []
        )

        confidence = _safe_float(
            confidence_data[index]
            if index < len(
                confidence_data
            )
            else 0.0
        )

        # ----------------------------------------------------
        # COORDINATES
        # ----------------------------------------------------

        left_data = data.get(
            "left",
            []
        )

        top_data = data.get(
            "top",
            []
        )

        width_data = data.get(
            "width",
            []
        )

        height_data = data.get(
            "height",
            []
        )

        left = _safe_int(
            left_data[index]
            if index < len(
                left_data
            )
            else 0
        )

        top = _safe_int(
            top_data[index]
            if index < len(
                top_data
            )
            else 0
        )

        width = _safe_int(
            width_data[index]
            if index < len(
                width_data
            )
            else 0
        )

        height = _safe_int(
            height_data[index]
            if index < len(
                height_data
            )
            else 0
        )

        # ----------------------------------------------------
        # TESSERACT HIERARCHY
        # ----------------------------------------------------

        block_data = data.get(
            "block_num",
            []
        )

        paragraph_data = data.get(
            "par_num",
            []
        )

        line_data = data.get(
            "line_num",
            []
        )

        word_data = data.get(
            "word_num",
            []
        )

        block_num = _safe_int(
            block_data[index]
            if index < len(
                block_data
            )
            else 0
        )

        paragraph_num = _safe_int(
            paragraph_data[index]
            if index < len(
                paragraph_data
            )
            else 0
        )

        line_num = _safe_int(
            line_data[index]
            if index < len(
                line_data
            )
            else 0
        )

        word_num = _safe_int(
            word_data[index]
            if index < len(
                word_data
            )
            else 0
        )

        # ----------------------------------------------------
        # STORE WORD
        # ----------------------------------------------------

        words.append(
            {
                "text": text,

                "confidence": confidence,

                "left": left,
                "top": top,
                "width": width,
                "height": height,

                "block_num": block_num,
                "paragraph_num": paragraph_num,
                "line_num": line_num,
                "word_num": word_num,
            }
        )

    # ========================================================
    # BUILD COMPLETE TEXT
    # ========================================================

    text_lines: List[str] = []

    current_line_key = None

    current_line_words: List[str] = []

    for word in words:

        line_key = (
            word["block_num"],
            word["paragraph_num"],
            word["line_num"]
        )

        # ----------------------------------------------------
        # New OCR line
        # ----------------------------------------------------

        if (
            current_line_key is not None
            and line_key != current_line_key
        ):

            if current_line_words:

                text_lines.append(
                    " ".join(
                        current_line_words
                    )
                )

            current_line_words = []

        current_line_key = line_key

        current_line_words.append(
            word["text"]
        )

    # ========================================================
    # LAST LINE
    # ========================================================

    if current_line_words:

        text_lines.append(
            " ".join(
                current_line_words
            )
        )

    # ========================================================
    # COMPLETE TEXT
    # ========================================================

    full_text = "\n".join(
        text_lines
    )

    cleaned_text = clean_text(
        full_text
    )

    # ========================================================
    # OCR STATISTICS
    # ========================================================

    word_count = len(
        words
    )

    if word_count > 0:

        average_confidence = (
            sum(
                word["confidence"]
                for word in words
            )
            / word_count
        )

    else:

        average_confidence = 0.0

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        # ----------------------------------------------------
        # OCR CONTENT
        # ----------------------------------------------------

        "text": cleaned_text,

        "words": words,

        # ----------------------------------------------------
        # OCR STATISTICS
        # ----------------------------------------------------

        "word_count": word_count,

        "average_confidence": average_confidence,

        # ----------------------------------------------------
        # OCR CONFIGURATION
        # ----------------------------------------------------

        "psm": psm,

        "language": language,

        # ----------------------------------------------------
        # IMAGE INFORMATION
        # ----------------------------------------------------

        "image_width": image.width,

        "image_height": image.height,
    }


# ============================================================
# SIMPLE OCR API
# ============================================================

def extract_text(
    image: Image.Image,
    language: str = "eng"
) -> str:
    """
    Simple OCR API.

    Returns only extracted OCR text.
    """

    result = extract_text_with_data(
        image=image,
        psm=DEFAULT_PSM,
        language=language
    )

    return result["text"]


# ============================================================
# BACKWARD COMPATIBILITY API
# ============================================================

def extract_ocr_data(
    image: Image.Image,
    psm: int = 3,
    language: str = "eng"
) -> Dict[str, Any]:
    """
    Backward-compatible OCR data API.

    Returns the complete OCR result dictionary.

    Example:

        result = extract_ocr_data(
            image=image,
            psm=3,
            language="eng"
        )

        words = result["words"]
    """

    return extract_text_with_data(
        image=image,
        psm=psm,
        language=language
    )