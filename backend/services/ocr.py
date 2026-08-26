from typing import Dict, Any, List

import pytesseract

from PIL import Image, ImageEnhance, ImageFilter


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image
) -> Image.Image:
    """
    Prepare an image for OCR.

    The original image is not modified.

    Processing:
        1. Convert to grayscale
        2. Improve contrast
        3. Apply slight sharpening
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be a PIL.Image.Image"
        )

    # Convert RGB/RGBA/etc. to grayscale.
    processed = image.convert("L")

    # Improve contrast.
    processed = ImageEnhance.Contrast(
        processed
    ).enhance(1.5)

    # Slight sharpening.
    processed = processed.filter(
        ImageFilter.SHARPEN
    )

    return processed


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_text(
    text: str
) -> str:
    """
    Clean OCR output while preserving
    useful mathematical and symbolic content.

    Empty lines are removed.
    """

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lines.append(line)

    return "\n".join(lines)


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

            "image_width": int,
            "image_height": int
        }

    Parameters
    ----------
    image:
        PIL image to process.

    psm:
        Tesseract Page Segmentation Mode.

        Common values:
            3  = Fully automatic page segmentation
            4  = Assume a single column
            6  = Assume a single uniform block of text
            11 = Sparse text

    language:
        Tesseract language code.
        Default: "eng"
    """

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not isinstance(
        image,
        Image.Image
    ):
        raise TypeError(
            "image must be a PIL.Image.Image"
        )

    # --------------------------------------------------------
    # Validate PSM
    # --------------------------------------------------------

    try:
        psm = int(psm)
    except (
        TypeError,
        ValueError
    ):
        raise ValueError(
            f"Invalid PSM value: {psm}"
        )

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    processed_image = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # Tesseract configuration
    # --------------------------------------------------------

    config = (
        f"--oem 3 --psm {psm}"
    )

    # --------------------------------------------------------
    # Run Tesseract
    # --------------------------------------------------------

    data = pytesseract.image_to_data(
        processed_image,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    # ========================================================
    # EXTRACT OCR WORDS
    # ========================================================

    words: List[Dict[str, Any]] = []

    total_items = len(
        data.get(
            "text",
            []
        )
    )

    for index in range(
        total_items
    ):

        # ----------------------------------------------------
        # OCR text
        # ----------------------------------------------------

        text = str(
            data["text"][index]
        ).strip()

        # Ignore empty OCR entries.
        if not text:
            continue

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        try:

            confidence = float(
                data["conf"][index]
            )

        except (
            ValueError,
            TypeError
        ):

            confidence = 0.0

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        try:

            left = int(
                data["left"][index]
            )

            top = int(
                data["top"][index]
            )

            width = int(
                data["width"][index]
            )

            height = int(
                data["height"][index]
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        # ----------------------------------------------------
        # Tesseract hierarchy
        # ----------------------------------------------------

        try:

            block_num = int(
                data["block_num"][index]
            )

        except (
            ValueError,
            TypeError
        ):

            block_num = 0

        try:

            paragraph_num = int(
                data["par_num"][index]
            )

        except (
            ValueError,
            TypeError
        ):

            paragraph_num = 0

        try:

            line_num = int(
                data["line_num"][index]
            )

        except (
            ValueError,
            TypeError
        ):

            line_num = 0

        try:

            word_num = int(
                data["word_num"][index]
            )

        except (
            ValueError,
            TypeError
        ):

            word_num = 0

        # ----------------------------------------------------
        # Store OCR word
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
    # BUILD COMPLETE OCR TEXT
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
        # New line detected
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
    # ADD LAST LINE
    # ========================================================

    if current_line_words:

        text_lines.append(
            " ".join(
                current_line_words
            )
        )

    # ========================================================
    # COMPLETE OCR TEXT
    # ========================================================

    full_text = "\n".join(
        text_lines
    )

    cleaned_text = clean_text(
        full_text
    )

    # ========================================================
    # RETURN OCR RESULT
    # ========================================================

    return {
        "text": cleaned_text,

        "words": words,

        "word_count": len(
            words
        ),

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

    Returns only the extracted text.

    Existing project modules use this function,
    so its interface remains unchanged.
    """

    result = extract_text_with_data(
        image=image,
        psm=3,
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
    Backward-compatible OCR API.

    IMPORTANT:
    This function returns the COMPLETE OCR RESULT,
    not just the words list.

    Therefore this works:

        result = extract_ocr_data(image)

        words = result["words"]

    It also supports:

        result = extract_ocr_data(
            image=image,
            psm=3,
            language="eng"
        )
    """

    return extract_text_with_data(
        image=image,
        psm=psm,
        language=language
    )