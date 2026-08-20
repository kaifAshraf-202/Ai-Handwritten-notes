from typing import Dict, Any, List

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = (
    TESSERACT_PATH
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image
) -> Image.Image:
    """
    Prepare image for OCR.

    The original image is not modified.
    """

    processed = image.convert(
        "L"
    )

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
    useful mathematical/symbolic content.
    """

    lines = []

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
# RAW OCR DATA
# ============================================================

def extract_text_with_data(
    image: Image.Image,
    language: str = "eng"
) -> Dict[str, Any]:
    """
    Run Tesseract OCR and return:

        - complete OCR text
        - individual words
        - bounding boxes
        - confidence
        - Tesseract block information
        - line information

    This function is used by the region-analysis
    pipeline.
    """

    processed_image = (
        preprocess_image(
            image
        )
    )

    # --------------------------------------------------------
    # Tesseract configuration
    # --------------------------------------------------------

    config = (
        "--oem 3 "
        "--psm 3"
    )

    # --------------------------------------------------------
    # Get detailed OCR data
    # --------------------------------------------------------

    data = pytesseract.image_to_data(
        processed_image,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DICT
    )

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

        text = str(
            data["text"][index]
        ).strip()

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

        # ----------------------------------------------------
        # Hierarchy
        # ----------------------------------------------------

        block_num = int(
            data["block_num"][index]
        )

        paragraph_num = int(
            data["par_num"][index]
        )

        line_num = int(
            data["line_num"][index]
        )

        word_num = int(
            data["word_num"][index]
        )

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

    # --------------------------------------------------------
    # Build complete text
    # --------------------------------------------------------

    text_lines = []

    current_line_key = None
    current_line_words = []

    for word in words:

        line_key = (
            word["block_num"],
            word["paragraph_num"],
            word["line_num"]
        )

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

        current_line_key = (
            line_key
        )

        current_line_words.append(
            word["text"]
        )

    # --------------------------------------------------------
    # Last line
    # --------------------------------------------------------

    if current_line_words:

        text_lines.append(
            " ".join(
                current_line_words
            )
        )

    full_text = "\n".join(
        text_lines
    )

    return {

        "text": clean_text(
            full_text
        ),

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

    Existing parts of the project use this function,
    so its interface remains unchanged.
    """

    result = extract_text_with_data(
        image,
        language=language
    )

    return result["text"]