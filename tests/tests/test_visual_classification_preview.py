from pathlib import Path
from io import BytesIO

import fitz
from PIL import Image, ImageDraw, ImageFont

from backend.services.ocr import extract_text_with_data
from backend.services.region_detector import RegionDetector
from backend.services.visual_merger import VisualMerger
from backend.services.visual_classifier import (
    VisualCandidateClassifier,
)


# ============================================================
# PATHS
# ============================================================

PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

OUTPUT_PATH = Path(
    "storage/processing/"
    "page_1_classification_preview.png"
)


# ============================================================
# FONT
# ============================================================

def get_font(size=24):

    possible_fonts = [

        "C:/Windows/Fonts/arial.ttf",

        "C:/Windows/Fonts/segoeui.ttf",

        "C:/Windows/Fonts/calibri.ttf",

    ]

    for font_path in possible_fonts:

        path = Path(font_path)

        if path.exists():

            try:

                return ImageFont.truetype(
                    str(path),
                    size
                )

            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# CLASSIFICATION COLORS
# ============================================================

CLASSIFICATION_COLORS = {

    "handwriting": (
        255,
        0,
        255
    ),

    "highlight": (
        255,
        200,
        0
    ),

    "diagram": (
        0,
        120,
        255
    ),

    "graphic": (
        0,
        220,
        100
    ),

    "annotation": (
        255,
        120,
        0
    ),

    "text_artifact": (
        180,
        180,
        180
    ),

    "unknown": (
        255,
        255,
        255
    ),
}


# ============================================================
# DRAW CLASSIFICATION
# ============================================================

def draw_classification(
    draw,
    result,
    font
):

    classification = (
        result.classification
    )

    color = (
        CLASSIFICATION_COLORS.get(
            classification,
            (255, 255, 255)
        )
    )

    x = result.x
    y = result.y

    width = result.width
    height = result.height

    x2 = x + width
    y2 = y + height

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    draw.rectangle(
        (
            x,
            y,
            x2,
            y2
        ),
        outline=color,
        width=4
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (
        f"R{result.region_id} "
        f"{classification.upper()}"
    )

    bbox = draw.textbbox(
        (
            0,
            0
        ),
        label,
        font=font
    )

    label_width = (
        bbox[2] - bbox[0]
    )

    label_height = (
        bbox[3] - bbox[1]
    )

    label_x = x

    label_y = max(
        0,
        y - label_height - 8
    )

    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

    draw.rectangle(
        (
            label_x,
            label_y,
            label_x + label_width + 10,
            label_y + label_height + 8
        ),
        fill=color
    )

    # --------------------------------------------------------
    # Label text
    # --------------------------------------------------------

    draw.text(
        (
            label_x + 5,
            label_y + 3
        ),
        label,
        fill=(0, 0, 0),
        font=font
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 46)
    print("  HANDNOTE AI - CLASSIFICATION PREVIEW")
    print("=" * 46)

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # OPEN PDF
    # ========================================================

    document = fitz.open(
        PDF_PATH
    )

    page_number = 1

    page = document.load_page(
        page_number - 1
    )

    print(
        f"\nProcessing page: "
        f"{page_number}"
    )

    # ========================================================
    # RENDER PAGE
    # ========================================================

    print(
        "Rendering page..."
    )

    pixmap = page.get_pixmap(
        dpi=200,
        alpha=False
    )

    image = Image.open(
        BytesIO(
            pixmap.tobytes(
                "png"
            )
        )
    ).convert(
        "RGB"
    )

    print(
        f"Image size: "
        f"{image.width} x "
        f"{image.height}"
    )

    # ========================================================
    # OCR
    # ========================================================

    print(
        "\nRunning OCR..."
    )

    ocr_result = (
        extract_text_with_data(
            image
        )
    )

    ocr_words = (
        ocr_result["words"]
    )

    print(
        f"OCR words: "
        f"{len(ocr_words)}"
    )

    # ========================================================
    # REGION DETECTION
    # ========================================================

    print(
        "\nRunning region detection..."
    )

    detector = RegionDetector()

    visual_candidates = (
        detector.detect_contours(
            image
        )
    )

    print(
        f"Raw visual candidates: "
        f"{len(visual_candidates)}"
    )

    # ========================================================
    # VISUAL MERGING
    # ========================================================

    print(
        "\nRunning visual merger..."
    )

    merger = VisualMerger()

    merged_regions = (
        merger.merge(
            visual_candidates
        )
    )

    print(
        f"Merged regions: "
        f"{len(merged_regions)}"
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    print(
        "\nRunning visual classification..."
    )

    classifier = (
        VisualCandidateClassifier()
    )

    classifications = (
        classifier.classify(
            image,
            merged_regions,
            ocr_words
        )
    )

    # ========================================================
    # CREATE PREVIEW
    # ========================================================

    preview = image.copy()

    draw = ImageDraw.Draw(
        preview
    )

    font = get_font(
        24
    )

    for result in classifications:

        draw_classification(
            draw,
            result,
            font
        )

    # ========================================================
    # SAVE
    # ========================================================

    preview.save(
        OUTPUT_PATH
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    counts = {}

    for result in classifications:

        classification = (
            result.classification
        )

        counts[classification] = (
            counts.get(
                classification,
                0
            ) + 1
        )

    print()
    print("=" * 46)
    print("          CLASSIFICATION SUMMARY")
    print("=" * 46)

    for classification, count in sorted(
        counts.items()
    ):

        print(
            f"{classification:<18}"
            f"{count}"
        )

    print()
    print(
        "Preview saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Legend:"
    )

    print(
        "PINK   = HANDWRITING"
    )

    print(
        "YELLOW = HIGHLIGHT"
    )

    print(
        "BLUE   = DIAGRAM"
    )

    print(
        "GREEN  = GRAPHIC"
    )

    print()
    print("=" * 46)
    print(
        "   CLASSIFICATION PREVIEW COMPLETED"
    )
    print("=" * 46)

    document.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()