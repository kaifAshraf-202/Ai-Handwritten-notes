from pathlib import Path
from io import BytesIO

import fitz
from PIL import Image

from backend.services.ocr import (
    extract_text_with_data
)

from backend.services.text_block_merger import (
    TextBlockMerger
)

from backend.services.content_region_classifier import (
    ContentRegionClassifier
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    "storage/uploads/test.pdf"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 56)
    print("        HANDNOTE AI - TEXT RECONSTRUCTION")
    print("=" * 56)

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # --------------------------------------------------------
    # OPEN PDF
    # --------------------------------------------------------

    document = fitz.open(
        PDF_PATH
    )

    page = document.load_page(
        0
    )

    # --------------------------------------------------------
    # RENDER PAGE
    # --------------------------------------------------------

    print()
    print("Rendering page...")

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

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print()
    print("Running OCR...")

    ocr_result = extract_text_with_data(
        image=image,
        language="eng",
        psm=3,
    )

    ocr_words = ocr_result[
        "words"
    ]

    print(
        f"OCR words: "
        f"{len(ocr_words)}"
    )

    # --------------------------------------------------------
    # TEXT BLOCK MERGING
    # --------------------------------------------------------

    print()
    print("Merging OCR words into text blocks...")

    text_merger = TextBlockMerger()

    text_blocks = text_merger.merge(
        ocr_words
    )

    print(
        f"Text blocks: "
        f"{len(text_blocks)}"
    )

    # --------------------------------------------------------
    # CONTENT CLASSIFICATION
    # --------------------------------------------------------

    print()
    print("Classifying text blocks...")

    classifier = ContentRegionClassifier()

    classified_regions = []

    for block in text_blocks:

        classified = (
            classifier.classify_text_block(
                block
            )
        )

        classified_regions.append(
            classified
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 56)
    print("             TEXT BLOCKS")
    print("=" * 56)

    for block in text_blocks:

        print()

        print(
            f"[T{block.block_id}] "
            f"{block.text}"
        )

        print(
            "    bbox=("
            f"{block.x}, "
            f"{block.y}, "
            f"{block.width}, "
            f"{block.height}"
            ")"
        )

        print(
            f"    confidence="
            f"{block.average_confidence:.2f}%"
        )

        print(
            f"    words="
            f"{block.word_count}"
        )

        print(
            f"    tesseract="
            f"block:{block.tesseract_block} "
            f"paragraph:{block.tesseract_paragraph} "
            f"line:{block.tesseract_line}"
        )

    # --------------------------------------------------------
    # CLASSIFICATION RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 56)
    print("         CONTENT CLASSIFICATION")
    print("=" * 56)

    counts = {}

    for region in classified_regions:

        counts[
            region.region_type
        ] = (
            counts.get(
                region.region_type,
                0
            )
            + 1
        )

        print()

        print(
            f"[T{region.region_id}] "
            f"{region.region_type.upper()}"
        )

        print(
            f"    text="
            f"{region.text}"
        )

        print(
            "    bbox=("
            f"{region.x}, "
            f"{region.y}, "
            f"{region.width}, "
            f"{region.height}"
            ")"
        )

        print(
            f"    confidence="
            f"{region.confidence:.2f}%"
        )

        print(
            f"    reason="
            f"{region.reason}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 56)
    print("          RECONSTRUCTION SUMMARY")
    print("=" * 56)

    print(
        f"OCR words:        "
        f"{len(ocr_words)}"
    )

    print(
        f"Text blocks:      "
        f"{len(text_blocks)}"
    )

    print(
        f"Classified:       "
        f"{len(classified_regions)}"
    )

    print()

    for region_type, count in sorted(
        counts.items()
    ):

        print(
            f"{region_type:<20}"
            f"{count}"
        )

    print()
    print("=" * 56)
    print(
        "   TEXT RECONSTRUCTION COMPLETED"
    )
    print("=" * 56)

    document.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()