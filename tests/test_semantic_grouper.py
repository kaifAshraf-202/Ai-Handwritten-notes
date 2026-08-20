from pathlib import Path
from io import BytesIO

import fitz
from PIL import Image

from backend.services.ocr import (
    extract_text_with_data
)

from backend.services.region_detector import (
    RegionDetector
)

from backend.services.visual_merger import (
    VisualMerger
)

from backend.services.semantic_grouper import (
    SemanticGrouper
)


# ============================================================
# PATH
# ============================================================

PDF_PATH = Path(
    "storage/uploads/test.pdf"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 48)
    print("     HANDNOTE AI - SEMANTIC GROUPER")
    print("=" * 48)

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    document = fitz.open(
        PDF_PATH
    )

    page = document.load_page(
        0
    )

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    print(
        "\nRendering page..."
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

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Region detection
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Visual merger
    # --------------------------------------------------------

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
        f"Merged visual regions: "
        f"{len(merged_regions)}"
    )

    # --------------------------------------------------------
    # Semantic grouping
    # --------------------------------------------------------

    print(
        "\nRunning semantic grouping..."
    )

    grouper = SemanticGrouper()

    result = (
        grouper.analyze_page(
            merged_regions,
            ocr_words
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 48)
    print("        SEMANTIC REGIONS")
    print("=" * 48)

    for region in result["regions"]:

        print()

        print(
            f"[R{region['region_id']}] "
            f"{region['region_type'].upper()}"
        )

        print(
            "    "
            f"bbox=("
            f"{region['x']}, "
            f"{region['y']}, "
            f"{region['width']}, "
            f"{region['height']}"
            ")"
        )

        text = (
            region["metadata"]
            .get(
                "text",
                ""
            )
        )

        if text:

            print(
                f"    text={text}"
            )

        print(
            "    "
            f"parent="
            f"{region['parent_region_id']}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 48)
    print("          GROUPING SUMMARY")
    print("=" * 48)

    for region_type, count in sorted(
        result["counts"].items()
    ):

        print(
            f"{region_type:<18}"
            f"{count}"
        )

    print()
    print(
        f"Total semantic regions: "
        f"{result['total']}"
    )

    print()
    print("=" * 48)
    print(
        "   SEMANTIC GROUPING COMPLETED"
    )
    print("=" * 48)

    document.close()


if __name__ == "__main__":
    main()