from pathlib import Path
from io import BytesIO

import pymupdf

from PIL import Image

from backend.services.ocr import (
    extract_ocr_data
)

from backend.services.region_detector import (
    RegionDetector
)

from backend.services.visual_region_merger import (
    VisualRegionMerger
)

from backend.services.visual_classifier import (
    VisualCandidateClassifier
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

PAGE_NUMBER = 1
DPI = 200


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("    HANDNOTE AI - VISUAL CLASSIFIER")
    print("==========================================\n")

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: "
            f"{PDF_PATH.resolve()}"
        )

    document = pymupdf.open(
        PDF_PATH
    )

    try:

        # ----------------------------------------------------
        # Load page
        # ----------------------------------------------------

        page = document.load_page(
            PAGE_NUMBER - 1
        )

        print(
            f"PDF: {PDF_PATH.name}"
        )

        print(
            f"Page: {PAGE_NUMBER}"
        )

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        print(
            "\nRendering page..."
        )

        pixmap = page.get_pixmap(
            dpi=DPI,
            alpha=False
        )

        image = Image.open(
            BytesIO(
                pixmap.tobytes("png")
            )
        ).convert("RGB")

        print(
            f"Image size: "
            f"{image.width} x {image.height}"
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        print(
            "\nRunning OCR..."
        )

        ocr_result = extract_ocr_data(
            image=image,
            language="eng",
            psm=3
        )

        ocr_words = ocr_result[
            "words"
        ]

        print(
            f"OCR words: "
            f"{len(ocr_words)}"
        )

        # ----------------------------------------------------
        # Region detection
        # ----------------------------------------------------

        print(
            "\nRunning region detection..."
        )

        detector = RegionDetector()

        analysis = detector.analyze_page(
            image=image,
            ocr_words=ocr_words
        )

        visual_candidates = (
            analysis[
                "visual_candidates"
            ]
        )

        print(
            f"Raw visual candidates: "
            f"{len(visual_candidates)}"
        )

        # ----------------------------------------------------
        # Visual merging
        # ----------------------------------------------------

        print(
            "\nRunning visual merger..."
        )

        merger = VisualRegionMerger()

        merged_regions = merger.merge(
            visual_candidates
        )

        print(
            f"Merged regions: "
            f"{len(merged_regions)}"
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        print(
            "\nRunning visual classification..."
        )

        classifier = (
            VisualCandidateClassifier()
        )

        classifications = (
            classifier.classify(
                image=image,
                regions=merged_regions,
                ocr_words=ocr_words
            )
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "        VISUAL CLASSIFICATION"
        )

        print(
            "==========================================\n"
        )

        counts = {}

        for result in classifications:

            classification = (
                result.classification
            )

            counts[
                classification
            ] = (
                counts.get(
                    classification,
                    0
                )
                + 1
            )

        for result in classifications:

            print(
                f"[R{result.region_id}] "
                f"{result.classification.upper()}"
            )

            print(
                f"    bbox=("
                f"{result.x}, "
                f"{result.y}, "
                f"{result.width}, "
                f"{result.height}"
                f")"
            )

            print(
                f"    confidence="
                f"{result.confidence * 100:.2f}%"
            )

            print(
                f"    OCR overlap="
                f"{result.ocr_overlap_ratio * 100:.2f}%"
            )

            print(
                f"    visual ink="
                f"{result.visual_ink_ratio * 100:.2f}%"
            )

            print(
                f"    colour="
                f"{result.color_ratio * 100:.2f}%"
            )

            print(
                f"    edges="
                f"{result.edge_ratio * 100:.2f}%"
            )

            print(
                f"    reason="
                f"{result.reason}"
            )

            print()

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            "=========================================="
        )

        print(
            "             CLASSIFICATION SUMMARY"
        )

        print(
            "==========================================\n"
        )

        for classification, count in sorted(
            counts.items()
        ):

            print(
                f"{classification:<18}"
                f"{count}"
            )

        print(
            "\n=========================================="
        )

        print(
            "     VISUAL CLASSIFIER COMPLETED"
        )

        print(
            "==========================================\n"
        )

    finally:

        document.close()


if __name__ == "__main__":
    main()