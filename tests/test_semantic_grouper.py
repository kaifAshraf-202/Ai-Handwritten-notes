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

from backend.services.visual_classifier import (
    VisualCandidateClassifier
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

    # --------------------------------------------------------
    # Validate PDF
    # --------------------------------------------------------

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

    try:

        page = document.load_page(
            0
        )

        # ----------------------------------------------------
        # Render page
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        print(
            "\nRunning OCR..."
        )

        ocr_result = (
            extract_text_with_data(
                image=image,
                language="eng",
                psm=3
            )
        )

        ocr_words = (
            ocr_result["words"]
        )

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

        visual_candidates = (
            detector.detect_contours(
                image
            )
        )

        print(
            f"Raw visual candidates: "
            f"{len(visual_candidates)}"
        )

        # ----------------------------------------------------
        # Visual merger
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Visual classification
        # ----------------------------------------------------

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

        print(
            f"Visual classifications: "
            f"{len(classifications)}"
        )

        # ----------------------------------------------------
        # Semantic grouping
        # ----------------------------------------------------

        print(
            "\nRunning semantic grouping..."
        )

        grouper = SemanticGrouper()

        result = (
            grouper.analyze_page(
                visual_regions=merged_regions,
                ocr_words=ocr_words,
                classifications=classifications
            )
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

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

            metadata = region.get(
                "metadata",
                {}
            )

            text = metadata.get(
                "text",
                ""
            )

            if text:

                print(
                    f"    text={text}"
                )

            parent_id = region.get(
                "parent_region_id",
                None
            )

            print(
                "    "
                f"parent={parent_id}"
            )

            # ------------------------------------------------
            # Classifier information
            # ------------------------------------------------

            classifier_confidence = (
                metadata.get(
                    "classifier_confidence",
                    0.0
                )
            )

            classifier_reason = (
                metadata.get(
                    "classifier_reason",
                    ""
                )
            )

            if classifier_confidence:

                print(
                    "    "
                    f"classifier_confidence="
                    f"{classifier_confidence:.2f}%"
                )

            if classifier_reason:

                print(
                    "    "
                    f"reason="
                    f"{classifier_reason}"
                )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

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

    finally:

        document.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()