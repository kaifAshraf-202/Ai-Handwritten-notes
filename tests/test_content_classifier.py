from pathlib import Path
from io import BytesIO

import pymupdf
from PIL import Image

from backend.services.ocr import extract_ocr_data
from backend.services.text_block_merger import (
    TextBlockMerger
)
from backend.services.region_detector import (
    RegionDetector
)
from backend.services.content_region_classifier import (
    ContentRegionClassifier
)


PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

PAGE_NUMBER = 1
OCR_DPI = 200


def main():

    print("\n==========================================")
    print("     HANDNOTE AI - CONTENT CLASSIFIER")
    print("==========================================\n")

    document = pymupdf.open(
        PDF_PATH
    )

    try:

        page = document.load_page(
            PAGE_NUMBER - 1
        )

        # ----------------------------------------------------
        # Render page
        # ----------------------------------------------------

        pixmap = page.get_pixmap(
            dpi=OCR_DPI,
            alpha=False
        )

        image = Image.open(
            BytesIO(
                pixmap.tobytes("png")
            )
        ).convert("RGB")

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        print("Running OCR...")

        ocr_result = extract_ocr_data(
            image=image,
            language="eng",
            psm=3
        )

        # ----------------------------------------------------
        # Text blocks
        # ----------------------------------------------------

        merger = TextBlockMerger()

        text_blocks = merger.merge(
            ocr_result["words"]
        )

        # ----------------------------------------------------
        # Visual detection
        # ----------------------------------------------------

        detector = RegionDetector()

        analysis = detector.analyze_page(
            image=image,
            ocr_words=ocr_result["words"]
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        classifier = ContentRegionClassifier()

        regions = classifier.classify_page(
            text_blocks=text_blocks,
            visual_candidates=(
                analysis["visual_candidates"]
            )
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        reliable = [
            r for r in regions
            if r.region_type == "reliable_text"
        ]

        uncertain = [
            r for r in regions
            if r.region_type == "uncertain_text"
        ]

        visual = [
            r for r in regions
            if r.region_type == "visual_candidate"
        ]

        print(
            "\n=========================================="
        )

        print(
            "           CLASSIFICATION"
        )

        print(
            "==========================================\n"
        )

        print(
            f"Reliable text:     {len(reliable)}"
        )

        print(
            f"Uncertain text:    {len(uncertain)}"
        )

        print(
            f"Visual candidates: {len(visual)}"
        )

        # ----------------------------------------------------
        # Reliable text
        # ----------------------------------------------------

        print(
            "\n------------------------------------------"
        )

        print(
            "RELIABLE TEXT"
        )

        print(
            "------------------------------------------\n"
        )

        for region in reliable:

            print(
                f"[{region.region_id}] "
                f"{region.text}"
            )

            print(
                f"    confidence="
                f"{region.confidence:.2f}%"
            )

        # ----------------------------------------------------
        # Uncertain text
        # ----------------------------------------------------

        print(
            "\n------------------------------------------"
        )

        print(
            "UNCERTAIN / VISUAL OCR"
        )

        print(
            "------------------------------------------\n"
        )

        for region in uncertain:

            print(
                f"[{region.region_id}] "
                f"{region.text}"
            )

            print(
                f"    confidence="
                f"{region.confidence:.2f}%"
            )

            print(
                f"    bbox=("
                f"{region.x}, "
                f"{region.y}, "
                f"{region.width}, "
                f"{region.height}"
                f")"
            )

        print(
            "\n=========================================="
        )

        print(
            "     CLASSIFICATION COMPLETED"
        )

        print(
            "==========================================\n"
        )

    finally:

        document.close()


if __name__ == "__main__":
    main()