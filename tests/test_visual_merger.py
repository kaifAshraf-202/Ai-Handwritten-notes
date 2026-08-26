from pathlib import Path
from io import BytesIO

import pymupdf
from PIL import Image

from backend.services.ocr import extract_text_with_data
from backend.services.region_detector import RegionDetector
from backend.services.visual_region_merger import (
    VisualRegionMerger
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

PAGE_NUMBER = 1
OCR_DPI = 200


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("      HANDNOTE AI - VISUAL MERGER")
    print("==========================================\n")

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
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
            dpi=OCR_DPI,
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

        ocr_result = extract_text_with_data(
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
        # IMPORTANT
        #
        # Pass OCR words to RegionDetector.
        #
        # Previously we used:
        #
        #     ocr_words=[]
        #
        # which caused OpenCV to detect text as visual
        # content.
        # ----------------------------------------------------

        print(
            "\nRunning visual detection..."
        )

        detector = RegionDetector()

        analysis = detector.analyze_page(
            image=image,
            ocr_words=ocr_words
        )

        raw_candidates = analysis[
            "visual_candidates"
        ]

        print(
            f"Raw visual candidates: "
            f"{len(raw_candidates)}"
        )

        # ----------------------------------------------------
        # Visual merging
        # ----------------------------------------------------

        print(
            "\nMerging visual candidates..."
        )

        merger = VisualRegionMerger()

        regions = merger.merge(
            raw_candidates
        )

        print(
            f"Merged visual regions: "
            f"{len(regions)}"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "        MERGED VISUAL REGIONS"
        )

        print(
            "==========================================\n"
        )

        for region in regions:

            print(
                f"[{region.region_id}] "
                f"bbox=("
                f"{region.x}, "
                f"{region.y}, "
                f"{region.width}, "
                f"{region.height}"
                f")"
            )

            print(
                f"    area={region.area}"
            )

            print(
                f"    components="
                f"{region.component_count}"
            )

            print()

        print(
            "=========================================="
        )

        print(
            "       VISUAL MERGING COMPLETED"
        )

        print(
            "==========================================\n"
        )

    finally:

        document.close()


if __name__ == "__main__":
    main()



    