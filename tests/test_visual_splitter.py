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

from backend.services.visual_splitter import (
    VisualRegionSplitter
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
    print("       HANDNOTE AI - VISUAL SPLITTER")
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

        # ====================================================
        # OCR
        # ====================================================

        print()
        print("Running OCR...")

        ocr_result = extract_text_with_data(
            image=image,
            language="eng",
            psm=3,
        )

        ocr_words = (
            ocr_result["words"]
        )

        print(
            f"OCR words: "
            f"{len(ocr_words)}"
        )

        # ====================================================
        # REGION DETECTION
        # ====================================================

        print()
        print("Running region detection...")

        detector = RegionDetector()

        raw_regions = (
            detector.detect_contours(
                image
            )
        )

        print(
            f"Raw visual candidates: "
            f"{len(raw_regions)}"
        )

        # ====================================================
        # VISUAL MERGING
        # ====================================================

        print()
        print("Running visual merger...")

        merger = VisualMerger()

        merged_regions = (
            merger.merge(
                raw_regions
            )
        )

        print(
            f"Merged visual regions: "
            f"{len(merged_regions)}"
        )

        # ====================================================
        # VISUAL SPLITTING
        # ====================================================

        print()
        print("Running visual splitter...")

        splitter = VisualRegionSplitter()

        # IMPORTANT:
        #
        # Pass OCR words to the splitter.
        #
        # This allows the splitter to avoid breaking
        # regions that are actually OCR/text dominated.
        #
        split_regions = splitter.split(
            image=image,
            regions=merged_regions,
            ocr_words=ocr_words,
        )

        print(
            f"Split visual regions: "
            f"{len(split_regions)}"
        )

        # ====================================================
        # RESULTS
        # ====================================================

        print()
        print("=" * 48)
        print("          SPLIT REGIONS")
        print("=" * 48)

        for region in split_regions:

            print()

            print(
                f"[R{region.region_id}]"
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
                f"    source_region="
                f"{region.source_region_id}"
            )

            print(
                f"    components="
                f"{region.component_count}"
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 48)
        print("          SPLITTER SUMMARY")
        print("=" * 48)

        print(
            f"Raw candidates:   "
            f"{len(raw_regions)}"
        )

        print(
            f"Merged regions:   "
            f"{len(merged_regions)}"
        )

        print(
            f"Split regions:    "
            f"{len(split_regions)}"
        )

        # ----------------------------------------------------
        # Split delta
        # ----------------------------------------------------

        difference = (
            len(split_regions)
            -
            len(merged_regions)
        )

        if difference > 0:

            print(
                f"Additional regions: "
                f"+{difference}"
            )

        elif difference < 0:

            print(
                f"Reduced regions:     "
                f"{difference}"
            )

        else:

            print(
                "Additional regions: "
                "0"
            )

        # ====================================================
        # COMPLETED
        # ====================================================

        print()
        print("=" * 48)
        print(
            "   VISUAL SPLITTING COMPLETED"
        )
        print("=" * 48)

    finally:

        document.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()