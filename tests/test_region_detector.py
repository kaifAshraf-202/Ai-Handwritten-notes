from pathlib import Path
from io import BytesIO
import json

import pymupdf
from PIL import Image

from backend.services.ocr import extract_ocr_data
from backend.services.region_detector import RegionDetector


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

TEST_PAGE = 1

OCR_DPI = 200


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("     HANDNOTE AI - REGION DETECTION")
    print("==========================================\n")

    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: "
            f"{PDF_PATH.resolve()}"
        )

    print(
        f"PDF: {PDF_PATH.name}"
    )

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    document = pymupdf.open(
        PDF_PATH
    )

    try:

        total_pages = (
            document.page_count
        )

        print(
            f"Total pages: {total_pages}"
        )

        # ----------------------------------------------------
        # Validate page
        # ----------------------------------------------------

        if (
            TEST_PAGE < 1
            or TEST_PAGE > total_pages
        ):

            raise ValueError(
                f"Page {TEST_PAGE} is outside "
                f"range 1-{total_pages}"
            )

        # ----------------------------------------------------
        # Load page
        # ----------------------------------------------------

        page = document.load_page(
            TEST_PAGE - 1
        )

        print(
            f"\nProcessing page: {TEST_PAGE}"
        )

        # ----------------------------------------------------
        # Render page
        # ----------------------------------------------------

        print(
            "Rendering page..."
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
            f"{image.width} x "
            f"{image.height}"
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

        print(
            f"OCR words: "
            f"{ocr_result['word_count']}"
        )

        print(
            f"OCR confidence: "
            f"{ocr_result['average_confidence']:.2f}%"
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
            ocr_words=ocr_result["words"]
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "             REGION SUMMARY"
        )

        print(
            "==========================================\n"
        )

        print(
            f"Text regions: "
            f"{analysis['counts']['text_regions']}"
        )

        print(
            f"Low-confidence regions: "
            f"{analysis['counts']['low_confidence_regions']}"
        )

        print(
            f"Visual candidates: "
            f"{analysis['counts']['visual_candidates']}"
        )

        # ----------------------------------------------------
        # Low confidence regions
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "       LOW CONFIDENCE REGIONS"
        )

        print(
            "==========================================\n"
        )

        for region in (
            analysis["low_confidence_regions"]
        ):

            print(
                f"confidence="
                f"{region['confidence']:.2f}% "
                f"bbox=("
                f"{region['x']}, "
                f"{region['y']}, "
                f"{region['width']}, "
                f"{region['height']}"
                f")"
            )

        # ----------------------------------------------------
        # Visual candidates
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "          VISUAL CANDIDATES"
        )

        print(
            "==========================================\n"
        )

        for region in (
            analysis["visual_candidates"][:30]
        ):

            print(
                f"bbox=("
                f"{region['x']}, "
                f"{region['y']}, "
                f"{region['width']}, "
                f"{region['height']}"
                f") "
                f"area="
                f"{region['width'] * region['height']}"
            )

        # ----------------------------------------------------
        # Save analysis
        # ----------------------------------------------------

        output_path = Path(
            "storage/processing/"
            "page_1_regions.json"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                analysis,
                file,
                indent=2,
                ensure_ascii=False
            )

        print(
            "\nAnalysis saved to:"
        )

        print(
            output_path
        )

        print(
            "\n=========================================="
        )

        print(
            "      REGION DETECTION COMPLETED"
        )

        print(
            "==========================================\n"
        )

    finally:

        document.close()


if __name__ == "__main__":
    main()