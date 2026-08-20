from pathlib import Path
from io import BytesIO

import pymupdf

from PIL import Image

from backend.services.ocr import extract_ocr_data


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

    print("\n===================================")
    print("       HANDNOTE AI - OCR TEST")
    print("===================================\n")

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

        print(
            f"\nProcessing page: {TEST_PAGE}"
        )

        page = document.load_page(
            TEST_PAGE - 1
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

        result = extract_ocr_data(
            image=image,
            language="eng",
            psm=3
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        print(
            "\n==================================="
        )

        print(
            "             OCR RESULT"
        )

        print(
            "===================================\n"
        )

        print(
            result["text"]
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print(
            "\n==================================="
        )

        print(
            "          OCR STATISTICS"
        )

        print(
            "===================================\n"
        )

        print(
            f"Words detected: "
            f"{result['word_count']}"
        )

        print(
            f"Average confidence: "
            f"{result['average_confidence']:.2f}%"
        )

        print(
            f"PSM used: "
            f"{result['psm']}"
        )

        # ----------------------------------------------------
        # Low confidence words
        # ----------------------------------------------------

        low_confidence = [
            word
            for word in result["words"]
            if 0 <= word["confidence"] < 60
        ]

        print(
            f"Low-confidence words: "
            f"{len(low_confidence)}"
        )

        print(
            "\n==================================="
        )

        print(
            "     LOW CONFIDENCE REGIONS"
        )

        print(
            "===================================\n"
        )

        for word in low_confidence[:30]:

            print(
                f"{word['text']:<20} "
                f"{word['confidence']:>6.2f}% "
                f"position=("
                f"{word['left']}, "
                f"{word['top']}, "
                f"{word['width']}, "
                f"{word['height']}"
                f")"
            )

        print(
            "\n==================================="
        )

        print(
            "       OCR TEST COMPLETED"
        )

        print(
            "===================================\n"
        )

    finally:

        document.close()


if __name__ == "__main__":
    main()