from pathlib import Path
from io import BytesIO

import pymupdf

from PIL import Image

from backend.services.ocr import extract_ocr_data
from backend.services.text_block_merger import (
    TextBlockMerger
)


PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

PAGE_NUMBER = 1
OCR_DPI = 200


def main():

    print("\n==========================================")
    print("       HANDNOTE AI - TEXT BLOCK TEST")
    print("==========================================\n")

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    document = pymupdf.open(
        PDF_PATH
    )

    try:

        page = document.load_page(
            PAGE_NUMBER - 1
        )

        print(
            f"Processing page: {PAGE_NUMBER}"
        )

        # ----------------------------------------------------
        # Render
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

        print(
            "Running OCR..."
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

        # ----------------------------------------------------
        # Merge text
        # ----------------------------------------------------

        merger = TextBlockMerger()

        blocks = merger.merge(
            ocr_result["words"]
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "             TEXT BLOCKS"
        )

        print(
            "==========================================\n"
        )

        for block in blocks:

            print(
                f"[{block.block_id}] "
                f"{block.text}"
            )

            print(
                f"    bbox=("
                f"{block.x}, "
                f"{block.y}, "
                f"{block.width}, "
                f"{block.height}"
                f")"
            )

            print(
                f"    confidence="
                f"{block.average_confidence:.2f}%"
            )

            print()

        print(
            "=========================================="
        )

        print(
            f"Total text blocks: {len(blocks)}"
        )

        print(
            "==========================================\n"
        )

    finally:

        document.close()


if __name__ == "__main__":
    main()