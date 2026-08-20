from pathlib import Path
from io import BytesIO

import pymupdf
from PIL import Image

from backend.services.ocr import extract_ocr_data


PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

PAGE_NUMBER = 1
OCR_DPI = 200


def right(word):
    return (
        int(word["left"])
        + int(word["width"])
    )


def center_y(word):
    return (
        int(word["top"])
        + int(word["height"]) / 2
    )


def main():

    print("\n==========================================")
    print("       OCR LAYOUT DIAGNOSTIC")
    print("==========================================\n")

    document = pymupdf.open(PDF_PATH)

    try:

        page = document.load_page(
            PAGE_NUMBER - 1
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

        result = extract_ocr_data(
            image=image,
            language="eng",
            psm=3
        )

        words = result["words"]

        # ----------------------------------------------------
        # Group by Tesseract line
        # ----------------------------------------------------

        lines = {}

        for word in words:

            key = (
                word["block_num"],
                word["par_num"],
                word["line_num"],
            )

            lines.setdefault(
                key,
                []
            ).append(word)

        # ----------------------------------------------------
        # Print each line and gaps
        # ----------------------------------------------------

        line_number = 1

        for key, line_words in lines.items():

            line_words.sort(
                key=lambda word:
                word["left"]
            )

            print(
                f"\nLINE {line_number}"
            )

            print(
                f"Tesseract "
                f"block={key[0]} "
                f"paragraph={key[1]} "
                f"line={key[2]}"
            )

            print("-" * 70)

            previous = None

            for word in line_words:

                if previous is not None:

                    gap = (
                        word["left"]
                        - right(previous)
                    )

                    print(
                        f"GAP: {gap:4d}px   "
                        f"{previous['text']!r} "
                        f"→ "
                        f"{word['text']!r}"
                    )

                print(
                    f"WORD: "
                    f"{word['text']!r:<30} "
                    f"x={word['left']:<5} "
                    f"y={word['top']:<5} "
                    f"w={word['width']:<5} "
                    f"h={word['height']:<5} "
                    f"conf={word['confidence']:.1f}"
                )

                previous = word

            line_number += 1

    finally:

        document.close()


if __name__ == "__main__":
    main()