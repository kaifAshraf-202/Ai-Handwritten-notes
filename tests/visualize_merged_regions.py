from pathlib import Path
from io import BytesIO

import pymupdf
import cv2
import numpy as np

from PIL import Image

from backend.services.ocr import extract_ocr_data
from backend.services.region_detector import RegionDetector
from backend.services.visual_region_merger import (
    VisualRegionMerger
)


PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

OUTPUT_PATH = Path(
    "storage/processing/"
    "page_1_merged_regions_preview.png"
)

PAGE_NUMBER = 1
DPI = 200


def main():

    print("\n==========================================")
    print(" HANDNOTE AI - MERGED REGION VISUALIZER")
    print("==========================================\n")

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

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        print(
            "Rendering page..."
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

        ocr_words = ocr_result[
            "words"
        ]

        print(
            f"OCR words: "
            f"{len(ocr_words)}"
        )

        # ----------------------------------------------------
        # Visual detection
        # ----------------------------------------------------

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
        # Merge
        # ----------------------------------------------------

        merger = VisualRegionMerger()

        regions = merger.merge(
            raw_candidates
        )

        print(
            f"Merged visual regions: "
            f"{len(regions)}"
        )

        # ----------------------------------------------------
        # Convert image
        # ----------------------------------------------------

        preview = np.array(
            image
        )

        preview = cv2.cvtColor(
            preview,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # Draw merged regions
        # ----------------------------------------------------

        for region in regions:

            x = region.x
            y = region.y

            width = region.width
            height = region.height

            cv2.rectangle(
                preview,
                (x, y),
                (
                    x + width,
                    y + height
                ),
                (0, 0, 255),
                3
            )

            label = (
                f"R{region.region_id}"
            )

            cv2.putText(
                preview,
                label,
                (
                    x,
                    max(
                        25,
                        y - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(OUTPUT_PATH),
            preview
        )

        print(
            "\n=========================================="
        )

        print(
            "       VISUALIZATION COMPLETED"
        )

        print(
            "==========================================\n"
        )

        print(
            "Red boxes = merged visual candidates"
        )

        print(
            "\nSaved to:"
        )

        print(
            OUTPUT_PATH
        )

    finally:

        document.close()


if __name__ == "__main__":
    main()