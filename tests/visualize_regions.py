from pathlib import Path
from io import BytesIO

import pymupdf
import cv2
import numpy as np

from PIL import Image

from backend.services.ocr import extract_ocr_data
from backend.services.region_detector import RegionDetector


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

OUTPUT_PATH = Path(
    "storage/processing/page_1_regions_preview.png"
)

PAGE_NUMBER = 1
OCR_DPI = 200


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("   HANDNOTE AI - REGION VISUALIZATION")
    print("==========================================\n")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    document = pymupdf.open(PDF_PATH)

    try:

        # ----------------------------------------------------
        # Load page
        # ----------------------------------------------------

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

        print("Running OCR...")

        ocr_result = extract_ocr_data(
            image=image,
            language="eng",
            psm=3
        )

        # ----------------------------------------------------
        # Region detection
        # ----------------------------------------------------

        detector = RegionDetector()

        analysis = detector.analyze_page(
            image=image,
            ocr_words=ocr_result["words"]
        )

        # ----------------------------------------------------
        # Convert PIL → OpenCV
        # ----------------------------------------------------

        preview = np.array(image)

        preview = cv2.cvtColor(
            preview,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # Draw visual candidates
        # ----------------------------------------------------

        for index, region in enumerate(
            analysis["visual_candidates"],
            start=1
        ):

            x = region["x"]
            y = region["y"]
            w = region["width"]
            h = region["height"]

            cv2.rectangle(
                preview,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                preview,
                str(index),
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Draw low-confidence OCR regions
        # ----------------------------------------------------

        for region in (
            analysis["low_confidence_regions"]
        ):

            x = region["x"]
            y = region["y"]
            w = region["width"]
            h = region["height"]

            cv2.rectangle(
                preview,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
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

        print("\n==========================================")
        print("            VISUALIZATION")
        print("==========================================\n")

        print(
            f"Visual candidates: "
            f"{len(analysis['visual_candidates'])}"
        )

        print(
            f"Low-confidence regions: "
            f"{len(analysis['low_confidence_regions'])}"
        )

        print("\nSaved preview to:")

        print(
            OUTPUT_PATH
        )

        print("\nLegend:")
        print(
            "RED   = OpenCV visual candidates"
        )
        print(
            "BLUE  = Low-confidence OCR regions"
        )

        print("\n==========================================")
        print("       VISUALIZATION COMPLETED")
        print("==========================================\n")

    finally:

        document.close()


if __name__ == "__main__":
    main()