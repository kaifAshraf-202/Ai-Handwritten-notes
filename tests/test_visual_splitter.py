from pathlib import Path
from io import BytesIO

import fitz
from PIL import Image

from backend.services.ocr import extract_text_with_data
from backend.services.region_detector import RegionDetector
from backend.services.visual_merger import VisualMerger
from backend.services.visual_splitter import VisualRegionSplitter


PDF_PATH = Path(
    "storage/uploads/test.pdf"
)


def main():

    print()
    print("=" * 48)
    print("       HANDNOTE AI - VISUAL SPLITTER")
    print("=" * 48)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    document = fitz.open(
        PDF_PATH
    )

    page = document.load_page(
        0
    )

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

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print()
    print("Running OCR...")

    ocr_result = extract_text_with_data(
        image=image,
        language="eng",
        psm=3,
    )

    ocr_words = ocr_result["words"]

    print(
        f"OCR words: {len(ocr_words)}"
    )

    # --------------------------------------------------------
    # REGION DETECTION
    # --------------------------------------------------------

    print()
    print("Running region detection...")

    detector = RegionDetector()

    raw_regions = detector.detect_contours(
        image
    )

    print(
        f"Raw visual candidates: "
        f"{len(raw_regions)}"
    )

    # --------------------------------------------------------
    # MERGING
    # --------------------------------------------------------

    print()
    print("Running visual merger...")

    merger = VisualMerger()

    merged_regions = merger.merge(
        raw_regions
    )

    print(
        f"Merged visual regions: "
        f"{len(merged_regions)}"
    )

    # --------------------------------------------------------
    # SPLITTING
    # --------------------------------------------------------

    print()
    print("Running visual splitter...")

    splitter = VisualRegionSplitter()

    split_regions = splitter.split(
        image,
        merged_regions,
    )

    print(
        f"Split visual regions: "
        f"{len(split_regions)}"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

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

    print()
    print("=" * 48)
    print(
        "   VISUAL SPLITTING COMPLETED"
    )
    print("=" * 48)

    document.close()


if __name__ == "__main__":
    main()