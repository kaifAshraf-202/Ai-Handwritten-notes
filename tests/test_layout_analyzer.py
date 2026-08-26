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

from backend.services.layout_analyzer import (
    LayoutAnalyzer
)


PDF_PATH = Path(
    "storage/uploads/test.pdf"
)


def main():

    print()
    print("=" * 52)
    print("        HANDNOTE AI - LAYOUT ANALYZER")
    print("=" * 52)

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

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

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
        psm=3,
        language="eng",
    )

    ocr_words = ocr_result["words"]

    print(
        f"OCR words: "
        f"{len(ocr_words)}"
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
        f"Merged regions: "
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
        ocr_words=ocr_words,
    )

    print(
        f"Split regions: "
        f"{len(split_regions)}"
    )

    # --------------------------------------------------------
    # LAYOUT ANALYSIS
    # --------------------------------------------------------

    print()
    print("Running layout analysis...")

    analyzer = LayoutAnalyzer()

    blocks = analyzer.analyze(
        regions=split_regions,
        ocr_words=ocr_words,
        image_width=image.width,
        image_height=image.height,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 52)
    print("             LAYOUT BLOCKS")
    print("=" * 52)

    for block in blocks:

        print()

        print(
            f"[B{block.block_id}] "
            f"{block.block_type.upper()}"
        )

        print(
            "    bbox=("
            f"{block.x}, "
            f"{block.y}, "
            f"{block.width}, "
            f"{block.height}"
            ")"
        )

        print(
            f"    regions="
            f"{block.region_ids}"
        )

        print(
            f"    confidence="
            f"{block.confidence:.2f}"
        )

        print(
            f"    region_count="
            f"{len(block.region_ids)}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 52)
    print("             LAYOUT SUMMARY")
    print("=" * 52)

    print(
        f"Split regions: "
        f"{len(split_regions)}"
    )

    print(
        f"Layout blocks: "
        f"{len(blocks)}"
    )

    print()
    print("=" * 52)
    print(
        "       LAYOUT ANALYSIS COMPLETED"
    )
    print("=" * 52)

    document.close()


if __name__ == "__main__":
    main()