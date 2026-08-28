from pathlib import Path
from io import BytesIO

import fitz
from PIL import Image

from backend.services.ocr import extract_text_with_data
from backend.services.text_block_merger import TextBlockMerger
from backend.services.content_region_classifier import ContentRegionClassifier
from backend.services.region_detector import RegionDetector
from backend.services.visual_merger import VisualMerger
from backend.services.visual_splitter import VisualRegionSplitter
from backend.services.visual_classifier import VisualCandidateClassifier
from backend.services.semantic_grouper import SemanticGrouper
from backend.services.layout_analyzer import LayoutAnalyzer
from backend.services.page_model import PageModel
from backend.services.page_renderer import PageRenderer


# ============================================================
# PATH
# ============================================================

PDF_PATH = Path(
    "storage/uploads/test.pdf"
)

OUTPUT_PATH = Path(
    "storage/output/rendered_page_1.png"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 56)
    print("          HANDNOTE AI - PAGE RENDERER")
    print("=" * 56)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # ========================================================
    # OPEN PDF
    # ========================================================

    document = fitz.open(
        PDF_PATH
    )

    try:

        page = document.load_page(
            0
        )

        # ====================================================
        # SOURCE RENDERING
        # ====================================================

        print()
        print("Rendering source page...")

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

        ocr_words = ocr_result[
            "words"
        ]

        print(
            f"OCR words: "
            f"{len(ocr_words)}"
        )

        # ====================================================
        # TEXT BLOCK MERGING
        # ====================================================

        print()
        print("Merging text blocks...")

        text_merger = TextBlockMerger()

        text_blocks = text_merger.merge(
            ocr_words
        )

        print(
            f"Text blocks: "
            f"{len(text_blocks)}"
        )

        # ====================================================
        # TEXT CLASSIFICATION
        # ====================================================

        print()
        print("Classifying text...")

        content_classifier = (
            ContentRegionClassifier()
        )

        text_classifications = []

        for block in text_blocks:

            classification = (
                content_classifier.classify_text_block(
                    block
                )
            )

            text_classifications.append(
                classification
            )

        print(
            f"Text classifications: "
            f"{len(text_classifications)}"
        )

        # ====================================================
        # VISUAL DETECTION
        # ====================================================

        print()
        print("Detecting visual regions...")

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
        print("Merging visual regions...")

        merger = VisualMerger()

        merged_regions = merger.merge(
            raw_regions
        )

        print(
            f"Merged regions: "
            f"{len(merged_regions)}"
        )

        # ====================================================
        # VISUAL SPLITTING
        # ====================================================

        print()
        print("Splitting visual regions...")

        splitter = VisualRegionSplitter()

        split_regions = splitter.split(
            image=image,
            regions=merged_regions,
            ocr_words=ocr_words,
        )

        print(
            f"Split regions: "
            f"{len(split_regions)}"
        )

        # ====================================================
        # VISUAL CLASSIFICATION
        # ====================================================

        print()
        print("Classifying visual regions...")

        visual_classifier = (
            VisualCandidateClassifier()
        )

        visual_classifications = (
            visual_classifier.classify(
                image,
                split_regions,
                ocr_words,
            )
        )

        print(
            f"Visual classifications: "
            f"{len(visual_classifications)}"
        )

        # ====================================================
        # SEMANTIC GROUPING
        # ====================================================

        print()
        print("Running semantic grouping...")

        grouper = SemanticGrouper()

        semantic_result = (
            grouper.analyze_page(
                visual_regions=split_regions,
                ocr_words=ocr_words,
                classifications=visual_classifications,
            )
        )

        semantic_regions = (
            semantic_result.get(
                "regions",
                []
            )
        )

        print(
            f"Semantic regions: "
            f"{len(semantic_regions)}"
        )

        # ====================================================
        # LAYOUT ANALYSIS
        # ====================================================

        print()
        print("Running layout analysis...")

        layout_analyzer = LayoutAnalyzer()

        layout_blocks = (
            layout_analyzer.analyze(
                regions=split_regions,
                ocr_words=ocr_words,
                image_width=image.width,
                image_height=image.height,
            )
        )

        print(
            f"Layout blocks: "
            f"{len(layout_blocks)}"
        )

        # ====================================================
        # PAGE MODEL
        # ====================================================

        print()
        print("Building PageModel...")

        page_model = PageModel(
            page_number=1,
            image_width=image.width,
            image_height=image.height,
            ocr_words=ocr_words,
            text_blocks=text_blocks,
            text_classifications=text_classifications,
            visual_regions=split_regions,
            visual_classifications=visual_classifications,
            semantic_regions=semantic_regions,
            layout_blocks=layout_blocks,
        )

        print(
            "PageModel: OK"
        )

        # ====================================================
        # PAGE RENDERER
        # ====================================================

        print()
        print("Rendering reconstructed page...")

        renderer = PageRenderer(
            debug=False,
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # render_to_file() is the existing renderer API.
        #
        # Signature:
        #   render_to_file(
        #       page_model,
        #       source_image,
        #       output_path
        #   )
        #
        # It performs the rendering and saves the image.

        rendered = renderer.render_to_file(
            page_model=page_model,
            source_image=image,
            output_path=OUTPUT_PATH,
        )

        print()
        print(
            f"Output saved to: "
            f"{OUTPUT_PATH}"
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 56)
        print("          RENDERER SUMMARY")
        print("=" * 56)

        print(
            f"Source size:     "
            f"{image.size}"
        )

        print(
            f"Rendered size:   "
            f"{rendered.size}"
        )

        print(
            f"Text blocks:     "
            f"{len(text_blocks)}"
        )

        print(
            f"Visual regions:  "
            f"{len(split_regions)}"
        )

        print(
            f"Semantic regions:"
            f" {len(semantic_regions)}"
        )

        print(
            f"Layout blocks:   "
            f"{len(layout_blocks)}"
        )

        print()
        print("=" * 56)
        print(
            "       PAGE RENDERING COMPLETED"
        )
        print("=" * 56)

    finally:

        document.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()