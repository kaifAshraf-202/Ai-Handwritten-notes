from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from PIL import Image

from backend.services.ocr import (
    extract_text_with_data,
)

from backend.services.region_detector import (
    RegionDetector,
)

from backend.services.visual_merger import (
    VisualMerger,
)

from backend.services.visual_splitter import (
    VisualRegionSplitter,
)

from backend.services.visual_classifier import (
    VisualCandidateClassifier,
)

from backend.services.semantic_grouper import (
    SemanticGrouper,
)

from backend.services.layout_analyzer import (
    LayoutAnalyzer,
)

from backend.services.text_block_merger import (
    TextBlockMerger,
)

from backend.services.content_region_classifier import (
    ContentRegionClassifier,
)

from backend.services.page_model import (
    PageModel,
    build_page_model,
    object_to_dict,
)


# ============================================================
# PIPELINE RESULT
# ============================================================

@dataclass
class PagePipelineResult:
    """
    Complete result produced by the HandNote AI page pipeline.

    The pipeline itself only orchestrates the individual
    analysis services.
    """

    page: PageModel

    raw_visual_regions: List[Any]
    merged_visual_regions: List[Any]
    split_visual_regions: List[Any]

    visual_classifications: List[Any]

    semantic_result: Dict[str, Any]

    layout_blocks: List[Any]

    text_blocks: List[Any]
    text_classifications: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the complete pipeline result to a dictionary.
        """

        return {
            "page": self.page.to_dict(),

            "raw_visual_regions": object_to_dict(
                self.raw_visual_regions
            ),

            "merged_visual_regions": object_to_dict(
                self.merged_visual_regions
            ),

            "split_visual_regions": object_to_dict(
                self.split_visual_regions
            ),

            "visual_classifications": object_to_dict(
                self.visual_classifications
            ),

            "semantic_result": object_to_dict(
                self.semantic_result
            ),

            "layout_blocks": object_to_dict(
                self.layout_blocks
            ),

            "text_blocks": object_to_dict(
                self.text_blocks
            ),

            "text_classifications": object_to_dict(
                self.text_classifications
            ),
        }


# ============================================================
# HANDNOTE PAGE PIPELINE
# ============================================================

class HandNotePagePipeline:
    """
    Main orchestration layer for HandNote AI.

    Pipeline:

        PDF page
            ↓
        Render
            ↓
        OCR
            ↓
        Region Detection
            ↓
        Visual Merger
            ↓
        Visual Splitter
            ↓
        Visual Classification
            ↓
        Semantic Grouping
            ↓
        Layout Analysis
            ↓
        Text Block Merging
            ↓
        Content Classification
            ↓
        Page Model
    """

    def __init__(
        self,
        region_detector: Optional[RegionDetector] = None,
        visual_merger: Optional[VisualMerger] = None,
        visual_splitter: Optional[VisualRegionSplitter] = None,
        visual_classifier: Optional[
            VisualCandidateClassifier
        ] = None,
        semantic_grouper: Optional[
            SemanticGrouper
        ] = None,
        layout_analyzer: Optional[
            LayoutAnalyzer
        ] = None,
        text_block_merger: Optional[
            TextBlockMerger
        ] = None,
        content_classifier: Optional[
            ContentRegionClassifier
        ] = None,
    ):
        """
        Allow services to be injected for testing.
        """

        self.region_detector = (
            region_detector
            or RegionDetector()
        )

        self.visual_merger = (
            visual_merger
            or VisualMerger()
        )

        self.visual_splitter = (
            visual_splitter
            or VisualRegionSplitter()
        )

        self.visual_classifier = (
            visual_classifier
            or VisualCandidateClassifier()
        )

        self.semantic_grouper = (
            semantic_grouper
            or SemanticGrouper()
        )

        self.layout_analyzer = (
            layout_analyzer
            or LayoutAnalyzer()
        )

        self.text_block_merger = (
            text_block_merger
            or TextBlockMerger()
        )

        self.content_classifier = (
            content_classifier
            or ContentRegionClassifier()
        )

    # ========================================================
    # PDF RENDERING
    # ========================================================

    @staticmethod
    def render_page(
        document,
        page_number: int,
        dpi: int = 200,
    ) -> Image.Image:
        """
        Render one PDF page into a PIL RGB image.

        Important:
        We do NOT use:

            pixmap.tobytes("png")

        because that performs PNG encoding before PIL
        decodes the image again.

        Instead we copy the raw Pixmap samples directly
        into a PIL image.
        """

        if dpi <= 0:
            raise ValueError(
                "dpi must be greater than 0"
            )

        page = document.load_page(
            page_number
        )

        pixmap = None

        try:
            pixmap = page.get_pixmap(
                dpi=dpi,
                alpha=False,
            )

            # ------------------------------------------------
            # PyMuPDF Pixmap → PIL
            # ------------------------------------------------

            if pixmap.n < 3:

                image = Image.frombytes(
                    "L",
                    (
                        pixmap.width,
                        pixmap.height,
                    ),
                    pixmap.samples,
                )

                return image.convert(
                    "RGB"
                )

            image = Image.frombytes(
                "RGB",
                (
                    pixmap.width,
                    pixmap.height,
                ),
                pixmap.samples,
            )

            return image

        finally:
            # Explicitly release the Pixmap as soon as
            # conversion is complete.
            pixmap = None

    # ========================================================
    # OCR
    # ========================================================

    def run_ocr(
        self,
        image: Image.Image,
        language: str = "eng",
        psm: int = 3,
    ) -> Dict[str, Any]:
        """
        Run the existing OCR service.
        """

        return extract_text_with_data(
            image=image,
            language=language,
            psm=psm,
        )

    # ========================================================
    # VISUAL PIPELINE
    # ========================================================

    def run_visual_pipeline(
        self,
        image: Image.Image,
        ocr_words: List[Dict[str, Any]],
    ):
        """
        Run the complete visual pipeline:

            detection
                ↓
            merging
                ↓
            classification
                ↓
            splitting
                ↓
            final classification
        """

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        raw_regions = (
            self.region_detector.detect_contours(
                image
            )
        )

        # ----------------------------------------------------
        # Merge
        # ----------------------------------------------------

        merged_regions = (
            self.visual_merger.merge(
                raw_regions
            )
        )

        # ----------------------------------------------------
        # Initial classification
        # ----------------------------------------------------

        merged_classifications = (
            self.visual_classifier.classify(
                image,
                merged_regions,
                ocr_words,
            )
        )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        split_regions = (
            self.visual_splitter.split(
                image,
                merged_regions,
                ocr_words=ocr_words,
                classifications=merged_classifications,
            )
        )

        # ----------------------------------------------------
        # Final classification
        # ----------------------------------------------------

        final_classifications = (
            self.visual_classifier.classify(
                image,
                split_regions,
                ocr_words,
            )
        )

        return (
            raw_regions,
            merged_regions,
            split_regions,
            final_classifications,
        )

    # ========================================================
    # SEMANTIC GROUPING
    # ========================================================

    def run_semantic_grouping(
        self,
        visual_regions,
        ocr_words,
        classifications,
    ) -> Dict[str, Any]:
        """
        Run semantic grouping.
        """

        return (
            self.semantic_grouper.analyze_page(
                visual_regions=visual_regions,
                ocr_words=ocr_words,
                classifications=classifications,
            )
        )

    # ========================================================
    # LAYOUT
    # ========================================================

    def run_layout_analysis(
        self,
        regions,
        ocr_words,
        image_width: int,
        image_height: int,
    ):
        """
        Run layout analysis.
        """

        return (
            self.layout_analyzer.analyze(
                regions=regions,
                ocr_words=ocr_words,
                image_width=image_width,
                image_height=image_height,
            )
        )

    # ========================================================
    # TEXT RECONSTRUCTION
    # ========================================================

    def run_text_reconstruction(
        self,
        ocr_words: List[Dict[str, Any]],
    ):
        """
        Merge OCR words into logical text blocks.
        """

        return (
            self.text_block_merger.merge(
                ocr_words
            )
        )

    # ========================================================
    # TEXT CLASSIFICATION
    # ========================================================

    def run_text_classification(
        self,
        text_blocks,
        visual_regions=None,
    ):
        """
        Classify reconstructed text blocks.

        ContentRegionClassifier expects visual candidates
        as dictionaries.
        """

        visual_candidates = [
            object_to_dict(region)
            for region in (
                visual_regions or []
            )
        ]

        return (
            self.content_classifier.classify_page(
                text_blocks=text_blocks,
                visual_candidates=visual_candidates,
            )
        )

    # ========================================================
    # BUILD PAGE MODEL
    # ========================================================

    def build_page(
        self,
        page_number: int,
        image: Image.Image,
        ocr_words,
        text_blocks,
        text_classifications,
        visual_regions,
        visual_classifications,
        semantic_result,
        layout_blocks,
        dpi: int = 200,
    ) -> PageModel:
        """
        Assemble all pipeline outputs into PageModel.
        """

        # ----------------------------------------------------
        # Semantic regions
        # ----------------------------------------------------

        if isinstance(
            semantic_result,
            dict,
        ):
            semantic_regions = (
                semantic_result.get(
                    "regions",
                    [],
                )
            )
        else:
            semantic_regions = []

        # ----------------------------------------------------
        # Build page model
        # ----------------------------------------------------

        page = build_page_model(
            page_number=page_number,

            image_width=image.width,

            image_height=image.height,

            ocr_words=ocr_words,

            text_blocks=text_blocks,

            text_classifications=(
                text_classifications
            ),

            visual_regions=visual_regions,

            visual_classifications=(
                visual_classifications
            ),

            semantic_regions=(
                semantic_regions
            ),

            layout_blocks=layout_blocks,

            metadata={
                "pipeline": "HandNotePagePipeline",
                "dpi": dpi,
            },
        )

        return page

    # ========================================================
    # ANALYZE IMAGE
    # ========================================================

    def analyze_image(
        self,
        image: Image.Image,
        page_number: int = 1,
        language: str = "eng",
        psm: int = 3,
        dpi: int = 200,
    ) -> PagePipelineResult:
        """
        Analyze an already-rendered page image.
        """

        if image is None:
            raise ValueError(
                "image cannot be None"
            )

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_result = self.run_ocr(
            image=image,
            language=language,
            psm=psm,
        )

        if not isinstance(
            ocr_result,
            dict,
        ):
            raise TypeError(
                "OCR service must return a dictionary"
            )

        ocr_words = (
            ocr_result.get(
                "words",
                [],
            )
        )

        # ----------------------------------------------------
        # Visual pipeline
        # ----------------------------------------------------

        (
            raw_visual_regions,
            merged_visual_regions,
            split_visual_regions,
            visual_classifications,
        ) = self.run_visual_pipeline(
            image=image,
            ocr_words=ocr_words,
        )

        # ----------------------------------------------------
        # Semantic grouping
        # ----------------------------------------------------

        semantic_result = (
            self.run_semantic_grouping(
                visual_regions=(
                    split_visual_regions
                ),
                ocr_words=ocr_words,
                classifications=(
                    visual_classifications
                ),
            )
        )

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        layout_blocks = (
            self.run_layout_analysis(
                regions=split_visual_regions,
                ocr_words=ocr_words,
                image_width=image.width,
                image_height=image.height,
            )
        )

        # ----------------------------------------------------
        # Text reconstruction
        # ----------------------------------------------------

        text_blocks = (
            self.run_text_reconstruction(
                ocr_words
            )
        )

        # ----------------------------------------------------
        # Content classification
        # ----------------------------------------------------

        text_classifications = (
            self.run_text_classification(
                text_blocks=text_blocks,
                visual_regions=(
                    split_visual_regions
                ),
            )
        )

        # ----------------------------------------------------
        # Page model
        # ----------------------------------------------------

        page = self.build_page(
            page_number=page_number,
            image=image,
            ocr_words=ocr_words,
            text_blocks=text_blocks,
            text_classifications=(
                text_classifications
            ),
            visual_regions=(
                split_visual_regions
            ),
            visual_classifications=(
                visual_classifications
            ),
            semantic_result=(
                semantic_result
            ),
            layout_blocks=layout_blocks,
            dpi=dpi,
        )

        return PagePipelineResult(
            page=page,

            raw_visual_regions=(
                raw_visual_regions
            ),

            merged_visual_regions=(
                merged_visual_regions
            ),

            split_visual_regions=(
                split_visual_regions
            ),

            visual_classifications=(
                visual_classifications
            ),

            semantic_result=(
                semantic_result
            ),

            layout_blocks=(
                layout_blocks
            ),

            text_blocks=(
                text_blocks
            ),

            text_classifications=(
                text_classifications
            ),
        )

    # ========================================================
    # ANALYZE PDF PAGE
    # ========================================================

    def analyze_pdf_page(
        self,
        pdf_path: Path,
        page_number: int = 0,
        dpi: int = 200,
        language: str = "eng",
        psm: int = 3,
    ) -> PagePipelineResult:
        """
        Analyze one page from a PDF.
        """

        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        document = fitz.open(
            pdf_path
        )

        try:

            if page_number < 0:
                raise ValueError(
                    "page_number cannot be negative"
                )

            if page_number >= len(document):
                raise IndexError(
                    f"page_number {page_number} "
                    f"out of range; PDF has "
                    f"{len(document)} page(s)"
                )

            image = self.render_page(
                document=document,
                page_number=page_number,
                dpi=dpi,
            )

            return self.analyze_image(
                image=image,
                page_number=page_number + 1,
                language=language,
                psm=psm,
                dpi=dpi,
            )

        finally:
            document.close()

    # ========================================================
    # ANALYZE COMPLETE PDF
    # ========================================================

    def analyze_pdf(
        self,
        pdf_path: Path,
        dpi: int = 200,
        language: str = "eng",
        psm: int = 3,
    ) -> List[PagePipelineResult]:
        """
        Analyze every page of a PDF.
        """

        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        document = fitz.open(
            pdf_path
        )

        results: List[
            PagePipelineResult
        ] = []

        try:

            for page_index in range(
                len(document)
            ):

                image = self.render_page(
                    document=document,
                    page_number=page_index,
                    dpi=dpi,
                )

                result = self.analyze_image(
                    image=image,
                    page_number=page_index + 1,
                    language=language,
                    psm=psm,
                    dpi=dpi,
                )

                results.append(
                    result
                )

                # Explicitly release the image reference
                # before processing the next page.
                image.close()

        finally:

            document.close()

        return results


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def analyze_pdf(
    pdf_path: Path,
    dpi: int = 200,
    language: str = "eng",
    psm: int = 3,
) -> List[PagePipelineResult]:
    """
    Convenience API for analyzing a complete PDF.
    """

    pipeline = (
        HandNotePagePipeline()
    )

    return pipeline.analyze_pdf(
        pdf_path=pdf_path,
        dpi=dpi,
        language=language,
        psm=psm,
    )