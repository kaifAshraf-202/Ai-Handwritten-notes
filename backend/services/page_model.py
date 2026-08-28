from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


# ============================================================
# PAGE MODEL
# ============================================================

@dataclass
class PageModel:
    """
    Unified representation of one analyzed PDF page.

    This class does NOT perform analysis.

    It only stores the results produced by the existing
    HandNote AI pipeline.
    """

    page_number: int

    image_width: int
    image_height: int

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    ocr_words: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_blocks: List[Dict[str, Any]] = field(
        default_factory=list
    )

    text_classifications: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # VISUAL PIPELINE
    # --------------------------------------------------------

    visual_regions: List[Dict[str, Any]] = field(
        default_factory=list
    )

    visual_classifications: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # SEMANTIC
    # --------------------------------------------------------

    semantic_regions: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    layout_blocks: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the complete page model into a dictionary.
        """

        return asdict(self)

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact summary of the analyzed page.
        """

        return {
            "page_number": self.page_number,
            "image_width": self.image_width,
            "image_height": self.image_height,

            "ocr_words": len(
                self.ocr_words
            ),

            "text_blocks": len(
                self.text_blocks
            ),

            "text_classifications": len(
                self.text_classifications
            ),

            "visual_regions": len(
                self.visual_regions
            ),

            "visual_classifications": len(
                self.visual_classifications
            ),

            "semantic_regions": len(
                self.semantic_regions
            ),

            "layout_blocks": len(
                self.layout_blocks
            ),
        }


# ============================================================
# OBJECT SERIALIZATION HELPERS
# ============================================================

def object_to_dict(
    value: Any
) -> Any:
    """
    Convert known project objects into dictionaries.

    Supports:

        - dataclasses
        - dictionaries
        - lists
        - tuples
        - primitive values

    This keeps the PageModel independent from the individual
    service implementations.
    """

    if value is None:
        return None

    # --------------------------------------------------------
    # Dataclass
    # --------------------------------------------------------

    if hasattr(
        value,
        "__dataclass_fields__"
    ):
        return asdict(
            value
        )

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        value,
        dict
    ):
        return {
            key: object_to_dict(
                item
            )
            for key, item in value.items()
        }

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(
        value,
        list
    ):
        return [
            object_to_dict(
                item
            )
            for item in value
        ]

    # --------------------------------------------------------
    # Tuple
    # --------------------------------------------------------

    if isinstance(
        value,
        tuple
    ):
        return [
            object_to_dict(
                item
            )
            for item in value
        ]

    # --------------------------------------------------------
    # Primitive
    # --------------------------------------------------------

    return value


# ============================================================
# PAGE MODEL BUILDER
# ============================================================

class PageModelBuilder:
    """
    Builds a PageModel from outputs of the existing services.

    IMPORTANT:

    This class is an orchestration/data-assembly layer.

    It does not replace:

        OCR
        RegionDetector
        VisualMerger
        VisualSplitter
        VisualClassifier
        SemanticGrouper
        TextBlockMerger
        ContentRegionClassifier
        LayoutAnalyzer
    """

    def __init__(
        self,
        page_number: int,
        image_width: int,
        image_height: int,
    ):

        self.page = PageModel(
            page_number=page_number,
            image_width=image_width,
            image_height=image_height,
        )

    # ========================================================
    # OCR
    # ========================================================

    def set_ocr_words(
        self,
        words: List[Dict[str, Any]]
    ):

        self.page.ocr_words = [
            object_to_dict(
                word
            )
            for word in (
                words or []
            )
        ]

        return self

    # ========================================================
    # TEXT
    # ========================================================

    def set_text_blocks(
        self,
        blocks: List[Any]
    ):

        self.page.text_blocks = [
            object_to_dict(
                block
            )
            for block in (
                blocks or []
            )
        ]

        return self

    # ========================================================

    def set_text_classifications(
        self,
        classifications: List[Any]
    ):

        self.page.text_classifications = [
            object_to_dict(
                item
            )
            for item in (
                classifications or []
            )
        ]

        return self

    # ========================================================
    # VISUAL
    # ========================================================

    def set_visual_regions(
        self,
        regions: List[Any]
    ):

        self.page.visual_regions = [
            object_to_dict(
                region
            )
            for region in (
                regions or []
            )
        ]

        return self

    # ========================================================

    def set_visual_classifications(
        self,
        classifications: List[Any]
    ):

        self.page.visual_classifications = [
            object_to_dict(
                item
            )
            for item in (
                classifications or []
            )
        ]

        return self

    # ========================================================
    # SEMANTIC
    # ========================================================

    def set_semantic_regions(
        self,
        regions: List[Any]
    ):

        self.page.semantic_regions = [
            object_to_dict(
                region
            )
            for region in (
                regions or []
            )
        ]

        return self

    # ========================================================
    # LAYOUT
    # ========================================================

    def set_layout_blocks(
        self,
        blocks: List[Any]
    ):

        self.page.layout_blocks = [
            object_to_dict(
                block
            )
            for block in (
                blocks or []
            )
        ]

        return self

    # ========================================================
    # METADATA
    # ========================================================

    def set_metadata(
        self,
        metadata: Dict[str, Any]
    ):

        self.page.metadata = dict(
            metadata or {}
        )

        return self

    # ========================================================
    # BUILD
    # ========================================================

    def build(self) -> PageModel:

        return self.page


# ============================================================
# FACTORY FUNCTION
# ============================================================

def build_page_model(
    page_number: int,
    image_width: int,
    image_height: int,
    ocr_words=None,
    text_blocks=None,
    text_classifications=None,
    visual_regions=None,
    visual_classifications=None,
    semantic_regions=None,
    layout_blocks=None,
    metadata=None,
) -> PageModel:
    """
    Convenience function for constructing a PageModel.
    """

    builder = PageModelBuilder(
        page_number=page_number,
        image_width=image_width,
        image_height=image_height,
    )

    builder.set_ocr_words(
        ocr_words
    )

    builder.set_text_blocks(
        text_blocks
    )

    builder.set_text_classifications(
        text_classifications
    )

    builder.set_visual_regions(
        visual_regions
    )

    builder.set_visual_classifications(
        visual_classifications
    )

    builder.set_semantic_regions(
        semantic_regions
    )

    builder.set_layout_blocks(
        layout_blocks
    )

    builder.set_metadata(
        metadata
    )

    return builder.build()