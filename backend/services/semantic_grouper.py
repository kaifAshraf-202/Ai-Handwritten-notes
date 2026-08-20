from dataclasses import dataclass, asdict
from typing import List, Dict, Any


# ============================================================
# SEMANTIC REGION
# ============================================================

@dataclass
class SemanticRegion:

    region_id: int

    region_type: str

    x: int
    y: int
    width: int
    height: int

    confidence: float

    source: str

    parent_region_id: int = 0

    text_overlap: float = 0.0

    visual_overlap: float = 0.0

    metadata: Dict[str, Any] = None

    def to_dict(self):

        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data

    @property
    def bbox(self):

        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )


# ============================================================
# SEMANTIC GROUPER
# ============================================================

class SemanticGrouper:

    """
    Convert broad visual regions into smaller semantic
    content regions.

    Important principle:

        Spatial proximity alone is NOT enough.

    We use:

        1. OCR overlap
        2. colour information
        3. visual region size
        4. spatial position
        5. visual classification
        6. handwriting/highlight information
    """

    def __init__(
        self,

        text_overlap_threshold=0.25,

        containment_threshold=0.70,

        min_visual_width=25,

        min_visual_height=20,

        padding=8,
    ):

        self.text_overlap_threshold = (
            text_overlap_threshold
        )

        self.containment_threshold = (
            containment_threshold
        )

        self.min_visual_width = (
            min_visual_width
        )

        self.min_visual_height = (
            min_visual_height
        )

        self.padding = padding

    # ========================================================
    # GENERIC ACCESS
    # ========================================================

    @staticmethod
    def get_value(
        obj,
        key,
        default=None
    ):

        if isinstance(obj, dict):

            return obj.get(
                key,
                default
            )

        return getattr(
            obj,
            key,
            default
        )

    # ========================================================
    # AREA
    # ========================================================

    @classmethod
    def area(
        cls,
        region
    ):

        width = int(
            cls.get_value(
                region,
                "width",
                0
            )
        )

        height = int(
            cls.get_value(
                region,
                "height",
                0
            )
        )

        return width * height

    # ========================================================
    # BBOX
    # ========================================================

    @classmethod
    def bbox(
        cls,
        region
    ):

        x = int(
            cls.get_value(
                region,
                "x",
                0
            )
        )

        y = int(
            cls.get_value(
                region,
                "y",
                0
            )
        )

        width = int(
            cls.get_value(
                region,
                "width",
                0
            )
        )

        height = int(
            cls.get_value(
                region,
                "height",
                0
            )
        )

        return (
            x,
            y,
            x + width,
            y + height
        )

    # ========================================================
    # INTERSECTION
    # ========================================================

    @classmethod
    def intersection_area(
        cls,
        region_a,
        region_b
    ):

        ax1, ay1, ax2, ay2 = cls.bbox(
            region_a
        )

        bx1, by1, bx2, by2 = cls.bbox(
            region_b
        )

        x1 = max(
            ax1,
            bx1
        )

        y1 = max(
            ay1,
            by1
        )

        x2 = min(
            ax2,
            bx2
        )

        y2 = min(
            ay2,
            by2
        )

        width = max(
            0,
            x2 - x1
        )

        height = max(
            0,
            y2 - y1
        )

        return width * height

    # ========================================================
    # IOU
    # ========================================================

    @classmethod
    def iou(
        cls,
        region_a,
        region_b
    ):

        intersection = (
            cls.intersection_area(
                region_a,
                region_b
            )
        )

        area_a = cls.area(
            region_a
        )

        area_b = cls.area(
            region_b
        )

        union = (
            area_a
            +
            area_b
            -
            intersection
        )

        if union <= 0:

            return 0.0

        return (
            intersection / union
        )

    # ========================================================
    # TEXT OVERLAP
    # ========================================================

    @classmethod
    def overlap_ratio(
        cls,
        visual_region,
        text_region
    ):

        intersection = (
            cls.intersection_area(
                visual_region,
                text_region
            )
        )

        visual_area = cls.area(
            visual_region
        )

        if visual_area <= 0:

            return 0.0

        return (
            intersection / visual_area
        )

    # ========================================================
    # CONTAINS
    # ========================================================

    @classmethod
    def contains(
        cls,
        outer,
        inner
    ):

        ox1, oy1, ox2, oy2 = cls.bbox(
            outer
        )

        ix1, iy1, ix2, iy2 = cls.bbox(
            inner
        )

        return (
            ix1 >= ox1
            and
            iy1 >= oy1
            and
            ix2 <= ox2
            and
            iy2 <= oy2
        )

    # ========================================================
    # CONVERT
    # ========================================================

    @staticmethod
    def to_dict(
        region
    ):

        if isinstance(
            region,
            dict
        ):

            return dict(
                region
            )

        if hasattr(
            region,
            "to_dict"
        ):

            return region.to_dict()

        return {

            "region_id":
                getattr(
                    region,
                    "region_id",
                    0
                ),

            "region_type":
                getattr(
                    region,
                    "region_type",
                    "unknown"
                ),

            "x":
                getattr(
                    region,
                    "x",
                    0
                ),

            "y":
                getattr(
                    region,
                    "y",
                    0
                ),

            "width":
                getattr(
                    region,
                    "width",
                    0
                ),

            "height":
                getattr(
                    region,
                    "height",
                    0
                ),

            "confidence":
                getattr(
                    region,
                    "confidence",
                    0.0
                ),

            "source":
                getattr(
                    region,
                    "source",
                    "unknown"
                ),
        }

    # ========================================================
    # GET OCR TEXT INSIDE REGION
    # ========================================================

    def collect_text(
        self,
        region,
        ocr_words
    ):

        selected = []

        for word in ocr_words:

            overlap = (
                self.overlap_ratio(
                    region,
                    word
                )
            )

            if (
                overlap
                >=
                self.text_overlap_threshold
            ):

                text = str(
                    self.get_value(
                        word,
                        "text",
                        ""
                    )
                ).strip()

                if text:

                    selected.append(
                        text
                    )

        return selected

    # ========================================================
    # TEXT REGION CLASSIFICATION
    # ========================================================

    def classify_text(
        self,
        text_words
    ):

        if not text_words:

            return "visual"

        text = " ".join(
            text_words
        ).strip()

        if not text:

            return "visual"

        # -----------------------------------------------
        # Heading-like text
        # -----------------------------------------------

        if (
            len(text) <= 60
            and text.isupper()
        ):

            return "heading"

        # -----------------------------------------------
        # Normal text
        # -----------------------------------------------

        return "text"

    # ========================================================
    # VISUAL TYPE
    # ========================================================

    def classify_visual(
        self,
        region,
        ocr_words
    ):

        region_dict = (
            self.to_dict(
                region
            )
        )

        region_type = str(
            region_dict.get(
                "region_type",
                ""
            )
        ).lower()

        # ------------------------------------------------
        # Existing semantic classification
        # ------------------------------------------------

        if region_type in {
            "handwriting",
            "highlight",
            "diagram",
            "graphic",
            "annotation",
        }:

            return region_type

        # ------------------------------------------------
        # Collect OCR
        # ------------------------------------------------

        text_words = (
            self.collect_text(
                region,
                ocr_words
            )
        )

        if text_words:

            return self.classify_text(
                text_words
            )

        return "visual"

    # ========================================================
    # CREATE SEMANTIC REGION
    # ========================================================

    def create_region(
        self,
        region,
        region_type,
        region_id,
        parent_id,
        text_words=None
    ):

        data = self.to_dict(
            region
        )

        return SemanticRegion(

            region_id=region_id,

            region_type=region_type,

            x=int(
                data.get(
                    "x",
                    0
                )
            ),

            y=int(
                data.get(
                    "y",
                    0
                )
            ),

            width=int(
                data.get(
                    "width",
                    0
                )
            ),

            height=int(
                data.get(
                    "height",
                    0
                )
            ),

            confidence=float(
                data.get(
                    "confidence",
                    0.0
                )
            ),

            source="semantic_grouping",

            parent_region_id=parent_id,

            text_overlap=0.0,

            visual_overlap=1.0,

            metadata={
                "text": (
                    " ".join(
                        text_words or []
                    )
                ),
                "original_source":
                    data.get(
                        "source",
                        "unknown"
                    ),
            },
        )

    # ========================================================
    # GROUP VISUAL REGIONS
    # ========================================================

    def group_visual_regions(
        self,
        visual_regions,
        ocr_words
    ):

        semantic_regions = []

        next_id = 1

        for region in visual_regions:

            region_dict = self.to_dict(
                region
            )

            width = int(
                region_dict.get(
                    "width",
                    0
                )
            )

            height = int(
                region_dict.get(
                    "height",
                    0
                )
            )

            if (
                width
                <
                self.min_visual_width
            ):

                continue

            if (
                height
                <
                self.min_visual_height
            ):

                continue

            # ------------------------------------------------
            # OCR inside visual region
            # ------------------------------------------------

            text_words = (
                self.collect_text(
                    region,
                    ocr_words
                )
            )

            # ------------------------------------------------
            # Determine semantic type
            # ------------------------------------------------

            region_type = (
                self.classify_visual(
                    region,
                    ocr_words
                )
            )

            # ------------------------------------------------
            # Create semantic region
            # ------------------------------------------------

            semantic_region = (
                self.create_region(
                    region=region,
                    region_type=region_type,
                    region_id=next_id,
                    parent_id=int(
                        region_dict.get(
                            "region_id",
                            0
                        )
                    ),
                    text_words=text_words
                )
            )

            semantic_regions.append(
                semantic_region
            )

            next_id += 1

        return semantic_regions

    # ========================================================
    # ANALYZE PAGE
    # ========================================================

    def analyze_page(
        self,
        visual_regions,
        ocr_words
    ):

        semantic_regions = (
            self.group_visual_regions(
                visual_regions,
                ocr_words
            )
        )

        counts = {}

        for region in semantic_regions:

            region_type = (
                region.region_type
            )

            counts[region_type] = (
                counts.get(
                    region_type,
                    0
                )
                + 1
            )

        return {

            "regions": [
                region.to_dict()
                for region
                in semantic_regions
            ],

            "counts": counts,

            "total": len(
                semantic_regions
            ),
        }