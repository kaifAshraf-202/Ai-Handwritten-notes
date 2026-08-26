from dataclasses import dataclass, asdict
from typing import List, Dict, Any


# ============================================================
# SEMANTIC REGION
# ============================================================

@dataclass
class SemanticRegion:
    """
    Represents a semantic region extracted from a page.

    A semantic region is the higher-level interpretation of
    a visual region, for example:

        - text
        - heading
        - handwriting
        - highlight
        - diagram
        - graphic
        - annotation
        - visual
    """

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
    Convert merged visual regions into semantic content regions.

    Important principle:

        Spatial proximity alone is NOT enough.

    The grouper uses:

        1. OCR overlap
        2. visual region information
        3. spatial information
        4. visual classification
        5. handwriting information
        6. highlight information
        7. diagram information
    """

    def __init__(
        self,
        text_overlap_threshold: float = 0.25,
        containment_threshold: float = 0.70,
        min_visual_width: int = 25,
        min_visual_height: int = 20,
        padding: int = 8,
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
        default=None,
    ):
        """
        Safely read a value from either a dictionary
        or an object.
        """

        if isinstance(obj, dict):

            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    # ========================================================
    # AREA
    # ========================================================

    @classmethod
    def area(
        cls,
        region,
    ):

        width = int(
            cls.get_value(
                region,
                "width",
                0,
            )
        )

        height = int(
            cls.get_value(
                region,
                "height",
                0,
            )
        )

        return width * height

    # ========================================================
    # BBOX
    # ========================================================

    @classmethod
    def bbox(
        cls,
        region,
    ):

        x = int(
            cls.get_value(
                region,
                "x",
                0,
            )
        )

        y = int(
            cls.get_value(
                region,
                "y",
                0,
            )
        )

        width = int(
            cls.get_value(
                region,
                "width",
                0,
            )
        )

        height = int(
            cls.get_value(
                region,
                "height",
                0,
            )
        )

        return (
            x,
            y,
            x + width,
            y + height,
        )

    # ========================================================
    # INTERSECTION
    # ========================================================

    @classmethod
    def intersection_area(
        cls,
        region_a,
        region_b,
    ):

        ax1, ay1, ax2, ay2 = cls.bbox(
            region_a
        )

        bx1, by1, bx2, by2 = cls.bbox(
            region_b
        )

        x1 = max(
            ax1,
            bx1,
        )

        y1 = max(
            ay1,
            by1,
        )

        x2 = min(
            ax2,
            bx2,
        )

        y2 = min(
            ay2,
            by2,
        )

        width = max(
            0,
            x2 - x1,
        )

        height = max(
            0,
            y2 - y1,
        )

        return width * height

    # ========================================================
    # IOU
    # ========================================================

    @classmethod
    def iou(
        cls,
        region_a,
        region_b,
    ):

        intersection = (
            cls.intersection_area(
                region_a,
                region_b,
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
            + area_b
            - intersection
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
        text_region,
    ):

        intersection = (
            cls.intersection_area(
                visual_region,
                text_region,
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
        inner,
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
    # CONVERT REGION TO DICT
    # ========================================================

    @staticmethod
    def to_dict(
        region,
    ):

        if isinstance(
            region,
            dict,
        ):

            return dict(
                region
            )

        if hasattr(
            region,
            "to_dict",
        ):

            return region.to_dict()

        return {
            "region_id": getattr(
                region,
                "region_id",
                0,
            ),

            "region_type": getattr(
                region,
                "region_type",
                "unknown",
            ),

            "x": getattr(
                region,
                "x",
                0,
            ),

            "y": getattr(
                region,
                "y",
                0,
            ),

            "width": getattr(
                region,
                "width",
                0,
            ),

            "height": getattr(
                region,
                "height",
                0,
            ),

            "confidence": getattr(
                region,
                "confidence",
                0.0,
            ),

            "source": getattr(
                region,
                "source",
                "unknown",
            ),
        }

    # ========================================================
    # GET OCR TEXT INSIDE REGION
    # ========================================================

    def collect_text(
        self,
        region,
        ocr_words,
    ):

        selected = []

        for word in ocr_words:

            overlap = (
                self.overlap_ratio(
                    region,
                    word,
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
                        "",
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
        text_words,
    ):

        if not text_words:

            return "visual"

        text = " ".join(
            text_words
        ).strip()

        if not text:

            return "visual"

        # ----------------------------------------------------
        # Heading-like text
        # ----------------------------------------------------

        if (
            len(text) <= 60
            and text.isupper()
        ):

            return "heading"

        # ----------------------------------------------------
        # Normal text
        # ----------------------------------------------------

        return "text"

    # ========================================================
    # VISUAL TYPE FALLBACK
    # ========================================================

    def classify_visual(
        self,
        region,
        ocr_words,
    ):

        region_dict = (
            self.to_dict(
                region
            )
        )

        region_type = str(
            region_dict.get(
                "region_type",
                "",
            )
        ).lower()

        # ----------------------------------------------------
        # Existing semantic classification
        # ----------------------------------------------------

        if region_type in {
            "handwriting",
            "highlight",
            "diagram",
            "graphic",
            "annotation",
        }:

            return region_type

        # ----------------------------------------------------
        # Collect OCR
        # ----------------------------------------------------

        text_words = (
            self.collect_text(
                region,
                ocr_words,
            )
        )

        if text_words:

            return self.classify_text(
                text_words
            )

        return "visual"

    # ========================================================
    # EXTRACT CLASSIFIER REGION ID
    # ========================================================

    def get_classification_region_id(
        self,
        classification,
    ):
        """
        Extract the region ID from a VisualClassification.

        Supports several possible representations so the
        semantic grouper remains compatible with the classifier.
        """

        region_id = self.get_value(
            classification,
            "region_id",
            None,
        )

        if region_id is not None:

            try:
                return int(
                    region_id
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        region = self.get_value(
            classification,
            "region",
            None,
        )

        if region is not None:

            region_id = self.get_value(
                region,
                "region_id",
                None,
            )

            if region_id is not None:

                try:
                    return int(
                        region_id
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        return 0

    # ========================================================
    # EXTRACT CLASSIFIER TYPE
    # ========================================================

    def get_classification_type(
        self,
        classification,
    ):
        """
        Extract semantic type from VisualClassification.

        Current classifier terminology includes:

            handwriting
            highlight
            diagram
            graphic
        """

        possible_keys = (
            "classification",
            "region_type",
            "label",
            "type",
        )

        for key in possible_keys:

            value = self.get_value(
                classification,
                key,
                None,
            )

            if value is None:
                continue

            value = str(
                value
            ).strip().lower()

            if value:

                return value

        return ""

    # ========================================================
    # EXTRACT CLASSIFIER CONFIDENCE
    # ========================================================

    def get_classification_confidence(
        self,
        classification,
    ):

        value = self.get_value(
            classification,
            "confidence",
            0.0,
        )

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    # ========================================================
    # EXTRACT CLASSIFIER REASON
    # ========================================================

    def get_classification_reason(
        self,
        classification,
    ):

        value = self.get_value(
            classification,
            "reason",
            "",
        )

        if value is None:

            return ""

        return str(
            value
        )

    # ========================================================
    # CREATE SEMANTIC REGION
    # ========================================================

    def create_region(
        self,
        region,
        region_type,
        region_id,
        parent_id,
        text_words=None,
        classifier_confidence=0.0,
        classifier_reason="",
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
                    0,
                )
            ),

            y=int(
                data.get(
                    "y",
                    0,
                )
            ),

            width=int(
                data.get(
                    "width",
                    0,
                )
            ),

            height=int(
                data.get(
                    "height",
                    0,
                )
            ),

            confidence=float(
                classifier_confidence
                if classifier_confidence > 0
                else data.get(
                    "confidence",
                    0.0,
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

                "text_words": (
                    text_words or []
                ),

                "original_source":
                    data.get(
                        "source",
                        "unknown",
                    ),

                "classifier_confidence":
                    classifier_confidence,

                "classifier_reason":
                    classifier_reason,
            },
        )

    # ========================================================
    # GROUP VISUAL REGIONS
    # ========================================================

    def group_visual_regions(
        self,
        visual_regions,
        ocr_words,
        classifications=None,
    ):
        """
        Convert merged visual regions into semantic regions.

        `classifications` should contain the output of:

            VisualCandidateClassifier.classify(
                image,
                visual_regions,
                ocr_words
            )

        If classifications are not provided, the method falls
        back to OCR/text-based classification.
        """

        semantic_regions = []

        next_id = 1

        # ----------------------------------------------------
        # Build classification lookup
        # ----------------------------------------------------

        classification_map = {}

        if classifications:

            for classification in classifications:

                parent_id = (
                    self.get_classification_region_id(
                        classification
                    )
                )

                if parent_id != 0:

                    classification_map[
                        parent_id
                    ] = classification

        # ----------------------------------------------------
        # Process merged visual regions
        # ----------------------------------------------------

        for region in visual_regions:

            region_dict = (
                self.to_dict(
                    region
                )
            )

            parent_id = int(
                region_dict.get(
                    "region_id",
                    0,
                )
            )

            width = int(
                region_dict.get(
                    "width",
                    0,
                )
            )

            height = int(
                region_dict.get(
                    "height",
                    0,
                )
            )

            if width < self.min_visual_width:

                continue

            if height < self.min_visual_height:

                continue

            # ------------------------------------------------
            # OCR inside this region
            # ------------------------------------------------

            text_words = (
                self.collect_text(
                    region,
                    ocr_words,
                )
            )

            # ------------------------------------------------
            # Get visual classification
            # ------------------------------------------------

            classification = (
                classification_map.get(
                    parent_id
                )
            )

            semantic_type = ""

            classifier_confidence = 0.0

            classifier_reason = ""

            if classification is not None:

                semantic_type = (
                    self.get_classification_type(
                        classification
                    )
                )

                classifier_confidence = (
                    self.get_classification_confidence(
                        classification
                    )
                )

                classifier_reason = (
                    self.get_classification_reason(
                        classification
                    )
                )

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            if not semantic_type:

                semantic_type = (
                    self.classify_visual(
                        region,
                        ocr_words,
                    )
                )

            # ------------------------------------------------
            # Create semantic region
            # ------------------------------------------------

            semantic_region = (
                self.create_region(
                    region=region,
                    region_type=semantic_type,
                    region_id=next_id,
                    parent_id=parent_id,
                    text_words=text_words,
                    classifier_confidence=(
                        classifier_confidence
                    ),
                    classifier_reason=(
                        classifier_reason
                    ),
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
        ocr_words,
        classifications=None,
    ):
        """
        Analyze a page and return semantic regions.

        Parameters:

            visual_regions:
                Merged visual regions.

            ocr_words:
                OCR word dictionaries.

            classifications:
                Optional VisualClassification objects.
        """

        semantic_regions = (
            self.group_visual_regions(
                visual_regions=visual_regions,
                ocr_words=ocr_words,
                classifications=classifications,
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
                    0,
                )
                + 1
            )

        return {

            "regions": [
                region.to_dict()
                for region in semantic_regions
            ],

            "counts": counts,

            "total": len(
                semantic_regions
            ),
        }