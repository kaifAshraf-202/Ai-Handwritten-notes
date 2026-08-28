from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple


# ============================================================
# LAYOUT BLOCK
# ============================================================

@dataclass
class LayoutBlock:
    """
    Logical page-level layout block.

    A layout block contains one or more closely related
    visual regions.
    """

    block_id: int

    x: int
    y: int
    width: int
    height: int

    region_ids: List[int]

    block_type: str = "visual"

    confidence: float = 0.0

    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def bbox(self):
        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )


# ============================================================
# LAYOUT ANALYZER
# ============================================================

class LayoutAnalyzer:
    """
    Analyze spatial relationships between already classified
    visual regions.

    This stage does NOT perform:

        OCR
        detection
        merging
        splitting
        classification

    Those stages already exist.

    This stage determines which regions belong to the
    same logical page-level block.
    """

    # --------------------------------------------------------
    # Region families
    # --------------------------------------------------------

    HIGHLIGHT_TYPES = {
        "highlight",
    }

    HANDWRITING_TYPES = {
        "handwriting",
        "annotation",
    }

    DIAGRAM_TYPES = {
        "diagram",
        "visual_section",
    }

    GRAPHIC_TYPES = {
        "graphic",
        "graphic_section",
    }

    TEXT_TYPES = {
        "text",
        "text_artifact",
    }

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    def __init__(
        self,
        horizontal_gap: int = 70,
        vertical_gap: int = 70,
        alignment_threshold: float = 0.45,
        overlap_threshold: float = 0.20,
        max_block_area_ratio: float = 0.45,
    ):

        # IMPORTANT:
        #
        # These are configuration values.
        #
        # Geometry helper methods are deliberately named
        # calculate_horizontal_gap() and
        # calculate_vertical_gap() to avoid a naming collision.

        self.horizontal_gap = horizontal_gap
        self.vertical_gap = vertical_gap

        self.alignment_threshold = (
            alignment_threshold
        )

        self.overlap_threshold = (
            overlap_threshold
        )

        self.max_block_area_ratio = (
            max_block_area_ratio
        )

    # ========================================================
    # BASIC REGION ACCESS
    # ========================================================

    @staticmethod
    def get_region_type(region) -> str:

        if isinstance(region, dict):

            value = (
                region.get("region_type")
                or region.get("type")
                or region.get("label")
                or region.get("classification")
            )

            if value:
                return str(value).lower()

            return "visual"

        value = (
            getattr(
                region,
                "region_type",
                None,
            )
            or getattr(
                region,
                "type",
                None,
            )
            or getattr(
                region,
                "label",
                None,
            )
            or getattr(
                region,
                "classification",
                None,
            )
        )

        if value:
            return str(value).lower()

        return "visual"

    # --------------------------------------------------------

    @staticmethod
    def get_region_id(region) -> int:

        if isinstance(region, dict):

            return int(
                region.get(
                    "region_id",
                    region.get(
                        "id",
                        0,
                    ),
                )
            )

        return int(
            getattr(
                region,
                "region_id",
                0,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def get_bbox(
        region,
    ) -> Tuple[int, int, int, int]:

        if isinstance(region, dict):

            return (
                int(
                    region.get(
                        "x",
                        0,
                    )
                ),
                int(
                    region.get(
                        "y",
                        0,
                    )
                ),
                int(
                    region.get(
                        "width",
                        0,
                    )
                ),
                int(
                    region.get(
                        "height",
                        0,
                    )
                ),
            )

        return (
            int(
                getattr(
                    region,
                    "x",
                    0,
                )
            ),
            int(
                getattr(
                    region,
                    "y",
                    0,
                )
            ),
            int(
                getattr(
                    region,
                    "width",
                    0,
                )
            ),
            int(
                getattr(
                    region,
                    "height",
                    0,
                )
            ),
        )

    # ========================================================
    # GEOMETRY
    # ========================================================

    @staticmethod
    def bbox_area(
        bbox,
    ) -> int:

        return (
            max(
                0,
                bbox[2],
            )
            *
            max(
                0,
                bbox[3],
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def intersection_area(
        a,
        b,
    ) -> int:

        ax1 = a[0]
        ay1 = a[1]
        ax2 = a[0] + a[2]
        ay2 = a[1] + a[3]

        bx1 = b[0]
        by1 = b[1]
        bx2 = b[0] + b[2]
        by2 = b[1] + b[3]

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

        if (
            x2 <= x1
            or
            y2 <= y1
        ):
            return 0

        return (
            (x2 - x1)
            *
            (y2 - y1)
        )

    # --------------------------------------------------------

    def overlap_ratio(
        self,
        a,
        b,
    ) -> float:

        intersection = (
            self.intersection_area(
                a,
                b,
            )
        )

        if intersection <= 0:
            return 0.0

        smaller = min(
            self.bbox_area(a),
            self.bbox_area(b),
        )

        if smaller <= 0:
            return 0.0

        return (
            intersection
            /
            float(smaller)
        )

    # --------------------------------------------------------
    # FIXED METHOD NAME
    # --------------------------------------------------------

    @staticmethod
    def calculate_horizontal_gap(
        a,
        b,
    ) -> int:

        ax2 = a[0] + a[2]
        bx2 = b[0] + b[2]

        if ax2 < b[0]:
            return b[0] - ax2

        if bx2 < a[0]:
            return a[0] - bx2

        return 0

    # --------------------------------------------------------
    # FIXED METHOD NAME
    # --------------------------------------------------------

    @staticmethod
    def calculate_vertical_gap(
        a,
        b,
    ) -> int:

        ay2 = a[1] + a[3]
        by2 = b[1] + b[3]

        if ay2 < b[1]:
            return b[1] - ay2

        if by2 < a[1]:
            return a[1] - by2

        return 0

    # --------------------------------------------------------

    @staticmethod
    def horizontal_alignment(
        a,
        b,
    ) -> float:

        ay2 = a[1] + a[3]
        by2 = b[1] + b[3]

        overlap = max(
            0,
            min(
                ay2,
                by2,
            )
            -
            max(
                a[1],
                b[1],
            ),
        )

        denominator = min(
            a[3],
            b[3],
        )

        if denominator <= 0:
            return 0.0

        return (
            overlap
            /
            float(denominator)
        )

    # --------------------------------------------------------

    @staticmethod
    def vertical_alignment(
        a,
        b,
    ) -> float:

        ax2 = a[0] + a[2]
        bx2 = b[0] + b[2]

        overlap = max(
            0,
            min(
                ax2,
                bx2,
            )
            -
            max(
                a[0],
                b[0],
            ),
        )

        denominator = min(
            a[2],
            b[2],
        )

        if denominator <= 0:
            return 0.0

        return (
            overlap
            /
            float(denominator)
        )

    # ========================================================
    # TYPE FAMILY
    # ========================================================

    def type_family(
        self,
        region,
    ) -> str:

        region_type = (
            self.get_region_type(
                region
            )
        )

        if region_type in self.HIGHLIGHT_TYPES:
            return "highlight"

        if region_type in self.HANDWRITING_TYPES:
            return "handwriting"

        if region_type in self.DIAGRAM_TYPES:
            return "diagram"

        if region_type in self.GRAPHIC_TYPES:
            return "graphic"

        if region_type in self.TEXT_TYPES:
            return "text"

        return "visual"

    # ========================================================
    # TYPE COMPATIBILITY
    # ========================================================

    def compatible_types(
        self,
        region_a,
        region_b,
    ) -> bool:

        family_a = self.type_family(
            region_a
        )

        family_b = self.type_family(
            region_b
        )

        # Same semantic family.
        if family_a == family_b:
            return True

        # Diagram + graphic can form one visual section.
        if {
            family_a,
            family_b,
        } <= {
            "diagram",
            "graphic",
        }:
            return True

        # Text stays separate from visual regions.
        if (
            family_a == "text"
            or
            family_b == "text"
        ):
            return False

        # Highlight stays independent.
        if (
            family_a == "highlight"
            or
            family_b == "highlight"
        ):
            return False

        # Handwriting stays independent from diagrams.
        if (
            family_a == "handwriting"
            or
            family_b == "handwriting"
        ):
            return False

        return False

    # ========================================================
    # CAN JOIN
    # ========================================================

    def can_join(
        self,
        region_a,
        region_b,
    ) -> bool:

        # ----------------------------------------------------
        # Semantic compatibility first.
        # ----------------------------------------------------

        if not self.compatible_types(
            region_a,
            region_b,
        ):
            return False

        bbox_a = self.get_bbox(
            region_a
        )

        bbox_b = self.get_bbox(
            region_b
        )

        # ----------------------------------------------------
        # Strong overlap.
        # ----------------------------------------------------

        if (
            self.overlap_ratio(
                bbox_a,
                bbox_b,
            )
            >= self.overlap_threshold
        ):
            return True

        # ----------------------------------------------------
        # IMPORTANT FIX:
        #
        # Call geometry methods with their new names.
        # ----------------------------------------------------

        h_gap = (
            self.calculate_horizontal_gap(
                bbox_a,
                bbox_b,
            )
        )

        v_gap = (
            self.calculate_vertical_gap(
                bbox_a,
                bbox_b,
            )
        )

        h_alignment = (
            self.horizontal_alignment(
                bbox_a,
                bbox_b,
            )
        )

        v_alignment = (
            self.vertical_alignment(
                bbox_a,
                bbox_b,
            )
        )

        # ----------------------------------------------------
        # Same row.
        # ----------------------------------------------------

        if (
            h_gap <= self.horizontal_gap
            and
            h_alignment >= self.alignment_threshold
        ):
            return True

        # ----------------------------------------------------
        # Same column.
        # ----------------------------------------------------

        if (
            v_gap <= self.vertical_gap
            and
            v_alignment >= self.alignment_threshold
        ):
            return True

        return False

    # ========================================================
    # GROUP REGIONS
    # ========================================================

    def group_regions(
        self,
        regions,
    ) -> List[List[Any]]:

        if not regions:
            return []

        groups = []

        unassigned = list(
            regions
        )

        while unassigned:

            seed = unassigned.pop(
                0
            )

            current_group = [
                seed
            ]

            changed = True

            while changed:

                changed = False

                remaining = []

                for candidate in unassigned:

                    should_add = False

                    for existing in current_group:

                        if self.can_join(
                            existing,
                            candidate,
                        ):
                            should_add = True
                            break

                    if should_add:

                        current_group.append(
                            candidate
                        )

                        changed = True

                    else:

                        remaining.append(
                            candidate
                        )

                unassigned = remaining

            groups.append(
                current_group
            )

        return groups

    # ========================================================
    # GROUP BBOX
    # ========================================================

    def group_bbox(
        self,
        group,
    ) -> Tuple[int, int, int, int]:

        boxes = [
            self.get_bbox(
                region
            )
            for region in group
        ]

        if not boxes:
            return (
                0,
                0,
                0,
                0,
            )

        x1 = min(
            box[0]
            for box in boxes
        )

        y1 = min(
            box[1]
            for box in boxes
        )

        x2 = max(
            box[0] + box[2]
            for box in boxes
        )

        y2 = max(
            box[1] + box[3]
            for box in boxes
        )

        return (
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        )

    # ========================================================
    # BLOCK TYPE
    # ========================================================

    def infer_block_type(
        self,
        group,
    ) -> str:

        families = [
            self.type_family(
                region
            )
            for region in group
        ]

        unique = set(
            families
        )

        if unique == {"highlight"}:
            return "highlight"

        if unique == {"handwriting"}:
            return "annotation"

        if unique <= {
            "diagram",
            "graphic",
        }:
            return "visual_section"

        if unique == {"text"}:
            return "text_section"

        return "visual"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def calculate_confidence(
        self,
        group,
    ) -> float:

        if not group:
            return 0.0

        # Single region = lower confidence because no
        # grouping relationship was established.
        if len(group) == 1:
            return 0.80

        type_families = {
            self.type_family(
                region
            )
            for region in group
        }

        if len(type_families) == 1:
            base = 0.88

        elif type_families <= {
            "diagram",
            "graphic",
        }:
            base = 0.84

        else:
            base = 0.75

        size_bonus = min(
            0.05,
            len(group) * 0.01,
        )

        return round(
            min(
                0.95,
                base + size_bonus,
            ),
            2,
        )

    # ========================================================
    # BUILD BLOCK
    # ========================================================

    def build_block(
        self,
        group,
        block_id: int,
    ) -> LayoutBlock:

        bbox = self.group_bbox(
            group
        )

        region_ids = [
            self.get_region_id(
                region
            )
            for region in group
        ]

        block_type = (
            self.infer_block_type(
                group
            )
        )

        confidence = (
            self.calculate_confidence(
                group
            )
        )

        return LayoutBlock(
            block_id=block_id,
            x=bbox[0],
            y=bbox[1],
            width=bbox[2],
            height=bbox[3],
            region_ids=region_ids,
            block_type=block_type,
            confidence=confidence,
            metadata={
                "region_count": len(
                    group
                ),
                "region_types": [
                    self.get_region_type(
                        region
                    )
                    for region in group
                ],
            },
        )

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        regions,
        ocr_words=None,
        image_width=None,
        image_height=None,
    ) -> List[LayoutBlock]:

        if not regions:
            return []

        groups = self.group_regions(
            regions
        )

        blocks = []

        for index, group in enumerate(
            groups,
            start=1,
        ):

            blocks.append(
                self.build_block(
                    group,
                    index,
                )
            )

        # ----------------------------------------------------
        # Reading order:
        # top-to-bottom, then left-to-right.
        # ----------------------------------------------------

        blocks.sort(
            key=lambda block: (
                block.y,
                block.x,
            )
        )

        # Reassign IDs after sorting.
        for index, block in enumerate(
            blocks,
            start=1,
        ):

            block.block_id = index

        return blocks

    # ========================================================
    # DICT API
    # ========================================================

    def analyze_to_dicts(
        self,
        regions,
        ocr_words=None,
        image_width=None,
        image_height=None,
    ) -> List[Dict[str, Any]]:

        blocks = self.analyze(
            regions=regions,
            ocr_words=ocr_words,
            image_width=image_width,
            image_height=image_height,
        )

        return [
            block.to_dict()
            for block in blocks
        ]