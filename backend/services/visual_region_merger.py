from dataclasses import dataclass, asdict
from typing import List, Dict, Any


# ============================================================
# VISUAL REGION
# ============================================================

@dataclass
class VisualRegion:

    region_id: int

    x: int
    y: int
    width: int
    height: int

    area: int

    component_count: int

    source: str

    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# VISUAL REGION MERGER
# ============================================================

class VisualRegionMerger:

    """
    Conservative visual-region grouping.

    IMPORTANT:

    This class intentionally avoids recursive/chain merging.

    The previous implementation could turn many small nearby
    contours into one huge page-sized region.

    We now only merge components when BOTH:
        1. they overlap substantially, OR
        2. they are very close AND have similar vertical scale.

    The goal is information preservation, not aggressive
    diagram reconstruction.
    """

    def __init__(
        self,
        min_area: int = 1200,
        maximum_gap: int = 35,
        overlap_threshold: float = 0.20,
        size_ratio_limit: float = 3.0,
    ):

        self.min_area = min_area
        self.maximum_gap = maximum_gap
        self.overlap_threshold = overlap_threshold
        self.size_ratio_limit = size_ratio_limit

    # ========================================================
    # GEOMETRY
    # ========================================================

    @staticmethod
    def right(region):

        return (
            region["x"]
            + region["width"]
        )

    @staticmethod
    def bottom(region):

        return (
            region["y"]
            + region["height"]
        )

    @staticmethod
    def area(region):

        return (
            region["width"]
            * region["height"]
        )

    @staticmethod
    def center_x(region):

        return (
            region["x"]
            + region["width"] / 2
        )

    @staticmethod
    def center_y(region):

        return (
            region["y"]
            + region["height"] / 2
        )

    # ========================================================
    # IOU
    # ========================================================

    @staticmethod
    def intersection_over_union(
        a,
        b
    ):

        ax1 = a["x"]
        ay1 = a["y"]
        ax2 = a["x"] + a["width"]
        ay2 = a["y"] + a["height"]

        bx1 = b["x"]
        by1 = b["y"]
        bx2 = b["x"] + b["width"]
        by2 = b["y"] + b["height"]

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

        intersection = (
            width * height
        )

        union = (
            VisualRegionMerger.area(a)
            +
            VisualRegionMerger.area(b)
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection / union
        )

    # ========================================================
    # GAP
    # ========================================================

    @classmethod
    def horizontal_gap(
        cls,
        a,
        b
    ):

        if cls.right(a) < b["x"]:

            return (
                b["x"]
                - cls.right(a)
            )

        if cls.right(b) < a["x"]:

            return (
                a["x"]
                - cls.right(b)
            )

        return 0

    @classmethod
    def vertical_gap(
        cls,
        a,
        b
    ):

        if cls.bottom(a) < b["y"]:

            return (
                b["y"]
                - cls.bottom(a)
            )

        if cls.bottom(b) < a["y"]:

            return (
                a["y"]
                - cls.bottom(b)
            )

        return 0

    # ========================================================
    # SIZE SIMILARITY
    # ========================================================

    @classmethod
    def similar_size(
        cls,
        a,
        b
    ):

        width_ratio = (
            max(
                a["width"],
                b["width"]
            )
            /
            max(
                1,
                min(
                    a["width"],
                    b["width"]
                )
            )
        )

        height_ratio = (
            max(
                a["height"],
                b["height"]
            )
            /
            max(
                1,
                min(
                    a["height"],
                    b["height"]
                )
            )
        )

        return (
            width_ratio
            <= 3.0
            and
            height_ratio
            <= 3.0
        )

    # ========================================================
    # SHOULD MERGE
    # ========================================================

    def should_merge(
        self,
        a,
        b
    ):

        iou = self.intersection_over_union(
            a,
            b
        )

        # Strong overlap.
        if iou >= self.overlap_threshold:

            return True

        horizontal_gap = (
            self.horizontal_gap(
                a,
                b
            )
        )

        vertical_gap = (
            self.vertical_gap(
                a,
                b
            )
        )

        # Components must be close in BOTH dimensions.
        if (
            horizontal_gap
            <= self.maximum_gap
            and
            vertical_gap
            <= self.maximum_gap
            and
            self.similar_size(
                a,
                b
            )
        ):

            return True

        return False

    # ========================================================
    # MERGE TWO
    # ========================================================

    @staticmethod
    def merge_two(
        a,
        b
    ):

        x1 = min(
            a["x"],
            b["x"]
        )

        y1 = min(
            a["y"],
            b["y"]
        )

        x2 = max(
            a["x"] + a["width"],
            b["x"] + b["width"]
        )

        y2 = max(
            a["y"] + a["height"],
            b["y"] + b["height"]
        )

        return {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "component_count": (
                a.get(
                    "component_count",
                    1
                )
                +
                b.get(
                    "component_count",
                    1
                )
            ),
        }

    # ========================================================
    # ONE-PASS MERGING
    # ========================================================

    def merge_candidates(
        self,
        candidates: List[Dict[str, Any]]
    ):

        filtered = []

        for candidate in candidates:

            if (
                self.area(candidate)
                < self.min_area
            ):
                continue

            filtered.append(
                {
                    "x": candidate["x"],
                    "y": candidate["y"],
                    "width": candidate["width"],
                    "height": candidate["height"],
                    "component_count": 1,
                }
            )

        # Sort spatially.
        filtered.sort(
            key=lambda r: (
                r["y"],
                r["x"]
            )
        )

        result = []

        consumed = set()

        # ----------------------------------------------------
        # IMPORTANT:
        # Each original candidate is considered once.
        #
        # We DO NOT recursively merge the resulting region
        # with more candidates.
        # ----------------------------------------------------

        for i, current in enumerate(
            filtered
        ):

            if i in consumed:
                continue

            merged = current

            for j in range(
                i + 1,
                len(filtered)
            ):

                if j in consumed:
                    continue

                other = filtered[j]

                if self.should_merge(
                    merged,
                    other
                ):

                    merged = self.merge_two(
                        merged,
                        other
                    )

                    consumed.add(j)

            result.append(
                merged
            )

            consumed.add(i)

        return result

    # ========================================================
    # PUBLIC API
    # ========================================================

    def merge(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[VisualRegion]:

        merged = self.merge_candidates(
            candidates
        )

        regions = []

        for index, region in enumerate(
            merged,
            start=1
        ):

            regions.append(
                VisualRegion(

                    region_id=index,

                    x=region["x"],
                    y=region["y"],

                    width=region["width"],
                    height=region["height"],

                    area=(
                        region["width"]
                        *
                        region["height"]
                    ),

                    component_count=(
                        region[
                            "component_count"
                        ]
                    ),

                    source="opencv",

                    reason=(
                        "Conservative "
                        "visual grouping."
                    ),
                )
            )

        return regions