from dataclasses import dataclass, asdict
from typing import List, Dict, Any


# ============================================================
# MERGED VISUAL REGION
# ============================================================

@dataclass
class MergedVisualRegion:

    region_id: int

    x: int
    y: int
    width: int
    height: int

    area: int

    component_count: int

    source_regions: List[Dict[str, Any]]

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
# VISUAL MERGER
# ============================================================

class VisualMerger:

    """
    Conservative visual-region merger.

    IMPORTANT:

    The merger should NOT combine an entire page into
    one region simply because several small contours
    happen to be close to each other.

    We therefore use:

        1. Strong overlap
        2. Small spatial gap
        3. Similar region scale
        4. Protection against giant-region swallowing
    """

    def __init__(
        self,

        iou_threshold: float = 0.10,

        proximity_threshold: int = 25,

        padding: int = 8,

        max_gap: int = 35,

        max_size_ratio: float = 4.0,

    ):

        self.iou_threshold = iou_threshold

        self.proximity_threshold = (
            proximity_threshold
        )

        self.padding = padding

        self.max_gap = max_gap

        self.max_size_ratio = (
            max_size_ratio
        )

    # ========================================================
    # GENERIC ACCESS
    # ========================================================

    @staticmethod
    def get_value(
        region,
        key: str,
        default=0
    ):

        if isinstance(
            region,
            dict
        ):

            return region.get(
                key,
                default
            )

        return getattr(
            region,
            key,
            default
        )

    # ========================================================
    # CONVERT TO DICT
    # ========================================================

    @staticmethod
    def region_to_dict(
        region
    ) -> Dict[str, Any]:

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
                    "visual_candidate"
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
    # IOU
    # ========================================================

    @classmethod
    def calculate_iou(
        cls,
        region_a,
        region_b
    ) -> float:

        ax1 = int(
            cls.get_value(
                region_a,
                "x"
            )
        )

        ay1 = int(
            cls.get_value(
                region_a,
                "y"
            )
        )

        aw = int(
            cls.get_value(
                region_a,
                "width"
            )
        )

        ah = int(
            cls.get_value(
                region_a,
                "height"
            )
        )

        bx1 = int(
            cls.get_value(
                region_b,
                "x"
            )
        )

        by1 = int(
            cls.get_value(
                region_b,
                "y"
            )
        )

        bw = int(
            cls.get_value(
                region_b,
                "width"
            )
        )

        bh = int(
            cls.get_value(
                region_b,
                "height"
            )
        )

        ax2 = ax1 + aw
        ay2 = ay1 + ah

        bx2 = bx1 + bw
        by2 = by1 + bh

        ix1 = max(
            ax1,
            bx1
        )

        iy1 = max(
            ay1,
            by1
        )

        ix2 = min(
            ax2,
            bx2
        )

        iy2 = min(
            ay2,
            by2
        )

        iw = max(
            0,
            ix2 - ix1
        )

        ih = max(
            0,
            iy2 - iy1
        )

        intersection = (
            iw * ih
        )

        area_a = (
            aw * ah
        )

        area_b = (
            bw * bh
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
    # BOUNDING BOX GAP
    # ========================================================

    @classmethod
    def calculate_gap(
        cls,
        region_a,
        region_b
    ):

        ax1 = int(
            cls.get_value(
                region_a,
                "x"
            )
        )

        ay1 = int(
            cls.get_value(
                region_a,
                "y"
            )
        )

        ax2 = (
            ax1
            +
            int(
                cls.get_value(
                    region_a,
                    "width"
                )
            )
        )

        ay2 = (
            ay1
            +
            int(
                cls.get_value(
                    region_a,
                    "height"
                )
            )
        )

        bx1 = int(
            cls.get_value(
                region_b,
                "x"
            )
        )

        by1 = int(
            cls.get_value(
                region_b,
                "y"
            )
        )

        bx2 = (
            bx1
            +
            int(
                cls.get_value(
                    region_b,
                    "width"
                )
            )
        )

        by2 = (
            by1
            +
            int(
                cls.get_value(
                    region_b,
                    "height"
                )
            )
        )

        horizontal_gap = max(
            0,
            bx1 - ax2,
            ax1 - bx2
        )

        vertical_gap = max(
            0,
            by1 - ay2,
            ay1 - by2
        )

        return (
            horizontal_gap,
            vertical_gap
        )

    # ========================================================
    # SIZE SIMILARITY
    # ========================================================

    @classmethod
    def size_ratio(
        cls,
        region_a,
        region_b
    ):

        area_a = max(
            1,
            cls.area(
                region_a
            )
        )

        area_b = max(
            1,
            cls.area(
                region_b
            )
        )

        return max(
            area_a,
            area_b
        ) / min(
            area_a,
            area_b
        )

    # ========================================================
    # SHOULD MERGE
    # ========================================================

    def should_merge(
        self,
        region_a,
        region_b
    ) -> bool:

        area_a = self.area(
            region_a
        )

        area_b = self.area(
            region_b
        )

        if area_a <= 0 or area_b <= 0:

            return False

        iou = self.calculate_iou(
            region_a,
            region_b
        )

        # ----------------------------------------------------
        # Rule 1:
        # Strong overlap always wins.
        # ----------------------------------------------------

        if iou >= self.iou_threshold:

            return True

        # ----------------------------------------------------
        # Rule 2:
        # Nearby regions must have similar scale.
        # ----------------------------------------------------

        ratio = self.size_ratio(
            region_a,
            region_b
        )

        if (
            ratio
            >
            self.max_size_ratio
        ):

            return False

        # ----------------------------------------------------
        # Rule 3:
        # Small gap.
        # ----------------------------------------------------

        horizontal_gap, vertical_gap = (
            self.calculate_gap(
                region_a,
                region_b
            )
        )

        if (
            horizontal_gap
            <= self.proximity_threshold
            and
            vertical_gap
            <= self.proximity_threshold
        ):

            return True

        # ----------------------------------------------------
        # Rule 4:
        # Slightly larger gap is allowed only
        # when the regions are very similar in size.
        # ----------------------------------------------------

        if (
            horizontal_gap
            <= self.max_gap
            and
            vertical_gap
            <= self.max_gap
            and
            ratio
            <= 2.0
        ):

            return True

        return False

    # ========================================================
    # MERGE BBOX
    # ========================================================

    @classmethod
    def merge_bbox(
        cls,
        region_a,
        region_b
    ):

        ax1 = int(
            cls.get_value(
                region_a,
                "x"
            )
        )

        ay1 = int(
            cls.get_value(
                region_a,
                "y"
            )
        )

        ax2 = (
            ax1
            +
            int(
                cls.get_value(
                    region_a,
                    "width"
                )
            )
        )

        ay2 = (
            ay1
            +
            int(
                cls.get_value(
                    region_a,
                    "height"
                )
            )
        )

        bx1 = int(
            cls.get_value(
                region_b,
                "x"
            )
        )

        by1 = int(
            cls.get_value(
                region_b,
                "y"
            )
        )

        bx2 = (
            bx1
            +
            int(
                cls.get_value(
                    region_b,
                    "width"
                )
            )
        )

        by2 = (
            by1
            +
            int(
                cls.get_value(
                    region_b,
                    "height"
                )
            )
        )

        return {

            "x": min(
                ax1,
                bx1
            ),

            "y": min(
                ay1,
                by1
            ),

            "x2": max(
                ax2,
                bx2
            ),

            "y2": max(
                ay2,
                by2
            ),
        }

    # ========================================================
    # UNION-FIND
    # ========================================================

    @staticmethod
    def find(
        parent,
        value
    ):

        while (
            parent[value]
            != value
        ):

            parent[value] = (
                parent[
                    parent[value]
                ]
            )

            value = parent[
                value
            ]

        return value

    # ========================================================
    # MERGE
    # ========================================================

    def merge(
        self,
        regions
    ) -> List[MergedVisualRegion]:

        if not regions:

            return []

        data = [

            self.region_to_dict(
                region
            )

            for region in regions

        ]

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Sort by area descending.
        #
        # Large regions are protected from absorbing
        # unrelated small regions.
        # ----------------------------------------------------

        data.sort(
            key=lambda region:
                self.area(
                    region
                ),
            reverse=True
        )

        parent = list(
            range(
                len(data)
            )
        )

        # ----------------------------------------------------
        # Pairwise grouping
        # ----------------------------------------------------

        for i in range(
            len(data)
        ):

            for j in range(
                i + 1,
                len(data)
            ):

                if self.should_merge(
                    data[i],
                    data[j]
                ):

                    root_i = self.find(
                        parent,
                        i
                    )

                    root_j = self.find(
                        parent,
                        j
                    )

                    if root_i != root_j:

                        parent[root_j] = (
                            root_i
                        )

        # ----------------------------------------------------
        # Build groups
        # ----------------------------------------------------

        groups = {}

        for index in range(
            len(data)
        ):

            root = self.find(
                parent,
                index
            )

            groups.setdefault(
                root,
                []
            ).append(
                data[index]
            )

        # ----------------------------------------------------
        # Build merged regions
        # ----------------------------------------------------

        merged = []

        for group in groups.values():

            x1 = min(
                region["x"]
                for region in group
            )

            y1 = min(
                region["y"]
                for region in group
            )

            x2 = max(
                region["x"]
                +
                region["width"]
                for region in group
            )

            y2 = max(
                region["y"]
                +
                region["height"]
                for region in group
            )

            x1 = max(
                0,
                x1 - self.padding
            )

            y1 = max(
                0,
                y1 - self.padding
            )

            x2 += self.padding

            y2 += self.padding

            width = (
                x2 - x1
            )

            height = (
                y2 - y1
            )

            merged.append(

                MergedVisualRegion(

                    region_id=0,

                    x=x1,

                    y=y1,

                    width=width,

                    height=height,

                    area=(
                        width
                        * height
                    ),

                    component_count=len(
                        group
                    ),

                    source_regions=group,

                )
            )

        # ----------------------------------------------------
        # Sort spatially
        # ----------------------------------------------------

        merged.sort(
            key=lambda region: (
                region.y,
                region.x
            )
        )

        # ----------------------------------------------------
        # Assign IDs
        # ----------------------------------------------------

        for index, region in enumerate(
            merged,
            start=1
        ):

            region.region_id = index

        return merged