from dataclasses import dataclass, asdict
from typing import List, Dict, Any


# ============================================================
# MERGED VISUAL REGION
# ============================================================

@dataclass
class MergedVisualRegion:
    """
    Represents one logical visual region created by merging
    multiple OpenCV contour candidates.
    """

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
    Merge raw OpenCV visual candidates into logical visual
    regions.

    Design goals:

    1. Merge overlapping contours.
    2. Merge contours that are inside another contour.
    3. Merge nearby components when they clearly belong together.
    4. Prevent a large region from swallowing unrelated content.
    5. Avoid merging across large whitespace.
    6. Preserve separate diagrams/annotations when possible.
    """

    def __init__(
        self,
        iou_threshold: float = 0.08,
        containment_threshold: float = 0.65,
        proximity_threshold: int = 18,
        max_gap: int = 30,
        padding: int = 8,
        max_size_ratio: float = 12.0,
        max_merged_width: int = 1800,
        max_merged_height: int = 900,
        page_swallow_ratio: float = 0.72,
    ):

        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold
        self.proximity_threshold = proximity_threshold
        self.max_gap = max_gap
        self.padding = padding
        self.max_size_ratio = max_size_ratio
        self.max_merged_width = max_merged_width
        self.max_merged_height = max_merged_height
        self.page_swallow_ratio = page_swallow_ratio

    # ========================================================
    # GENERIC ACCESS
    # ========================================================

    @staticmethod
    def get_value(
        region,
        key: str,
        default=0,
    ):

        if isinstance(region, dict):
            return region.get(
                key,
                default,
            )

        return getattr(
            region,
            key,
            default,
        )

    # ========================================================
    # CONVERT REGION TO DICT
    # ========================================================

    @staticmethod
    def region_to_dict(
        region,
    ) -> Dict[str, Any]:

        if isinstance(region, dict):
            return dict(region)

        if hasattr(region, "to_dict"):
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
                "visual_candidate",
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
    # AREA
    # ========================================================

    @classmethod
    def area(
        cls,
        region,
    ) -> int:

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

        return max(
            0,
            width * height,
        )

    # ========================================================
    # BOUNDS
    # ========================================================

    @classmethod
    def bounds(
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
    # IOU
    # ========================================================

    @classmethod
    def calculate_iou(
        cls,
        region_a,
        region_b,
    ) -> float:

        ax1, ay1, ax2, ay2 = cls.bounds(
            region_a
        )

        bx1, by1, bx2, by2 = cls.bounds(
            region_b
        )

        ix1 = max(
            ax1,
            bx1,
        )

        iy1 = max(
            ay1,
            by1,
        )

        ix2 = min(
            ax2,
            bx2,
        )

        iy2 = min(
            ay2,
            by2,
        )

        iw = max(
            0,
            ix2 - ix1,
        )

        ih = max(
            0,
            iy2 - iy1,
        )

        intersection = iw * ih

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

        return intersection / union

    # ========================================================
    # CONTAINMENT
    # ========================================================

    @classmethod
    def containment_ratio(
        cls,
        region_a,
        region_b,
    ) -> float:
        """
        How much of the smaller region is contained inside
        the larger region.
        """

        ax1, ay1, ax2, ay2 = cls.bounds(
            region_a
        )

        bx1, by1, bx2, by2 = cls.bounds(
            region_b
        )

        ix1 = max(
            ax1,
            bx1,
        )

        iy1 = max(
            ay1,
            by1,
        )

        ix2 = min(
            ax2,
            bx2,
        )

        iy2 = min(
            ay2,
            by2,
        )

        iw = max(
            0,
            ix2 - ix1,
        )

        ih = max(
            0,
            iy2 - iy1,
        )

        intersection = iw * ih

        smaller_area = min(
            cls.area(region_a),
            cls.area(region_b),
        )

        if smaller_area <= 0:
            return 0.0

        return intersection / smaller_area

    # ========================================================
    # GAP
    # ========================================================

    @classmethod
    def calculate_gap(
        cls,
        region_a,
        region_b,
    ):

        ax1, ay1, ax2, ay2 = cls.bounds(
            region_a
        )

        bx1, by1, bx2, by2 = cls.bounds(
            region_b
        )

        horizontal_gap = max(
            0,
            bx1 - ax2,
            ax1 - bx2,
        )

        vertical_gap = max(
            0,
            by1 - ay2,
            ay1 - by2,
        )

        return (
            horizontal_gap,
            vertical_gap,
        )

    # ========================================================
    # CENTER DISTANCE
    # ========================================================

    @classmethod
    def center_distance(
        cls,
        region_a,
        region_b,
    ):

        ax1, ay1, ax2, ay2 = cls.bounds(
            region_a
        )

        bx1, by1, bx2, by2 = cls.bounds(
            region_b
        )

        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0

        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        dx = acx - bcx
        dy = acy - bcy

        return (
            dx * dx + dy * dy
        ) ** 0.5

    # ========================================================
    # SIZE RATIO
    # ========================================================

    @classmethod
    def size_ratio(
        cls,
        region_a,
        region_b,
    ) -> float:

        area_a = max(
            1,
            cls.area(region_a),
        )

        area_b = max(
            1,
            cls.area(region_b),
        )

        return (
            max(area_a, area_b)
            /
            min(area_a, area_b)
        )

    # ========================================================
    # ASPECT RATIO
    # ========================================================

    @classmethod
    def aspect_ratio(
        cls,
        region,
    ) -> float:

        width = max(
            1,
            int(
                cls.get_value(
                    region,
                    "width",
                    1,
                )
            ),
        )

        height = max(
            1,
            int(
                cls.get_value(
                    region,
                    "height",
                    1,
                )
            ),
        )

        return max(
            width / height,
            height / width,
        )

    # ========================================================
    # SHOULD MERGE
    # ========================================================

    def should_merge(
        self,
        region_a,
        region_b,
    ) -> bool:

        area_a = self.area(
            region_a
        )

        area_b = self.area(
            region_b
        )

        if area_a <= 0 or area_b <= 0:
            return False

        # ----------------------------------------------------
        # Strong overlap
        # ----------------------------------------------------

        iou = self.calculate_iou(
            region_a,
            region_b,
        )

        if iou >= self.iou_threshold:
            return True

        # ----------------------------------------------------
        # Containment
        #
        # Example:
        #
        # large contour
        #     ┌───────────────┐
        #     │   ┌───────┐   │
        #     │   │small  │   │
        #     │   └───────┘   │
        #     └───────────────┘
        #
        # These should be one logical region.
        # ----------------------------------------------------

        containment = self.containment_ratio(
            region_a,
            region_b,
        )

        if containment >= self.containment_threshold:
            return True

        # ----------------------------------------------------
        # Gap
        # ----------------------------------------------------

        horizontal_gap, vertical_gap = (
            self.calculate_gap(
                region_a,
                region_b,
            )
        )

        ratio = self.size_ratio(
            region_a,
            region_b,
        )

        # ----------------------------------------------------
        # Do not merge extremely different objects merely
        # because they are close.
        # ----------------------------------------------------

        if ratio > self.max_size_ratio:
            return False

        # ----------------------------------------------------
        # Directly touching / almost touching.
        # ----------------------------------------------------

        if (
            horizontal_gap <= self.proximity_threshold
            and
            vertical_gap <= self.proximity_threshold
        ):
            return True

        # ----------------------------------------------------
        # Slightly larger gap.
        #
        # Only allow this when:
        #
        # - objects have reasonably similar size
        # - one dimension is aligned
        # ----------------------------------------------------

        if (
            horizontal_gap <= self.max_gap
            and
            vertical_gap <= self.max_gap
            and
            ratio <= 4.0
        ):
            return True

        # ----------------------------------------------------
        # Same horizontal line.
        # ----------------------------------------------------

        if (
            horizontal_gap <= self.max_gap
            and
            vertical_gap <= 12
            and
            ratio <= 8.0
        ):
            return True

        # ----------------------------------------------------
        # Same vertical line.
        # ----------------------------------------------------

        if (
            vertical_gap <= self.max_gap
            and
            horizontal_gap <= 12
            and
            ratio <= 8.0
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
        region_b,
    ):

        ax1, ay1, ax2, ay2 = cls.bounds(
            region_a
        )

        bx1, by1, bx2, by2 = cls.bounds(
            region_b
        )

        return {
            "x": min(
                ax1,
                bx1,
            ),
            "y": min(
                ay1,
                by1,
            ),
            "x2": max(
                ax2,
                bx2,
            ),
            "y2": max(
                ay2,
                by2,
            ),
        }

    # ========================================================
    # UNION FIND
    # ========================================================

    @staticmethod
    def find(
        parent,
        value,
    ):

        while parent[value] != value:

            parent[value] = parent[
                parent[value]
            ]

            value = parent[value]

        return value

    # ========================================================
    # UNION
    # ========================================================

    @staticmethod
    def union(
        parent,
        rank,
        a,
        b,
    ):

        root_a = VisualMerger.find(
            parent,
            a,
        )

        root_b = VisualMerger.find(
            parent,
            b,
        )

        if root_a == root_b:
            return

        if rank[root_a] < rank[root_b]:

            parent[root_a] = root_b

        elif rank[root_a] > rank[root_b]:

            parent[root_b] = root_a

        else:

            parent[root_b] = root_a
            rank[root_a] += 1

    # ========================================================
    # GROUP BBOX
    # ========================================================

    @classmethod
    def group_bbox(
        cls,
        group,
    ):

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
            + region["width"]
            for region in group
        )

        y2 = max(
            region["y"]
            + region["height"]
            for region in group
        )

        return (
            x1,
            y1,
            x2,
            y2,
        )

    # ========================================================
    # GROUP AREA
    # ========================================================

    @classmethod
    def group_bbox_area(
        cls,
        group,
    ):

        x1, y1, x2, y2 = cls.group_bbox(
            group
        )

        return (
            max(0, x2 - x1)
            *
            max(0, y2 - y1)
        )

    # ========================================================
    # PROTECT GIANT GROUPS
    # ========================================================

    def group_is_reasonable(
        self,
        group,
    ) -> bool:

        if not group:
            return True

        x1, y1, x2, y2 = (
            self.group_bbox(group)
        )

        width = x2 - x1
        height = y2 - y1

        if (
            width > self.max_merged_width
            or
            height > self.max_merged_height
        ):
            return False

        return True

    # ========================================================
    # MERGE
    # ========================================================

    def merge(
        self,
        regions,
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
        # Remove invalid regions
        # ----------------------------------------------------

        data = [
            region
            for region in data
            if (
                int(region.get("width", 0)) > 0
                and
                int(region.get("height", 0)) > 0
            )
        ]

        if not data:
            return []

        # ----------------------------------------------------
        # Sort by area.
        # ----------------------------------------------------

        data.sort(
            key=lambda region:
                self.area(region),
            reverse=True,
        )

        parent = list(
            range(len(data))
        )

        rank = [
            0
            for _ in data
        ]

        # ----------------------------------------------------
        # Pairwise grouping.
        # ----------------------------------------------------

        for i in range(
            len(data)
        ):

            for j in range(
                i + 1,
                len(data),
            ):

                region_a = data[i]
                region_b = data[j]

                if not self.should_merge(
                    region_a,
                    region_b,
                ):
                    continue

                # ------------------------------------------------
                # Prevent a large contour from swallowing another
                # distant object.
                # ------------------------------------------------

                current_root_a = self.find(
                    parent,
                    i,
                )

                current_root_b = self.find(
                    parent,
                    j,
                )

                if current_root_a == current_root_b:
                    continue

                # ------------------------------------------------
                # Simulate resulting group.
                # ------------------------------------------------

                group_a = [
                    data[index]
                    for index in range(
                        len(data)
                    )
                    if self.find(
                        parent,
                        index,
                    ) == current_root_a
                ]

                group_b = [
                    data[index]
                    for index in range(
                        len(data)
                    )
                    if self.find(
                        parent,
                        index,
                    ) == current_root_b
                ]

                combined = (
                    group_a
                    +
                    group_b
                )

                # ------------------------------------------------
                # Don't allow enormous accidental page groups.
                # ------------------------------------------------

                if not self.group_is_reasonable(
                    combined
                ):

                    continue

                self.union(
                    parent,
                    rank,
                    i,
                    j,
                )

        # ----------------------------------------------------
        # Build groups.
        # ----------------------------------------------------

        groups = {}

        for index in range(
            len(data)
        ):

            root = self.find(
                parent,
                index,
            )

            groups.setdefault(
                root,
                [],
            ).append(
                data[index]
            )

        # ----------------------------------------------------
        # Build merged regions.
        # ----------------------------------------------------

        merged = []

        for group in groups.values():

            x1, y1, x2, y2 = (
                self.group_bbox(
                    group
                )
            )

            # ------------------------------------------------
            # Padding.
            # ------------------------------------------------

            x1 = max(
                0,
                x1 - self.padding,
            )

            y1 = max(
                0,
                y1 - self.padding,
            )

            x2 += self.padding
            y2 += self.padding

            width = x2 - x1
            height = y2 - y1

            merged.append(
                MergedVisualRegion(
                    region_id=0,
                    x=x1,
                    y=y1,
                    width=width,
                    height=height,
                    area=(
                        width
                        *
                        height
                    ),
                    component_count=len(
                        group
                    ),
                    source_regions=group,
                )
            )

        # ----------------------------------------------------
        # Spatial sorting.
        # ----------------------------------------------------

        merged.sort(
            key=lambda region: (
                region.y,
                region.x,
            )
        )

        # ----------------------------------------------------
        # Assign IDs.
        # ----------------------------------------------------

        for index, region in enumerate(
            merged,
            start=1,
        ):

            region.region_id = index

        return merged