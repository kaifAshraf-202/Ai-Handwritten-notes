from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image


# ============================================================
# SPLIT REGION
# ============================================================

@dataclass
class SplitRegion:
    """
    Represents a visual region after intelligent splitting.

    Coordinates are relative to the original page image.
    """

    region_id: int

    x: int
    y: int
    width: int
    height: int

    source_region_id: int = 0
    component_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (
            self.x,
            self.y,
            self.width,
            self.height,
        )


# ============================================================
# VISUAL REGION SPLITTER
# ============================================================

class VisualRegionSplitter:
    """
    Intelligent splitter for merged visual regions.

    Strategy:

        merged region
              |
              v
        create ink mask
              |
              v
        connected components
              |
              v
        remove tiny noise
              |
              v
        detect whitespace gaps
              |
              v
        projection-based splitting
              |
              v
        component grouping
              |
              v
        final visual regions

    Important design decision:

    We DO NOT use unrestricted transitive union-find merging.

    A chain such as:

        A -- B -- C -- D

    must not automatically force A/B/C/D into one region.

    Instead, components are grouped using local geometric
    relationships and whitespace evidence.
    """

    def __init__(
        self,
        min_component_area: int = 80,
        min_region_area: int = 1200,

        # Maximum distance for components that are genuinely
        # close to each other.
        component_gap: int = 22,

        # More generous gap for very small components such as
        # chemical symbols.
        small_component_gap: int = 14,

        # Padding around final regions.
        padding: int = 8,

        min_width: int = 15,
        min_height: int = 15,

        # Projection splitting.
        min_split_gap: int = 18,
        min_split_ratio: float = 0.025,

        # Prevent tiny fragments.
        min_group_components: int = 1,

        # A region must be sufficiently large before we attempt
        # aggressive splitting.
        large_region_area: int = 18000,

        # Large empty gap required for a strong split.
        strong_gap: int = 35,
    ):
        self.min_component_area = min_component_area
        self.min_region_area = min_region_area
        self.component_gap = component_gap
        self.small_component_gap = small_component_gap

        self.padding = padding

        self.min_width = min_width
        self.min_height = min_height

        self.min_split_gap = min_split_gap
        self.min_split_ratio = min_split_ratio

        self.min_group_components = min_group_components
        self.large_region_area = large_region_area
        self.strong_gap = strong_gap

    # ========================================================
    # PIL -> OPENCV
    # ========================================================

    @staticmethod
    def pil_to_cv(
        image: Image.Image
    ) -> np.ndarray:

        array = np.array(image)

        if len(array.shape) == 2:
            return cv2.cvtColor(
                array,
                cv2.COLOR_GRAY2BGR
            )

        return cv2.cvtColor(
            array,
            cv2.COLOR_RGB2BGR
        )

    # ========================================================
    # CROP
    # ========================================================

    @staticmethod
    def crop_region(
        image: Image.Image,
        region
    ) -> Image.Image:

        x = max(
            0,
            int(region.x)
        )

        y = max(
            0,
            int(region.y)
        )

        right = min(
            image.width,
            x + int(region.width)
        )

        bottom = min(
            image.height,
            y + int(region.height)
        )

        return image.crop(
            (
                x,
                y,
                right,
                bottom
            )
        )

    # ========================================================
    # CREATE INK MASK
    # ========================================================

    def create_ink_mask(
        self,
        image: Image.Image
    ) -> np.ndarray:
        """
        Create a robust foreground/ink mask.

        We combine:

        1. dark ink detection
        2. saturation/color detection

        This is important because the page contains colored
        handwriting and highlights.
        """

        cv_image = self.pil_to_cv(image)

        gray = cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Dark ink
        # ----------------------------------------------------

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        dark_mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12
        )

        # ----------------------------------------------------
        # Colored ink
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2HSV
        )

        saturation = hsv[:, :, 1]

        color_mask = np.where(
            saturation > 35,
            255,
            0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        mask = cv2.bitwise_or(
            dark_mask,
            color_mask
        )

        # ----------------------------------------------------
        # Remove tiny isolated noise
        # ----------------------------------------------------

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # ----------------------------------------------------
        # Close tiny holes in strokes
        # ----------------------------------------------------

        close_kernel = np.ones(
            (2, 2),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        return mask

    # ========================================================
    # CONNECTED COMPONENTS
    # ========================================================

    def find_components(
        self,
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int, int]]:

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        components = []

        for label in range(
            1,
            num_labels
        ):

            x = int(
                stats[
                    label,
                    cv2.CC_STAT_LEFT
                ]
            )

            y = int(
                stats[
                    label,
                    cv2.CC_STAT_TOP
                ]
            )

            width = int(
                stats[
                    label,
                    cv2.CC_STAT_WIDTH
                ]
            )

            height = int(
                stats[
                    label,
                    cv2.CC_STAT_HEIGHT
                ]
            )

            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA
                ]
            )

            if area < self.min_component_area:
                continue

            if width < 3:
                continue

            if height < 3:
                continue

            components.append(
                (
                    x,
                    y,
                    width,
                    height,
                    area
                )
            )

        return components

    # ========================================================
    # GEOMETRY
    # ========================================================

    @staticmethod
    def horizontal_gap(
        a,
        b
    ) -> int:

        ax1 = a[0]
        ax2 = a[0] + a[2]

        bx1 = b[0]
        bx2 = b[0] + b[2]

        if ax2 < bx1:
            return bx1 - ax2

        if bx2 < ax1:
            return ax1 - bx2

        return 0

    @staticmethod
    def vertical_gap(
        a,
        b
    ) -> int:

        ay1 = a[1]
        ay2 = a[1] + a[3]

        by1 = b[1]
        by2 = b[1] + b[3]

        if ay2 < by1:
            return by1 - ay2

        if by2 < ay1:
            return ay1 - by2

        return 0

    @staticmethod
    def intersection(
        a,
        b
    ) -> int:

        ax1 = a[0]
        ay1 = a[1]
        ax2 = a[0] + a[2]
        ay2 = a[1] + a[3]

        bx1 = b[0]
        by1 = b[1]
        bx2 = b[0] + b[2]
        by2 = b[1] + b[3]

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

        if ix2 <= ix1:
            return 0

        if iy2 <= iy1:
            return 0

        return (
            ix2 - ix1
        ) * (
            iy2 - iy1
        )

    @staticmethod
    def overlap_ratio(
        a,
        b
    ) -> float:

        intersection = (
            VisualRegionSplitter.intersection(
                a,
                b
            )
        )

        area_a = a[2] * a[3]
        area_b = b[2] * b[3]

        smaller = min(
            area_a,
            area_b
        )

        if smaller <= 0:
            return 0.0

        return (
            intersection
            / smaller
        )

    @staticmethod
    def center(
        component
    ) -> Tuple[float, float]:

        return (
            component[0]
            + component[2] / 2.0,

            component[1]
            + component[3] / 2.0
        )

    # ========================================================
    # ALIGNMENT
    # ========================================================

    @staticmethod
    def horizontal_alignment(
        a,
        b
    ) -> float:
        """
        Vertical overlap ratio.

        High value means two components sit on roughly
        the same horizontal line.
        """

        ay1 = a[1]
        ay2 = a[1] + a[3]

        by1 = b[1]
        by2 = b[1] + b[3]

        overlap = max(
            0,
            min(
                ay2,
                by2
            ) - max(
                ay1,
                by1
            )
        )

        denominator = min(
            a[3],
            b[3]
        )

        if denominator <= 0:
            return 0.0

        return overlap / denominator

    @staticmethod
    def vertical_alignment(
        a,
        b
    ) -> float:
        """
        Horizontal overlap ratio.
        """

        ax1 = a[0]
        ax2 = a[0] + a[2]

        bx1 = b[0]
        bx2 = b[0] + b[2]

        overlap = max(
            0,
            min(
                ax2,
                bx2
            ) - max(
                ax1,
                bx1
            )
        )

        denominator = min(
            a[2],
            b[2]
        )

        if denominator <= 0:
            return 0.0

        return overlap / denominator

    # ========================================================
    # SHOULD MERGE
    # ========================================================

    def should_merge(
        self,
        a,
        b
    ) -> bool:
        """
        Conservative local grouping.

        Unlike the previous implementation, this does not merge
        simply because either horizontal OR vertical distance is
        small.

        The components need geometric evidence that they belong
        to the same object.
        """

        horizontal = self.horizontal_gap(
            a,
            b
        )

        vertical = self.vertical_gap(
            a,
            b
        )

        overlap = self.overlap_ratio(
            a,
            b
        )

        h_align = self.horizontal_alignment(
            a,
            b
        )

        v_align = self.vertical_alignment(
            a,
            b
        )

        min_area = min(
            a[4],
            b[4]
        )

        # ----------------------------------------------------
        # 1. Significant overlap
        # ----------------------------------------------------

        if overlap >= 0.35:
            return True

        # ----------------------------------------------------
        # 2. Components touching each other
        # ----------------------------------------------------

        if horizontal == 0 and vertical == 0:

            # They touch and have some alignment.
            if (
                h_align >= 0.15
                or
                v_align >= 0.15
            ):
                return True

        # ----------------------------------------------------
        # 3. Horizontal neighbours
        # ----------------------------------------------------

        if (
            horizontal <= self.component_gap
            and
            h_align >= 0.35
        ):

            # Avoid joining two large independent objects
            # merely because they happen to be close.
            if min_area < 5000:
                return True

            if horizontal <= 8:
                return True

        # ----------------------------------------------------
        # 4. Vertical neighbours
        # ----------------------------------------------------

        if (
            vertical <= self.component_gap
            and
            v_align >= 0.35
        ):

            if min_area < 5000:
                return True

            if vertical <= 8:
                return True

        # ----------------------------------------------------
        # 5. Very small components near a larger component
        #
        # Useful for:
        #
        #   chemical symbols
        #   dots
        #   arrows
        #   handwriting strokes
        # ----------------------------------------------------

        small_a = a[4] < 1000
        small_b = b[4] < 1000

        if small_a or small_b:

            if (
                horizontal <= self.small_component_gap
                and
                h_align >= 0.25
            ):
                return True

            if (
                vertical <= self.small_component_gap
                and
                v_align >= 0.25
            ):
                return True

        return False

    # ========================================================
    # LOCAL GROUPING
    # ========================================================

    def group_components(
        self,
        components
    ):

        count = len(
            components
        )

        if count == 0:
            return []

        # ----------------------------------------------------
        # Start every component as its own group.
        # ----------------------------------------------------

        groups = [
            [component]
            for component in components
        ]

        # ----------------------------------------------------
        # Iteratively merge only strongly related groups.
        #
        # This is intentionally NOT unrestricted union-find.
        # ----------------------------------------------------

        changed = True

        while changed:

            changed = False

            best_pair = None
            best_score = 0.0

            for i in range(
                len(groups)
            ):

                for j in range(
                    i + 1,
                    len(groups)
                ):

                    score = (
                        self.group_relation_score(
                            groups[i],
                            groups[j]
                        )
                    )

                    if score > best_score:

                        best_score = score
                        best_pair = (
                            i,
                            j
                        )

            if (
                best_pair is not None
                and
                best_score >= 0.70
            ):

                i, j = best_pair

                merged = (
                    groups[i]
                    +
                    groups[j]
                )

                groups[i] = merged

                del groups[j]

                changed = True

        return groups

    # ========================================================
    # GROUP RELATION SCORE
    # ========================================================

    def group_relation_score(
        self,
        group_a,
        group_b
    ) -> float:
        """
        Calculate how strongly two groups belong together.

        We use the BEST pair rather than allowing any weak
        component to connect two large groups.
        """

        best = 0.0

        for a in group_a:

            for b in group_b:

                if not self.should_merge(
                    a,
                    b
                ):
                    continue

                horizontal = self.horizontal_gap(
                    a,
                    b
                )

                vertical = self.vertical_gap(
                    a,
                    b
                )

                overlap = self.overlap_ratio(
                    a,
                    b
                )

                h_align = self.horizontal_alignment(
                    a,
                    b
                )

                v_align = self.vertical_alignment(
                    a,
                    b
                )

                score = 0.0

                # Strong overlap.
                score += (
                    min(
                        overlap,
                        1.0
                    )
                    * 0.40
                )

                # Strong alignment.
                score += (
                    max(
                        h_align,
                        v_align
                    )
                    * 0.30
                )

                # Distance.
                distance = min(
                    horizontal,
                    vertical
                )

                if distance <= 3:
                    score += 0.25

                elif distance <= 8:
                    score += 0.20

                elif distance <= 15:
                    score += 0.12

                elif distance <= 22:
                    score += 0.05

                # Small components are easier to associate.
                if min(
                    a[4],
                    b[4]
                ) < 1000:
                    score += 0.10

                best = max(
                    best,
                    score
                )

        return min(
            best,
            1.0
        )

    # ========================================================
    # GROUP BBOX
    # ========================================================

    @staticmethod
    def group_bbox(
        group
    ):

        x1 = min(
            component[0]
            for component in group
        )

        y1 = min(
            component[1]
            for component in group
        )

        x2 = max(
            component[0]
            + component[2]
            for component in group
        )

        y2 = max(
            component[1]
            + component[3]
            for component in group
        )

        return (
            x1,
            y1,
            x2 - x1,
            y2 - y1
        )

    # ========================================================
    # VALID GROUP
    # ========================================================

    def valid_group(
        self,
        bbox
    ) -> bool:

        _, _, width, height = bbox

        area = (
            width
            * height
        )

        if area < self.min_region_area:
            return False

        if width < self.min_width:
            return False

        if height < self.min_height:
            return False

        return True

    # ========================================================
    # PROJECTION GAPS
    # ========================================================

    def projection_gaps(
        self,
        mask: np.ndarray
    ):
        """
        Detect large whitespace gaps.

        Returns:

            horizontal_gaps
            vertical_gaps
        """

        if mask.size == 0:
            return [], []

        binary = (
            mask > 0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Number of ink pixels per column.
        # ----------------------------------------------------

        column_ink = (
            binary.sum(
                axis=0
            )
        )

        # Number of ink pixels per row.
        row_ink = (
            binary.sum(
                axis=1
            )
        )

        width = mask.shape[1]
        height = mask.shape[0]

        column_threshold = max(
            1,
            int(
                height
                * self.min_split_ratio
            )
        )

        row_threshold = max(
            1,
            int(
                width
                * self.min_split_ratio
            )
        )

        empty_columns = (
            column_ink
            <= column_threshold
        )

        empty_rows = (
            row_ink
            <= row_threshold
        )

        horizontal_gaps = (
            self.runs_from_boolean(
                empty_columns,
                self.min_split_gap
            )
        )

        vertical_gaps = (
            self.runs_from_boolean(
                empty_rows,
                self.min_split_gap
            )
        )

        return (
            horizontal_gaps,
            vertical_gaps
        )

    # ========================================================
    # BOOLEAN RUNS
    # ========================================================

    @staticmethod
    def runs_from_boolean(
        values,
        minimum_length
    ):

        runs = []

        start = None

        for index, value in enumerate(
            values
        ):

            if value:

                if start is None:
                    start = index

            else:

                if start is not None:

                    length = (
                        index
                        - start
                    )

                    if length >= minimum_length:

                        runs.append(
                            (
                                start,
                                index
                            )
                        )

                    start = None

        if start is not None:

            length = (
                len(values)
                - start
            )

            if length >= minimum_length:

                runs.append(
                    (
                        start,
                        len(values)
                    )
                )

        return runs

    # ========================================================
    # STRONG SPLIT GAPS
    # ========================================================

    def strong_projection_splits(
        self,
        mask: np.ndarray
    ):
        """
        Find only strong whitespace gaps.

        This is intentionally conservative.
        """

        horizontal, vertical = (
            self.projection_gaps(
                mask
            )
        )

        strong_horizontal = [
            gap
            for gap in horizontal
            if (
                gap[1] - gap[0]
            ) >= self.strong_gap
        ]

        strong_vertical = [
            gap
            for gap in vertical
            if (
                gap[1] - gap[0]
            ) >= self.strong_gap
        ]

        return (
            strong_horizontal,
            strong_vertical
        )

    # ========================================================
    # SPLIT MASK USING GAPS
    # ========================================================

    def split_using_projection(
        self,
        mask: np.ndarray
    ):
        """
        Split a mask along strong whitespace gaps.

        Returns bounding boxes in crop coordinates.
        """

        height, width = (
            mask.shape[:2]
        )

        if (
            width < 2
            or
            height < 2
        ):
            return []

        horizontal_gaps, vertical_gaps = (
            self.strong_projection_splits(
                mask
            )
        )

        # ----------------------------------------------------
        # No meaningful gap.
        # ----------------------------------------------------

        if (
            not horizontal_gaps
            and
            not vertical_gaps
        ):
            return []

        x_cuts = [
            gap[1]
            for gap in horizontal_gaps
        ]

        y_cuts = [
            gap[1]
            for gap in vertical_gaps
        ]

        x_boundaries = (
            [0]
            + x_cuts
            + [width]
        )

        y_boundaries = (
            [0]
            + y_cuts
            + [height]
        )

        candidates = []

        for yi in range(
            len(y_boundaries) - 1
        ):

            y1 = y_boundaries[yi]
            y2 = y_boundaries[yi + 1]

            if y2 - y1 < self.min_height:
                continue

            for xi in range(
                len(x_boundaries) - 1
            ):

                x1 = x_boundaries[xi]
                x2 = x_boundaries[xi + 1]

                if x2 - x1 < self.min_width:
                    continue

                submask = mask[
                    y1:y2,
                    x1:x2
                ]

                ink = cv2.countNonZero(
                    submask
                )

                if ink < self.min_component_area:
                    continue

                candidates.append(
                    (
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    )
                )

        return candidates

    # ========================================================
    # CROP GROUP TO INK
    # ========================================================

    @staticmethod
    def tight_bbox_from_mask(
        mask: np.ndarray
    ):

        points = cv2.findNonZero(
            mask
        )

        if points is None:
            return None

        x, y, width, height = (
            cv2.boundingRect(
                points
            )
        )

        return (
            x,
            y,
            width,
            height
        )

    # ========================================================
    # ADD PADDING
    # ========================================================

    def padded_bbox(
        self,
        bbox,
        region_width,
        region_height
    ):

        x, y, width, height = bbox

        x -= self.padding
        y -= self.padding

        width += (
            self.padding
            * 2
        )

        height += (
            self.padding
            * 2
        )

        x = max(
            0,
            x
        )

        y = max(
            0,
            y
        )

        right = min(
            region_width,
            x + width
        )

        bottom = min(
            region_height,
            y + height
        )

        return (
            x,
            y,
            right - x,
            bottom - y
        )

    # ========================================================
    # MERGE OVERLAPPING FINAL BOXES
    # ========================================================

    @staticmethod
    def merge_overlapping_boxes(
        boxes
    ):

        if not boxes:
            return []

        result = []

        for box in boxes:

            merged = False

            for index, existing in enumerate(
                result
            ):

                ax1, ay1, aw, ah = existing
                bx1, by1, bw, bh = box

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

                if (
                    ix2 <= ix1
                    or
                    iy2 <= iy1
                ):
                    continue

                intersection = (
                    ix2 - ix1
                ) * (
                    iy2 - iy1
                )

                area_small = min(
                    aw * ah,
                    bw * bh
                )

                if (
                    area_small > 0
                    and
                    intersection
                    / area_small
                    >= 0.60
                ):

                    nx1 = min(
                        ax1,
                        bx1
                    )

                    ny1 = min(
                        ay1,
                        by1
                    )

                    nx2 = max(
                        ax2,
                        bx2
                    )

                    ny2 = max(
                        ay2,
                        by2
                    )

                    result[index] = (
                        nx1,
                        ny1,
                        nx2 - nx1,
                        ny2 - ny1
                    )

                    merged = True
                    break

            if not merged:
                result.append(
                    box
                )

        return result

    # ========================================================
    # SPLIT ONE REGION
    # ========================================================

    def split_region(
        self,
        image: Image.Image,
        region
    ) -> List[SplitRegion]:

        region_width = int(
            region.width
        )

        region_height = int(
            region.height
        )

        region_area = (
            region_width
            * region_height
        )

        source_id = int(
            getattr(
                region,
                "region_id",
                0
            )
        )

        # ----------------------------------------------------
        # Small regions are not split.
        # ----------------------------------------------------

        if (
            region_area
            < self.min_region_area
        ):

            return [
                SplitRegion(
                    region_id=0,
                    x=int(region.x),
                    y=int(region.y),
                    width=region_width,
                    height=region_height,
                    source_region_id=source_id,
                    component_count=1
                )
            ]

        cropped = self.crop_region(
            image,
            region
        )

        mask = self.create_ink_mask(
            cropped
        )

        components = self.find_components(
            mask
        )

        # ----------------------------------------------------
        # Nothing meaningful detected.
        # ----------------------------------------------------

        if not components:

            return [
                SplitRegion(
                    region_id=0,
                    x=int(region.x),
                    y=int(region.y),
                    width=region_width,
                    height=region_height,
                    source_region_id=source_id,
                    component_count=1
                )
            ]

        # ----------------------------------------------------
        # First grouping.
        # ----------------------------------------------------

        groups = self.group_components(
            components
        )

        # ----------------------------------------------------
        # Create group boxes.
        # ----------------------------------------------------

        group_boxes = []

        for group in groups:

            bbox = self.group_bbox(
                group
            )

            if not self.valid_group(
                bbox
            ):
                continue

            group_boxes.append(
                (
                    bbox,
                    group
                )
            )

        # ----------------------------------------------------
        # If there is already meaningful separation between
        # components, use those groups.
        # ----------------------------------------------------

        if len(group_boxes) >= 2:

            boxes = []

            for bbox, group in group_boxes:

                padded = self.padded_bbox(
                    bbox,
                    region_width,
                    region_height
                )

                boxes.append(
                    (
                        padded,
                        len(group)
                    )
                )

            # Prevent giant groups from being accidentally
            # fragmented into many tiny pieces.
            if len(boxes) <= 12:

                final_boxes = (
                    self.merge_overlapping_boxes(
                        [
                            box
                            for box, _
                            in boxes
                        ]
                    )
                )

                if len(final_boxes) >= 2:

                    results = []

                    for bbox in final_boxes:

                        x, y, width, height = (
                            bbox
                        )

                        results.append(
                            SplitRegion(
                                region_id=0,
                                x=int(
                                    region.x + x
                                ),
                                y=int(
                                    region.y + y
                                ),
                                width=int(width),
                                height=int(height),
                                source_region_id=source_id,
                                component_count=1
                            )
                        )

                    if results:
                        return results

        # ----------------------------------------------------
        # Projection splitting for very large regions.
        # ----------------------------------------------------

        if region_area >= self.large_region_area:

            projection_boxes = (
                self.split_using_projection(
                    mask
                )
            )

            valid_projection_boxes = []

            for bbox in projection_boxes:

                if self.valid_group(
                    bbox
                ):

                    valid_projection_boxes.append(
                        bbox
                    )

            if len(
                valid_projection_boxes
            ) >= 2:

                valid_projection_boxes = (
                    self.merge_overlapping_boxes(
                        valid_projection_boxes
                    )
                )

                if len(
                    valid_projection_boxes
                ) >= 2:

                    results = []

                    for bbox in valid_projection_boxes:

                        x, y, width, height = (
                            self.padded_bbox(
                                bbox,
                                region_width,
                                region_height
                            )
                        )

                        results.append(
                            SplitRegion(
                                region_id=0,
                                x=int(
                                    region.x + x
                                ),
                                y=int(
                                    region.y + y
                                ),
                                width=int(width),
                                height=int(height),
                                source_region_id=source_id,
                                component_count=1
                            )
                        )

                    if results:
                        return results

        # ----------------------------------------------------
        # Safety fallback.
        # ----------------------------------------------------

        return [
            SplitRegion(
                region_id=0,
                x=int(region.x),
                y=int(region.y),
                width=region_width,
                height=region_height,
                source_region_id=source_id,
                component_count=len(
                    components
                )
            )
        ]

    # ========================================================
    # SPLIT ALL
    # ========================================================

    def split(
        self,
        image: Image.Image,
        regions
    ) -> List[SplitRegion]:

        results = []

        next_id = 1

        for region in regions:

            split_regions = (
                self.split_region(
                    image,
                    region
                )
            )

            for split_region in split_regions:

                split_region.region_id = (
                    next_id
                )

                next_id += 1

                results.append(
                    split_region
                )

        return results

    # ========================================================
    # DICT API
    # ========================================================

    def split_to_dicts(
        self,
        image: Image.Image,
        regions
    ) -> List[Dict[str, Any]]:

        split_regions = self.split(
            image,
            regions
        )

        return [
            region.to_dict()
            for region in split_regions
        ]