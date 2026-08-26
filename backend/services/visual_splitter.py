from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

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
    OCR-aware and classification-aware visual region splitter.

    Pipeline:

        merged visual region
                |
                v
          classification
                |
                v
          OCR protection
                |
                v
          create ink mask
                |
                v
       connected components
                |
                v
       conservative grouping
                |
                v
        projection analysis
                |
                v
        semantic split
                |
                v
       final visual regions

    Design goals:

    1. Do not split ordinary OCR text into characters.
    2. Do not aggressively split handwriting.
    3. Preserve highlights.
    4. Allow diagrams to be split when there is strong evidence.
    5. Keep graphics conservative.
    6. Avoid tiny meaningless regions.
    7. Preserve backward compatibility with:

           splitter.split(image, regions)

       while supporting:

           splitter.split(
               image,
               regions,
               ocr_words=...,
               classifications=...
           )
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        min_component_area: int = 100,
        min_region_area: int = 1400,

        component_gap: int = 18,
        small_component_gap: int = 10,

        padding: int = 8,

        min_width: int = 20,
        min_height: int = 20,

        min_split_gap: int = 22,
        strong_gap: int = 42,

        min_split_ratio: float = 0.018,

        large_region_area: int = 30000,

        # Maximum number of output pieces generated
        # from one source region.
        max_splits_per_region: int = 8,

        # Prevent projection splitting from creating
        # extremely small pieces.
        min_split_piece_area: int = 2500,

        # OCR protection.
        ocr_overlap_protection: float = 0.25,

        # If a region contains enough OCR area,
        # treat it as text-dominated.
        text_dominated_threshold: float = 0.45,

        # Confidence below this is not considered
        # reliable OCR.
        min_ocr_confidence: float = 45.0,

    ):
        self.min_component_area = min_component_area
        self.min_region_area = min_region_area

        self.component_gap = component_gap
        self.small_component_gap = small_component_gap

        self.padding = padding

        self.min_width = min_width
        self.min_height = min_height

        self.min_split_gap = min_split_gap
        self.strong_gap = strong_gap

        self.min_split_ratio = min_split_ratio

        self.large_region_area = large_region_area

        self.max_splits_per_region = (
            max_splits_per_region
        )

        self.min_split_piece_area = (
            min_split_piece_area
        )

        self.ocr_overlap_protection = (
            ocr_overlap_protection
        )

        self.text_dominated_threshold = (
            text_dominated_threshold
        )

        self.min_ocr_confidence = (
            min_ocr_confidence
        )

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
                cv2.COLOR_GRAY2BGR,
            )

        return cv2.cvtColor(
            array,
            cv2.COLOR_RGB2BGR,
        )

    # ========================================================
    # CROP
    # ========================================================

    @staticmethod
    def crop_region(
        image: Image.Image,
        region,
    ) -> Image.Image:

        x = max(
            0,
            int(region.x),
        )

        y = max(
            0,
            int(region.y),
        )

        right = min(
            image.width,
            x + int(region.width),
        )

        bottom = min(
            image.height,
            y + int(region.height),
        )

        return image.crop(
            (
                x,
                y,
                right,
                bottom,
            )
        )

    # ========================================================
    # REGION TYPE
    # ========================================================

    @staticmethod
    def classification_type(
        classification
    ) -> str:

        if classification is None:
            return ""

        if isinstance(
            classification,
            str,
        ):
            return classification.lower()

        if isinstance(
            classification,
            dict,
        ):
            value = (
                classification.get(
                    "label"
                )
                or
                classification.get(
                    "classification"
                )
                or
                classification.get(
                    "region_type"
                )
                or
                classification.get(
                    "type"
                )
            )

            if value:
                return str(
                    value
                ).lower()

        value = (
            getattr(
                classification,
                "label",
                None,
            )
            or
            getattr(
                classification,
                "classification",
                None,
            )
            or
            getattr(
                classification,
                "region_type",
                None,
            )
            or
            getattr(
                classification,
                "type",
                None,
            )
        )

        if value:
            return str(
                value
            ).lower()

        return ""

    # ========================================================
    # CLASSIFICATION LOOKUP
    # ========================================================

    @classmethod
    def get_classification_for_region(
        cls,
        region,
        classifications,
    ):

        if not classifications:
            return None

        region_id = int(
            getattr(
                region,
                "region_id",
                0,
            )
        )

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(
            classifications,
            dict,
        ):

            if region_id in classifications:

                return classifications[
                    region_id
                ]

            if str(region_id) in classifications:

                return classifications[
                    str(region_id)
                ]

        # ----------------------------------------------------
        # List
        # ----------------------------------------------------

        for item in classifications:

            item_region_id = (
                getattr(
                    item,
                    "region_id",
                    None,
                )
            )

            if item_region_id is None:

                if isinstance(
                    item,
                    dict,
                ):

                    item_region_id = (
                        item.get(
                            "region_id"
                        )
                        or
                        item.get(
                            "id"
                        )
                    )

            if (
                item_region_id is not None
                and
                int(item_region_id)
                == region_id
            ):

                return item

        return None

    # ========================================================
    # OCR BBOX
    # ========================================================

    @staticmethod
    def ocr_bbox(
        word: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:

        return (
            int(
                word.get(
                    "left",
                    0,
                )
            ),
            int(
                word.get(
                    "top",
                    0,
                )
            ),
            int(
                word.get(
                    "width",
                    0,
                )
            ),
            int(
                word.get(
                    "height",
                    0,
                )
            ),
        )

    # ========================================================
    # INTERSECTION AREA
    # ========================================================

    @staticmethod
    def bbox_intersection(
        a,
        b,
    ) -> int:

        ax1 = a[0]
        ay1 = a[1]

        ax2 = (
            a[0]
            + a[2]
        )

        ay2 = (
            a[1]
            + a[3]
        )

        bx1 = b[0]
        by1 = b[1]

        bx2 = (
            b[0]
            + b[2]
        )

        by2 = (
            b[1]
            + b[3]
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

        if x2 <= x1:
            return 0

        if y2 <= y1:
            return 0

        return (
            x2 - x1
        ) * (
            y2 - y1
        )

    # ========================================================
    # OCR OVERLAP
    # ========================================================

    def calculate_ocr_overlap(
        self,
        region,
        ocr_words,
    ) -> Tuple[float, int]:

        if not ocr_words:
            return 0.0, 0

        region_bbox = (
            int(region.x),
            int(region.y),
            int(region.width),
            int(region.height),
        )

        region_area = (
            region_bbox[2]
            * region_bbox[3]
        )

        if region_area <= 0:
            return 0.0, 0

        total_ocr_area = 0

        for word in ocr_words:

            confidence = float(
                word.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                confidence
                < self.min_ocr_confidence
            ):
                continue

            bbox = self.ocr_bbox(
                word
            )

            total_ocr_area += (
                self.bbox_intersection(
                    region_bbox,
                    bbox,
                )
            )

        ratio = (
            total_ocr_area
            / float(region_area)
        )

        return (
            min(ratio, 1.0),
            total_ocr_area,
        )

    # ========================================================
    # OCR WORDS INSIDE REGION
    # ========================================================

    def words_inside_region(
        self,
        region,
        ocr_words,
    ) -> List[Dict[str, Any]]:

        if not ocr_words:
            return []

        result = []

        region_bbox = (
            int(region.x),
            int(region.y),
            int(region.width),
            int(region.height),
        )

        for word in ocr_words:

            confidence = float(
                word.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                confidence
                < self.min_ocr_confidence
            ):
                continue

            bbox = self.ocr_bbox(
                word
            )

            intersection = (
                self.bbox_intersection(
                    region_bbox,
                    bbox,
                )
            )

            word_area = (
                bbox[2]
                * bbox[3]
            )

            if word_area <= 0:
                continue

            if (
                intersection
                / float(word_area)
                >= self.ocr_overlap_protection
            ):

                result.append(
                    word
                )

        return result

    # ========================================================
    # CREATE INK MASK
    # ========================================================

    def create_ink_mask(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """
        Create foreground mask.

        Combines:

        - dark ink
        - colored ink
        - morphology

        The mask is intentionally conservative.
        """

        cv_image = self.pil_to_cv(
            image
        )

        gray = cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        # ----------------------------------------------------
        # Dark ink
        # ----------------------------------------------------

        dark_mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )

        # ----------------------------------------------------
        # Colored ink
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2HSV,
        )

        saturation = hsv[:, :, 1]

        color_mask = np.where(
            saturation > 35,
            255,
            0,
        ).astype(
            np.uint8
        )

        mask = cv2.bitwise_or(
            dark_mask,
            color_mask,
        )

        # ----------------------------------------------------
        # Remove tiny isolated noise
        # ----------------------------------------------------

        open_kernel = np.ones(
            (3, 3),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        # ----------------------------------------------------
        # Close tiny stroke gaps
        # ----------------------------------------------------

        close_kernel = np.ones(
            (2, 2),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        return mask

    # ========================================================
    # CONNECTED COMPONENTS
    # ========================================================

    def find_components(
        self,
        mask: np.ndarray,
    ) -> List[
        Tuple[int, int, int, int, int]
    ]:

        (
            num_labels,
            labels,
            stats,
            _,
        ) = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        components = []

        for label in range(
            1,
            num_labels,
        ):

            x = int(
                stats[
                    label,
                    cv2.CC_STAT_LEFT,
                ]
            )

            y = int(
                stats[
                    label,
                    cv2.CC_STAT_TOP,
                ]
            )

            width = int(
                stats[
                    label,
                    cv2.CC_STAT_WIDTH,
                ]
            )

            height = int(
                stats[
                    label,
                    cv2.CC_STAT_HEIGHT,
                ]
            )

            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )

            if (
                area
                < self.min_component_area
            ):
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
                    area,
                )
            )

        return components

    # ========================================================
    # GEOMETRY
    # ========================================================

    @staticmethod
    def horizontal_gap(
        a,
        b,
    ) -> int:

        ax2 = (
            a[0]
            + a[2]
        )

        bx2 = (
            b[0]
            + b[2]
        )

        if ax2 < b[0]:
            return b[0] - ax2

        if bx2 < a[0]:
            return a[0] - bx2

        return 0

    @staticmethod
    def vertical_gap(
        a,
        b,
    ) -> int:

        ay2 = (
            a[1]
            + a[3]
        )

        by2 = (
            b[1]
            + b[3]
        )

        if ay2 < b[1]:
            return b[1] - ay2

        if by2 < a[1]:
            return a[1] - by2

        return 0

    @staticmethod
    def horizontal_alignment(
        a,
        b,
    ) -> float:

        ay2 = (
            a[1]
            + a[3]
        )

        by2 = (
            b[1]
            + b[3]
        )

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
            / float(denominator)
        )

    @staticmethod
    def vertical_alignment(
        a,
        b,
    ) -> float:

        ax2 = (
            a[0]
            + a[2]
        )

        bx2 = (
            b[0]
            + b[2]
        )

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
            / float(denominator)
        )

    @staticmethod
    def intersection_ratio(
        a,
        b,
    ) -> float:

        intersection = (
            VisualRegionSplitter
            .bbox_intersection(
                a,
                b,
            )
        )

        area_a = (
            a[2]
            * a[3]
        )

        area_b = (
            b[2]
            * b[3]
        )

        smaller = min(
            area_a,
            area_b,
        )

        if smaller <= 0:
            return 0.0

        return (
            intersection
            / float(smaller)
        )

    # ========================================================
    # SHOULD GROUP COMPONENTS
    # ========================================================

    def should_merge(
        self,
        a,
        b,
    ) -> bool:

        horizontal = (
            self.horizontal_gap(
                a,
                b,
            )
        )

        vertical = (
            self.vertical_gap(
                a,
                b,
            )
        )

        h_align = (
            self.horizontal_alignment(
                a,
                b,
            )
        )

        v_align = (
            self.vertical_alignment(
                a,
                b,
            )
        )

        overlap = (
            self.intersection_ratio(
                a,
                b,
            )
        )

        min_area = min(
            a[4],
            b[4],
        )

        # Strong overlap.
        if overlap >= 0.30:
            return True

        # Touching components.
        if (
            horizontal == 0
            and vertical == 0
            and (
                h_align >= 0.15
                or
                v_align >= 0.15
            )
        ):
            return True

        # Horizontal neighbours.
        if (
            horizontal <= self.component_gap
            and
            h_align >= 0.40
        ):

            if min_area < 4000:
                return True

            if horizontal <= 6:
                return True

        # Vertical neighbours.
        if (
            vertical <= self.component_gap
            and
            v_align >= 0.40
        ):

            if min_area < 4000:
                return True

            if vertical <= 6:
                return True

        # Tiny component attached to larger content.
        if min_area < 700:

            if (
                horizontal
                <= self.small_component_gap
                and
                h_align >= 0.30
            ):
                return True

            if (
                vertical
                <= self.small_component_gap
                and
                v_align >= 0.30
            ):
                return True

        return False

    # ========================================================
    # COMPONENT GROUP SCORE
    # ========================================================

    def group_relation_score(
        self,
        group_a,
        group_b,
    ) -> float:

        best = 0.0

        for a in group_a:

            for b in group_b:

                if not self.should_merge(
                    a,
                    b,
                ):
                    continue

                horizontal = (
                    self.horizontal_gap(
                        a,
                        b,
                    )
                )

                vertical = (
                    self.vertical_gap(
                        a,
                        b,
                    )
                )

                overlap = (
                    self.intersection_ratio(
                        a,
                        b,
                    )
                )

                h_align = (
                    self.horizontal_alignment(
                        a,
                        b,
                    )
                )

                v_align = (
                    self.vertical_alignment(
                        a,
                        b,
                    )
                )

                score = 0.0

                score += (
                    min(
                        overlap,
                        1.0,
                    )
                    * 0.35
                )

                score += (
                    max(
                        h_align,
                        v_align,
                    )
                    * 0.35
                )

                distance = min(
                    horizontal,
                    vertical,
                )

                if distance <= 2:
                    score += 0.25

                elif distance <= 6:
                    score += 0.20

                elif distance <= 12:
                    score += 0.12

                elif distance <= 18:
                    score += 0.05

                if min(
                    a[4],
                    b[4],
                ) < 700:

                    score += 0.08

                best = max(
                    best,
                    score,
                )

        return min(
            best,
            1.0,
        )

    # ========================================================
    # LOCAL GROUPING
    # ========================================================

    def group_components(
        self,
        components,
    ):

        if not components:
            return []

        groups = [
            [component]
            for component in components
        ]

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
                    len(groups),
                ):

                    score = (
                        self.group_relation_score(
                            groups[i],
                            groups[j],
                        )
                    )

                    if score > best_score:

                        best_score = score
                        best_pair = (
                            i,
                            j,
                        )

            if (
                best_pair is not None
                and
                best_score >= 0.72
            ):

                i, j = best_pair

                groups[i] = (
                    groups[i]
                    +
                    groups[j]
                )

                del groups[j]

                changed = True

        return groups

    # ========================================================
    # GROUP BBOX
    # ========================================================

    @staticmethod
    def group_bbox(
        group,
    ):

        x1 = min(
            c[0]
            for c in group
        )

        y1 = min(
            c[1]
            for c in group
        )

        x2 = max(
            c[0] + c[2]
            for c in group
        )

        y2 = max(
            c[1] + c[3]
            for c in group
        )

        return (
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        )

    # ========================================================
    # VALID GROUP
    # ========================================================

    def valid_group(
        self,
        bbox,
    ) -> bool:

        _, _, width, height = bbox

        area = (
            width
            * height
        )

        if (
            area
            < self.min_region_area
        ):
            return False

        if width < self.min_width:
            return False

        if height < self.min_height:
            return False

        return True

    # ========================================================
    # PADDING
    # ========================================================

    def padded_bbox(
        self,
        bbox,
        region_width,
        region_height,
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
            x,
        )

        y = max(
            0,
            y,
        )

        right = min(
            region_width,
            x + width,
        )

        bottom = min(
            region_height,
            y + height,
        )

        return (
            x,
            y,
            right - x,
            bottom - y,
        )

    # ========================================================
    # PROJECTION GAPS
    # ========================================================

    def projection_gaps(
        self,
        mask: np.ndarray,
    ):

        if mask.size == 0:
            return [], []

        binary = (
            mask > 0
        ).astype(
            np.uint8
        )

        column_ink = (
            binary.sum(
                axis=0
            )
        )

        row_ink = (
            binary.sum(
                axis=1
            )
        )

        height, width = (
            mask.shape[:2]
        )

        column_threshold = max(
            1,
            int(
                height
                * self.min_split_ratio
            ),
        )

        row_threshold = max(
            1,
            int(
                width
                * self.min_split_ratio
            ),
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
                self.min_split_gap,
            )
        )

        vertical_gaps = (
            self.runs_from_boolean(
                empty_rows,
                self.min_split_gap,
            )
        )

        return (
            horizontal_gaps,
            vertical_gaps,
        )

    # ========================================================
    # BOOLEAN RUNS
    # ========================================================

    @staticmethod
    def runs_from_boolean(
        values,
        minimum_length,
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

                    if (
                        length
                        >= minimum_length
                    ):

                        runs.append(
                            (
                                start,
                                index,
                            )
                        )

                    start = None

        if start is not None:

            length = (
                len(values)
                - start
            )

            if (
                length
                >= minimum_length
            ):

                runs.append(
                    (
                        start,
                        len(values),
                    )
                )

        return runs

    # ========================================================
    # STRONG PROJECTION GAPS
    # ========================================================

    def strong_projection_splits(
        self,
        mask: np.ndarray,
    ):

        horizontal, vertical = (
            self.projection_gaps(
                mask
            )
        )

        strong_horizontal = [
            gap
            for gap in horizontal
            if (
                gap[1]
                - gap[0]
            )
            >= self.strong_gap
        ]

        strong_vertical = [
            gap
            for gap in vertical
            if (
                gap[1]
                - gap[0]
            )
            >= self.strong_gap
        ]

        return (
            strong_horizontal,
            strong_vertical,
        )

    # ========================================================
    # SPLIT USING PROJECTION
    # ========================================================

    def split_using_projection(
        self,
        mask: np.ndarray,
    ):

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
            +
            x_cuts
            +
            [width]
        )

        y_boundaries = (
            [0]
            +
            y_cuts
            +
            [height]
        )

        candidates = []

        for yi in range(
            len(y_boundaries) - 1
        ):

            y1 = y_boundaries[yi]
            y2 = y_boundaries[yi + 1]

            for xi in range(
                len(x_boundaries) - 1
            ):

                x1 = x_boundaries[xi]
                x2 = x_boundaries[xi + 1]

                piece_width = (
                    x2 - x1
                )

                piece_height = (
                    y2 - y1
                )

                if (
                    piece_width
                    < self.min_width
                ):
                    continue

                if (
                    piece_height
                    < self.min_height
                ):
                    continue

                submask = mask[
                    y1:y2,
                    x1:x2,
                ]

                ink = cv2.countNonZero(
                    submask
                )

                if (
                    ink
                    < self.min_split_piece_area
                ):
                    continue

                candidates.append(
                    (
                        x1,
                        y1,
                        piece_width,
                        piece_height,
                    )
                )

        return candidates

    # ========================================================
    # MERGE OVERLAPPING BOXES
    # ========================================================

    @staticmethod
    def merge_overlapping_boxes(
        boxes,
    ):

        if not boxes:
            return []

        changed = True

        result = list(
            boxes
        )

        while changed:

            changed = False

            new_result = []

            while result:

                current = result.pop(
                    0
                )

                merged = False

                for index, other in enumerate(
                    result
                ):

                    ax1, ay1, aw, ah = (
                        current
                    )

                    bx1, by1, bw, bh = (
                        other
                    )

                    ax2 = ax1 + aw
                    ay2 = ay1 + ah

                    bx2 = bx1 + bw
                    by2 = by1 + bh

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

                    smaller_area = min(
                        aw * ah,
                        bw * bh,
                    )

                    if (
                        smaller_area > 0
                        and
                        (
                            intersection
                            / float(
                                smaller_area
                            )
                        )
                        >= 0.65
                    ):

                        nx1 = min(
                            ax1,
                            bx1,
                        )

                        ny1 = min(
                            ay1,
                            by1,
                        )

                        nx2 = max(
                            ax2,
                            bx2,
                        )

                        ny2 = max(
                            ay2,
                            by2,
                        )

                        current = (
                            nx1,
                            ny1,
                            nx2 - nx1,
                            ny2 - ny1,
                        )

                        result.pop(
                            index
                        )

                        merged = True
                        changed = True

                        break

                if not merged:

                    new_result.append(
                        current
                    )

            result = new_result

        return result

    # ========================================================
    # SHOULD PROTECT REGION
    # ========================================================

    def should_protect_region(
        self,
        region,
        classification,
        ocr_words,
    ) -> bool:

        label = (
            self.classification_type(
                classification
            )
        )

        # ----------------------------------------------------
        # Explicit semantic protection
        # ----------------------------------------------------

        if label in {
            "highlight",
            "handwriting",
            "text",
            "text_artifact",
        }:

            return True

        # ----------------------------------------------------
        # OCR protection
        # ----------------------------------------------------

        overlap, _ = (
            self.calculate_ocr_overlap(
                region,
                ocr_words,
            )
        )

        if (
            overlap
            >= self.text_dominated_threshold
        ):

            return True

        # ----------------------------------------------------
        # Reliable OCR words strongly contained in region.
        # ----------------------------------------------------

        words = (
            self.words_inside_region(
                region,
                ocr_words,
            )
        )

        if len(words) >= 2:

            return True

        return False

    # ========================================================
    # SHOULD USE PROJECTION
    # ========================================================

    def should_use_projection(
        self,
        region,
        classification,
        ocr_words,
    ) -> bool:

        label = (
            self.classification_type(
                classification
            )
        )

        # Text-like content should remain intact.
        if label in {
            "highlight",
            "handwriting",
            "text",
            "text_artifact",
        }:
            return False

        overlap, _ = (
            self.calculate_ocr_overlap(
                region,
                ocr_words,
            )
        )

        if (
            overlap
            >= self.ocr_overlap_protection
        ):
            return False

        # Graphics are only split conservatively.
        if label == "graphic":

            return (
                region.width
                * region.height
                >= (
                    self.large_region_area
                    * 2
                )
            )

        # Diagrams are the main target.
        if label == "diagram":

            return (
                region.width
                * region.height
                >= self.large_region_area
            )

        # Unknown classification.
        return (
            region.width
            * region.height
            >= (
                self.large_region_area
                * 2
            )
        )

    # ========================================================
    # BUILD RESULT
    # ========================================================

    def make_split_region(
        self,
        region,
        bbox,
        component_count=1,
    ) -> SplitRegion:

        x, y, width, height = bbox

        return SplitRegion(
            region_id=0,
            x=int(
                region.x + x
            ),
            y=int(
                region.y + y
            ),
            width=int(width),
            height=int(height),
            source_region_id=int(
                getattr(
                    region,
                    "region_id",
                    0,
                )
            ),
            component_count=int(
                component_count
            ),
        )

    # ========================================================
    # SPLIT ONE REGION
    # ========================================================

    def split_region(
        self,
        image: Image.Image,
        region,
        ocr_words=None,
        classification=None,
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
                0,
            )
        )

        # ----------------------------------------------------
        # Tiny region -> preserve
        # ----------------------------------------------------

        if (
            region_area
            < self.min_region_area
        ):

            return [
                self.make_split_region(
                    region,
                    (
                        0,
                        0,
                        region_width,
                        region_height,
                    ),
                    1,
                )
            ]

        # ----------------------------------------------------
        # OCR/classification protection
        # ----------------------------------------------------

        if self.should_protect_region(
            region,
            classification,
            ocr_words or [],
        ):

            return [
                self.make_split_region(
                    region,
                    (
                        0,
                        0,
                        region_width,
                        region_height,
                    ),
                    1,
                )
            ]

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        cropped = self.crop_region(
            image,
            region,
        )

        mask = self.create_ink_mask(
            cropped
        )

        components = (
            self.find_components(
                mask
            )
        )

        # ----------------------------------------------------
        # No components
        # ----------------------------------------------------

        if not components:

            return [
                self.make_split_region(
                    region,
                    (
                        0,
                        0,
                        region_width,
                        region_height,
                    ),
                    1,
                )
            ]

        # ----------------------------------------------------
        # Component grouping
        # ----------------------------------------------------

        groups = (
            self.group_components(
                components
            )
        )

        group_boxes = []

        for group in groups:

            bbox = (
                self.group_bbox(
                    group
                )
            )

            if not self.valid_group(
                bbox
            ):
                continue

            group_boxes.append(
                (
                    bbox,
                    group,
                )
            )

        # ----------------------------------------------------
        # Use component grouping only when:
        #
        # - there are few groups
        # - groups are meaningful
        # - region is not text protected
        # ----------------------------------------------------

        if (
            2
            <= len(group_boxes)
            <= self.max_splits_per_region
        ):

            boxes = []

            for bbox, group in group_boxes:

                padded = (
                    self.padded_bbox(
                        bbox,
                        region_width,
                        region_height,
                    )
                )

                if (
                    padded[2]
                    * padded[3]
                    < self.min_split_piece_area
                ):
                    continue

                boxes.append(
                    padded
                )

            boxes = (
                self.merge_overlapping_boxes(
                    boxes
                )
            )

            # Only accept component splitting if
            # the pieces occupy a reasonable amount
            # of the original region.
            if (
                2
                <= len(boxes)
                <= self.max_splits_per_region
            ):

                total_area = sum(
                    b[2] * b[3]
                    for b in boxes
                )

                coverage = (
                    total_area
                    / float(
                        region_area
                    )
                )

                # Avoid fragmented representations.
                if (
                    coverage
                    >= 0.15
                ):

                    results = []

                    for bbox in boxes:

                        results.append(
                            self.make_split_region(
                                region,
                                bbox,
                                1,
                            )
                        )

                    return results

        # ----------------------------------------------------
        # Projection splitting
        #
        # ONLY if classification/OCR allows it.
        # ----------------------------------------------------

        if self.should_use_projection(
            region,
            classification,
            ocr_words or [],
        ):

            projection_boxes = (
                self.split_using_projection(
                    mask
                )
            )

            valid_projection_boxes = []

            for bbox in projection_boxes:

                if not self.valid_group(
                    bbox
                ):
                    continue

                piece_area = (
                    bbox[2]
                    * bbox[3]
                )

                if (
                    piece_area
                    < self.min_split_piece_area
                ):
                    continue

                valid_projection_boxes.append(
                    bbox
                )

            if (
                2
                <= len(
                    valid_projection_boxes
                )
                <= self.max_splits_per_region
            ):

                valid_projection_boxes = (
                    self.merge_overlapping_boxes(
                        valid_projection_boxes
                    )
                )

                if (
                    2
                    <= len(
                        valid_projection_boxes
                    )
                    <= self.max_splits_per_region
                ):

                    results = []

                    for bbox in (
                        valid_projection_boxes
                    ):

                        padded = (
                            self.padded_bbox(
                                bbox,
                                region_width,
                                region_height,
                            )
                        )

                        results.append(
                            self.make_split_region(
                                region,
                                padded,
                                1,
                            )
                        )

                    if results:

                        return results

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        return [
            self.make_split_region(
                region,
                (
                    0,
                    0,
                    region_width,
                    region_height,
                ),
                len(
                    components
                ),
            )
        ]

    # ========================================================
    # SPLIT ALL
    # ========================================================

    def split(
        self,
        image: Image.Image,
        regions,
        ocr_words=None,
        classifications=None,
    ) -> List[SplitRegion]:

        results = []

        next_id = 1

        for region in regions:

            classification = (
                self.get_classification_for_region(
                    region,
                    classifications,
                )
            )

            split_regions = (
                self.split_region(
                    image,
                    region,
                    ocr_words=ocr_words,
                    classification=classification,
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
        regions,
        ocr_words=None,
        classifications=None,
    ) -> List[Dict[str, Any]]:

        split_regions = (
            self.split(
                image,
                regions,
                ocr_words=ocr_words,
                classifications=classifications,
            )
        )

        return [
            region.to_dict()
            for region in split_regions
        ]