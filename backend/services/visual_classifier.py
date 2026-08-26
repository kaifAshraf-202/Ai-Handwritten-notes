from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import cv2
import numpy as np


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class VisualClassification:
    """
    Classification result for a merged visual region.

    Confidence is stored internally in the range:
        0.0 -> 1.0

    Example:
        0.95 = 95%
    """

    region_id: int

    classification: str

    confidence: float

    x: int
    y: int
    width: int
    height: int

    area: int

    ocr_overlap_ratio: float

    visual_ink_ratio: float

    color_ratio: float

    yellow_ratio: float

    pink_ratio: float

    edge_ratio: float

    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# VISUAL CANDIDATE CLASSIFIER
# ============================================================

class VisualCandidateClassifier:
    """
    Conservative heuristic classifier for merged visual regions.

    Supported classifications:

        diagram
        handwriting
        highlight
        annotation
        graphic
        text_artifact
        unknown

    Important principle:

        A visual contour is NOT automatically a diagram.

    A chemistry diagram usually contains structural evidence such as:

        - connected lines
        - bonds
        - arrows
        - rings
        - symbols
        - labels
        - geometric structure

    Therefore classification uses multiple signals:

        OCR overlap
        visual ink
        colour
        yellow
        pink
        edges
        region size
        aspect ratio
    """

    def __init__(
        self,
        text_overlap_threshold: float = 0.75,

        color_threshold: float = 0.05,

        yellow_threshold: float = 0.04,

        pink_threshold: float = 0.04,

        visual_threshold: float = 0.04,

        edge_threshold: float = 0.025,

        # Minimum area for a real diagram.
        min_visual_area: int = 1200,

        # More conservative threshold.
        min_diagram_area: int = 5000,

        # Very small regions are rarely complete diagrams.
        small_region_area: int = 8000,

        # Large regions are more likely to contain
        # meaningful visual structures.
        large_region_area: int = 30000,
    ):

        self.text_overlap_threshold = (
            text_overlap_threshold
        )

        self.color_threshold = (
            color_threshold
        )

        self.yellow_threshold = (
            yellow_threshold
        )

        self.pink_threshold = (
            pink_threshold
        )

        self.visual_threshold = (
            visual_threshold
        )

        self.edge_threshold = (
            edge_threshold
        )

        self.min_visual_area = (
            min_visual_area
        )

        self.min_diagram_area = (
            min_diagram_area
        )

        self.small_region_area = (
            small_region_area
        )

        self.large_region_area = (
            large_region_area
        )

    # ========================================================
    # GENERIC REGION ACCESS
    # ========================================================

    @staticmethod
    def _get_region_value(
        region,
        key: str,
        default=0
    ):

        if isinstance(region, dict):

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
    # CREATE RESULT
    # ========================================================

    @staticmethod
    def _result(
        region_id,
        classification,
        confidence,
        x,
        y,
        width,
        height,
        area,
        ocr_overlap_ratio,
        visual_ink_ratio,
        color_ratio,
        yellow_ratio,
        pink_ratio,
        edge_ratio,
        reason,
    ):

        return VisualClassification(
            region_id=region_id,
            classification=classification,
            confidence=float(confidence),
            x=x,
            y=y,
            width=width,
            height=height,
            area=area,
            ocr_overlap_ratio=ocr_overlap_ratio,
            visual_ink_ratio=visual_ink_ratio,
            color_ratio=color_ratio,
            yellow_ratio=yellow_ratio,
            pink_ratio=pink_ratio,
            edge_ratio=edge_ratio,
            reason=reason,
        )

    # ========================================================
    # OCR OVERLAP
    # ========================================================

    def calculate_ocr_overlap(
        self,
        region,
        ocr_words: List[Dict[str, Any]]
    ) -> float:

        x = int(
            self._get_region_value(
                region,
                "x"
            )
        )

        y = int(
            self._get_region_value(
                region,
                "y"
            )
        )

        width = int(
            self._get_region_value(
                region,
                "width"
            )
        )

        height = int(
            self._get_region_value(
                region,
                "height"
            )
        )

        if width <= 0 or height <= 0:
            return 0.0

        region_area = (
            width * height
        )

        rx1 = x
        ry1 = y
        rx2 = x + width
        ry2 = y + height

        overlap_mask = np.zeros(
            (
                height,
                width
            ),
            dtype=np.uint8
        )

        for word in ocr_words:

            confidence = float(
                word.get(
                    "confidence",
                    0
                )
            )

            # Only reliable OCR contributes.
            if confidence < 70:
                continue

            wx = int(
                word.get(
                    "left",
                    0
                )
            )

            wy = int(
                word.get(
                    "top",
                    0
                )
            )

            ww = int(
                word.get(
                    "width",
                    0
                )
            )

            wh = int(
                word.get(
                    "height",
                    0
                )
            )

            if ww <= 0 or wh <= 0:
                continue

            wx1 = wx
            wy1 = wy
            wx2 = wx + ww
            wy2 = wy + wh

            ix1 = max(
                rx1,
                wx1
            )

            iy1 = max(
                ry1,
                wy1
            )

            ix2 = min(
                rx2,
                wx2
            )

            iy2 = min(
                ry2,
                wy2
            )

            if ix2 <= ix1:
                continue

            if iy2 <= iy1:
                continue

            local_x1 = (
                ix1 - rx1
            )

            local_y1 = (
                iy1 - ry1
            )

            local_x2 = (
                ix2 - rx1
            )

            local_y2 = (
                iy2 - ry1
            )

            cv2.rectangle(
                overlap_mask,
                (
                    local_x1,
                    local_y1
                ),
                (
                    local_x2,
                    local_y2
                ),
                255,
                -1
            )

        overlap_pixels = np.count_nonzero(
            overlap_mask
        )

        return float(
            overlap_pixels
            / region_area
        )

    # ========================================================
    # IMAGE METRICS
    # ========================================================

    def calculate_image_metrics(
        self,
        image,
        region
    ):

        x = int(
            self._get_region_value(
                region,
                "x"
            )
        )

        y = int(
            self._get_region_value(
                region,
                "y"
            )
        )

        width = int(
            self._get_region_value(
                region,
                "width"
            )
        )

        height = int(
            self._get_region_value(
                region,
                "height"
            )
        )

        image_array = np.array(
            image
        )

        image_height, image_width = (
            image_array.shape[:2]
        )

        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

        x2 = min(
            image_width,
            x + width
        )

        y2 = min(
            image_height,
            y + height
        )

        if x2 <= x1 or y2 <= y1:

            return {
                "visual_ink_ratio": 0.0,
                "color_ratio": 0.0,
                "yellow_ratio": 0.0,
                "pink_ratio": 0.0,
                "edge_ratio": 0.0,
            }

        crop = image_array[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            return {
                "visual_ink_ratio": 0.0,
                "color_ratio": 0.0,
                "yellow_ratio": 0.0,
                "pink_ratio": 0.0,
                "edge_ratio": 0.0,
            }

        crop_rgb = crop[:, :, :3]

        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            crop_rgb,
            cv2.COLOR_RGB2GRAY
        )

        # ----------------------------------------------------
        # BACKGROUND-AWARE INK
        # ----------------------------------------------------

        median_brightness = float(
            np.median(gray)
        )

        if median_brightness < 128:

            visual_mask = (
                gray > 45
            )

        else:

            visual_mask = (
                gray < 220
            )

        visual_ink_ratio = (
            np.count_nonzero(
                visual_mask
            )
            / visual_mask.size
        )

        # ----------------------------------------------------
        # HSV
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            crop_rgb,
            cv2.COLOR_RGB2HSV
        )

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # ----------------------------------------------------
        # COLOUR
        # ----------------------------------------------------

        colored_pixels = (
            (saturation > 70)
            &
            (value > 50)
        )

        color_ratio = (
            np.count_nonzero(
                colored_pixels
            )
            / colored_pixels.size
        )

        # ----------------------------------------------------
        # YELLOW
        # ----------------------------------------------------

        yellow_mask = (
            (hue >= 15)
            &
            (hue <= 45)
            &
            (saturation >= 70)
            &
            (value >= 80)
        )

        yellow_ratio = (
            np.count_nonzero(
                yellow_mask
            )
            / yellow_mask.size
        )

        # ----------------------------------------------------
        # PINK / MAGENTA
        # ----------------------------------------------------

        pink_mask = (
            (
                (hue >= 140)
                &
                (hue <= 179)
            )
            |
            (
                (hue >= 0)
                &
                (hue <= 10)
            )
        ) & (
            saturation >= 70
        ) & (
            value >= 80
        )

        pink_ratio = (
            np.count_nonzero(
                pink_mask
            )
            / pink_mask.size
        )

        # ----------------------------------------------------
        # EDGE
        # ----------------------------------------------------

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        edge_ratio = (
            np.count_nonzero(
                edges
            )
            / edges.size
        )

        return {
            "visual_ink_ratio": float(
                visual_ink_ratio
            ),

            "color_ratio": float(
                color_ratio
            ),

            "yellow_ratio": float(
                yellow_ratio
            ),

            "pink_ratio": float(
                pink_ratio
            ),

            "edge_ratio": float(
                edge_ratio
            ),
        }

    # ========================================================
    # CLASSIFY REGION
    # ========================================================

    def classify_region(
        self,
        image,
        region,
        ocr_words
    ) -> VisualClassification:

        region_id = int(
            self._get_region_value(
                region,
                "region_id",
                0
            )
        )

        x = int(
            self._get_region_value(
                region,
                "x",
                0
            )
        )

        y = int(
            self._get_region_value(
                region,
                "y",
                0
            )
        )

        width = int(
            self._get_region_value(
                region,
                "width",
                0
            )
        )

        height = int(
            self._get_region_value(
                region,
                "height",
                0
            )
        )

        area = (
            width * height
        )

        # Avoid invalid geometry.
        if width <= 0 or height <= 0:

            return self._result(
                region_id,
                "unknown",
                0.50,
                x,
                y,
                width,
                height,
                area,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                "Invalid region geometry."
            )

        aspect_ratio = (
            max(width, height)
            / max(1, min(width, height))
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_overlap_ratio = (
            self.calculate_ocr_overlap(
                region,
                ocr_words
            )
        )

        # ----------------------------------------------------
        # IMAGE METRICS
        # ----------------------------------------------------

        metrics = (
            self.calculate_image_metrics(
                image,
                region
            )
        )

        visual_ink_ratio = (
            metrics["visual_ink_ratio"]
        )

        color_ratio = (
            metrics["color_ratio"]
        )

        yellow_ratio = (
            metrics["yellow_ratio"]
        )

        pink_ratio = (
            metrics["pink_ratio"]
        )

        edge_ratio = (
            metrics["edge_ratio"]
        )

        # ====================================================
        # RULE 1
        # YELLOW → HIGHLIGHT
        # ====================================================

        if (
            yellow_ratio
            >= self.yellow_threshold
        ):

            return self._result(
                region_id,
                "highlight",
                0.95,
                x,
                y,
                width,
                height,
                area,
                ocr_overlap_ratio,
                visual_ink_ratio,
                color_ratio,
                yellow_ratio,
                pink_ratio,
                edge_ratio,
                "Strong yellow/golden colour detected."
            )

        # ====================================================
        # RULE 2
        # PINK → HANDWRITING / ANNOTATION
        # ====================================================

        if (
            pink_ratio
            >= self.pink_threshold
        ):

            # Thin horizontal mark.
            if (
                height <= 45
                and width >= 80
            ):

                return self._result(
                    region_id,
                    "annotation",
                    0.91,
                    x,
                    y,
                    width,
                    height,
                    area,
                    ocr_overlap_ratio,
                    visual_ink_ratio,
                    color_ratio,
                    yellow_ratio,
                    pink_ratio,
                    edge_ratio,
                    "Pink/magenta coloured thin annotation or underline."
                )

            return self._result(
                region_id,
                "handwriting",
                0.90,
                x,
                y,
                width,
                height,
                area,
                ocr_overlap_ratio,
                visual_ink_ratio,
                color_ratio,
                yellow_ratio,
                pink_ratio,
                edge_ratio,
                "Strong pink/magenta handwritten or coloured annotation content."
            )

        # ====================================================
        # RULE 3
        # VERY SMALL REGION
        # ====================================================

        if area < self.min_visual_area:

            # Small regions with OCR are usually
            # text fragments / OCR artifacts.
            if (
                ocr_overlap_ratio
                >= 0.35
            ):

                return self._result(
                    region_id,
                    "text_artifact",
                    0.92,
                    x,
                    y,
                    width,
                    height,
                    area,
                    ocr_overlap_ratio,
                    visual_ink_ratio,
                    color_ratio,
                    yellow_ratio,
                    pink_ratio,
                    edge_ratio,
                    "Small region associated with OCR text."
                )

            # Small black graphical pieces should NOT
            # automatically become diagrams.
            if (
                edge_ratio
                >= self.edge_threshold
            ):

                return self._result(
                    region_id,
                    "graphic",
                    0.72,
                    x,
                    y,
                    width,
                    height,
                    area,
                    ocr_overlap_ratio,
                    visual_ink_ratio,
                    color_ratio,
                    yellow_ratio,
                    pink_ratio,
                    edge_ratio,
                    "Small graphical structure detected; insufficient evidence for a complete diagram."
                )

        # ====================================================
        # RULE 4
        # HIGH OCR OVERLAP
        # ====================================================

        if (
            ocr_overlap_ratio
            >= self.text_overlap_threshold
        ):

            # If OCR dominates and there isn't strong
            # structural evidence, preserve as text.
            if not (
                edge_ratio
                >= self.edge_threshold * 1.5
                and
                visual_ink_ratio
                >= self.visual_threshold
                and
                area
                >= self.min_diagram_area
            ):

                return self._result(
                    region_id,
                    "text_artifact",
                    0.90,
                    x,
                    y,
                    width,
                    height,
                    area,
                    ocr_overlap_ratio,
                    visual_ink_ratio,
                    color_ratio,
                    yellow_ratio,
                    pink_ratio,
                    edge_ratio,
                    "Region is dominated by reliable OCR."
                )

            # OCR + strong structure + sufficient size.
            return self._result(
                region_id,
                "diagram",
                0.82,
                x,
                y,
                width,
                height,
                area,
                ocr_overlap_ratio,
                visual_ink_ratio,
                color_ratio,
                yellow_ratio,
                pink_ratio,
                edge_ratio,
                "Reliable OCR is present together with substantial structural graphical evidence."
            )

        # ====================================================
        # RULE 5
        # CLEAR DIAGRAM
        # ====================================================

        strong_structure = (
            edge_ratio
            >= self.edge_threshold
        )

        enough_ink = (
            visual_ink_ratio
            >= self.visual_threshold
        )

        enough_size = (
            area
            >= self.min_diagram_area
        )

        # A real diagram should generally have
        # meaningful dimensions rather than being
        # a tiny isolated contour.
        reasonable_shape = (
            width >= 60
            and height >= 40
        )

        # Very thin regions are usually lines,
        # underlines, borders, or artifacts.
        not_too_thin = (
            aspect_ratio <= 12
        )

        if (
            strong_structure
            and enough_ink
            and enough_size
            and reasonable_shape
            and not_too_thin
        ):

            # Larger regions get stronger confidence.
            if area >= self.large_region_area:

                confidence = 0.90

            else:

                confidence = 0.84

            return self._result(
                region_id,
                "diagram",
                confidence,
                x,
                y,
                width,
                height,
                area,
                ocr_overlap_ratio,
                visual_ink_ratio,
                color_ratio,
                yellow_ratio,
                pink_ratio,
                edge_ratio,
                "Region has sufficient size and structural graphical evidence for a diagram."
            )

        # ====================================================
        # RULE 6
        # MEDIUM GRAPHIC
        # ====================================================

        if (
            strong_structure
            and enough_ink
            and area >= self.min_visual_area
        ):

            return self._result(
                region_id,
                "graphic",
                0.75,
                x,
                y,
                width,
                height,
                area,
                ocr_overlap_ratio,
                visual_ink_ratio,
                color_ratio,
                yellow_ratio,
                pink_ratio,
                edge_ratio,
                "Graphical content detected, but evidence is insufficient for a full diagram."
            )

        # ====================================================
        # RULE 7
        # COLOURED VISUAL CONTENT
        # ====================================================

        if (
            color_ratio
            >= self.color_threshold
        ):

            return self._result(
                region_id,
                "graphic",
                0.74,
                x,
                y,
                width,
                height,
                area,
                ocr_overlap_ratio,
                visual_ink_ratio,
                color_ratio,
                yellow_ratio,
                pink_ratio,
                edge_ratio,
                "Coloured visual content detected."
            )

        # ====================================================
        # RULE 8
        # GENERAL VISUAL CONTENT
        # ====================================================

        if (
            visual_ink_ratio
            >= self.visual_threshold
        ):

            return self._result(
                region_id,
                "graphic",
                0.68,
                x,
                y,
                width,
                height,
                area,
                ocr_overlap_ratio,
                visual_ink_ratio,
                color_ratio,
                yellow_ratio,
                pink_ratio,
                edge_ratio,
                "Meaningful non-text visual content detected."
            )

        # ====================================================
        # RULE 9
        # UNKNOWN
        # ====================================================

        return self._result(
            region_id,
            "unknown",
            0.50,
            x,
            y,
            width,
            height,
            area,
            ocr_overlap_ratio,
            visual_ink_ratio,
            color_ratio,
            yellow_ratio,
            pink_ratio,
            edge_ratio,
            "Insufficient evidence for stronger classification."
        )

    # ========================================================
    # CLASSIFY ALL
    # ========================================================

    def classify(
        self,
        image,
        regions,
        ocr_words
    ) -> List[VisualClassification]:

        results = []

        for region in regions:

            result = (
                self.classify_region(
                    image,
                    region,
                    ocr_words
                )
            )

            results.append(
                result
            )

        return results