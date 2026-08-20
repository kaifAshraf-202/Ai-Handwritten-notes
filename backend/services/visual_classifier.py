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
    Heuristic classifier for merged visual regions.

    Classification types:

        diagram
        handwriting
        highlight
        annotation
        graphic
        text_artifact
        unknown

    Important design principle:

        We NEVER discard a region merely because it
        contains OCR.

    Chemistry diagrams often contain:

        - labels
        - symbols
        - numbers
        - equations
        - arrows
        - printed text

    Therefore OCR overlap is only one signal.
    """

    def __init__(
        self,

        text_overlap_threshold: float = 0.75,

        # Minimum coloured-pixel ratio.
        color_threshold: float = 0.05,

        # Yellow highlighting threshold.
        yellow_threshold: float = 0.04,

        # Pink/magenta handwriting threshold.
        pink_threshold: float = 0.04,

        visual_threshold: float = 0.04,

        edge_threshold: float = 0.025,

        min_visual_area: int = 1200,
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

    # ========================================================
    # GENERIC REGION ACCESS
    # ========================================================

    @staticmethod
    def _get_region_value(
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

            # Only reliable OCR is used for the
            # text-overlap calculation.
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

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        crop_rgb = crop[:, :, :3]

        # ----------------------------------------------------
        # Grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            crop_rgb,
            cv2.COLOR_RGB2GRAY
        )

        # ====================================================
        # BACKGROUND-AWARE INK DETECTION
        # ====================================================

        median_brightness = float(
            np.median(gray)
        )

        if median_brightness < 128:

            # Dark page background.
            #
            # Ink/text is brighter than background.
            #
            visual_mask = (
                gray > 45
            )

        else:

            # Light page background.
            #
            # Ink/text is darker than background.
            #
            visual_mask = (
                gray < 220
            )

        visual_ink_ratio = (
            np.count_nonzero(
                visual_mask
            )
            / visual_mask.size
        )

        # ====================================================
        # HSV
        # ====================================================

        hsv = cv2.cvtColor(
            crop_rgb,
            cv2.COLOR_RGB2HSV
        )

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # ----------------------------------------------------
        # Coloured pixels
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

        # ====================================================
        # YELLOW DETECTION
        # ====================================================

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

        # ====================================================
        # PINK / MAGENTA DETECTION
        # ====================================================

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

        # ====================================================
        # EDGE DETECTION
        # ====================================================

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

        # ----------------------------------------------------
        # OCR overlap
        # ----------------------------------------------------

        ocr_overlap_ratio = (
            self.calculate_ocr_overlap(
                region,
                ocr_words
            )
        )

        # ----------------------------------------------------
        # Visual metrics
        # ----------------------------------------------------

        metrics = (
            self.calculate_image_metrics(
                image,
                region
            )
        )

        visual_ink_ratio = metrics[
            "visual_ink_ratio"
        ]

        color_ratio = metrics[
            "color_ratio"
        ]

        yellow_ratio = metrics[
            "yellow_ratio"
        ]

        pink_ratio = metrics[
            "pink_ratio"
        ]

        edge_ratio = metrics[
            "edge_ratio"
        ]

        # ====================================================
        # RULE 1
        # YELLOW → HIGHLIGHT
        # ====================================================

        if (
            yellow_ratio
            >= self.yellow_threshold
        ):

            return VisualClassification(

                region_id=region_id,

                classification="highlight",

                confidence=0.95,

                x=x,
                y=y,
                width=width,
                height=height,

                area=area,

                ocr_overlap_ratio=(
                    ocr_overlap_ratio
                ),

                visual_ink_ratio=(
                    visual_ink_ratio
                ),

                color_ratio=(
                    color_ratio
                ),

                yellow_ratio=(
                    yellow_ratio
                ),

                pink_ratio=(
                    pink_ratio
                ),

                edge_ratio=(
                    edge_ratio
                ),

                reason=(
                    "Strong yellow/golden "
                    "colour detected."
                ),
            )

        # ====================================================
        # RULE 2
        # PINK → HANDWRITING / ANNOTATION
        # ====================================================

        if (
            pink_ratio
            >= self.pink_threshold
        ):

            # Very thin pink region is more likely an
            # underline/annotation than handwriting.
            if (
                height <= 45
                and
                width >= 80
            ):

                return VisualClassification(

                    region_id=region_id,

                    classification="annotation",

                    confidence=0.91,

                    x=x,
                    y=y,
                    width=width,
                    height=height,

                    area=area,

                    ocr_overlap_ratio=(
                        ocr_overlap_ratio
                    ),

                    visual_ink_ratio=(
                        visual_ink_ratio
                    ),

                    color_ratio=(
                        color_ratio
                    ),

                    yellow_ratio=(
                        yellow_ratio
                    ),

                    pink_ratio=(
                        pink_ratio
                    ),

                    edge_ratio=(
                        edge_ratio
                    ),

                    reason=(
                        "Pink/magenta coloured "
                        "thin annotation or "
                        "underline."
                    ),
                )

            return VisualClassification(

                region_id=region_id,

                classification="handwriting",

                confidence=0.90,

                x=x,
                y=y,
                width=width,
                height=height,

                area=area,

                ocr_overlap_ratio=(
                    ocr_overlap_ratio
                ),

                visual_ink_ratio=(
                    visual_ink_ratio
                ),

                color_ratio=(
                    color_ratio
                ),

                yellow_ratio=(
                    yellow_ratio
                ),

                pink_ratio=(
                    pink_ratio
                ),

                edge_ratio=(
                    edge_ratio
                ),

                reason=(
                    "Strong pink/magenta "
                    "handwritten or coloured "
                    "annotation content."
                ),
            )

        # ====================================================
        # RULE 3
        # SMALL OCR ARTIFACT
        # ====================================================

        if (
            area
            < self.min_visual_area
        ):

            if (
                ocr_overlap_ratio
                >= 0.50
            ):

                return VisualClassification(

                    region_id=region_id,

                    classification="text_artifact",

                    confidence=0.96,

                    x=x,
                    y=y,
                    width=width,
                    height=height,

                    area=area,

                    ocr_overlap_ratio=(
                        ocr_overlap_ratio
                    ),

                    visual_ink_ratio=(
                        visual_ink_ratio
                    ),

                    color_ratio=(
                        color_ratio
                    ),

                    yellow_ratio=(
                        yellow_ratio
                    ),

                    pink_ratio=(
                        pink_ratio
                    ),

                    edge_ratio=(
                        edge_ratio
                    ),

                    reason=(
                        "Small region with "
                        "strong OCR overlap."
                    ),
                )

        # ====================================================
        # RULE 4
        # MOSTLY OCR BUT STRUCTURAL
        # ====================================================

        if (
            ocr_overlap_ratio
            >= self.text_overlap_threshold
        ):

            if (
                edge_ratio
                >= self.edge_threshold
                and
                visual_ink_ratio
                >= self.visual_threshold
            ):

                return VisualClassification(

                    region_id=region_id,

                    classification="diagram",

                    confidence=0.80,

                    x=x,
                    y=y,
                    width=width,
                    height=height,

                    area=area,

                    ocr_overlap_ratio=(
                        ocr_overlap_ratio
                    ),

                    visual_ink_ratio=(
                        visual_ink_ratio
                    ),

                    color_ratio=(
                        color_ratio
                    ),

                    yellow_ratio=(
                        yellow_ratio
                    ),

                    pink_ratio=(
                        pink_ratio
                    ),

                    edge_ratio=(
                        edge_ratio
                    ),

                    reason=(
                        "Contains OCR but also "
                        "contains structural "
                        "graphical content."
                    ),
                )

            return VisualClassification(

                region_id=region_id,

                classification="text_artifact",

                confidence=0.90,

                x=x,
                y=y,
                width=width,
                height=height,

                area=area,

                ocr_overlap_ratio=(
                    ocr_overlap_ratio
                ),

                visual_ink_ratio=(
                    visual_ink_ratio
                ),

                color_ratio=(
                    color_ratio
                ),

                yellow_ratio=(
                    yellow_ratio
                ),

                pink_ratio=(
                    pink_ratio
                ),

                edge_ratio=(
                    edge_ratio
                ),

                reason=(
                    "Region is dominated "
                    "by reliable OCR."
                ),
            )

        # ====================================================
        # RULE 5
        # LARGE STRUCTURAL GRAPHIC
        # ====================================================

        if (
            area >= self.min_visual_area
            and
            edge_ratio
            >= self.edge_threshold
        ):

            return VisualClassification(

                region_id=region_id,

                classification="diagram",

                confidence=0.84,

                x=x,
                y=y,
                width=width,
                height=height,

                area=area,

                ocr_overlap_ratio=(
                    ocr_overlap_ratio
                ),

                visual_ink_ratio=(
                    visual_ink_ratio
                ),

                color_ratio=(
                    color_ratio
                ),

                yellow_ratio=(
                    yellow_ratio
                ),

                pink_ratio=(
                    pink_ratio
                ),

                edge_ratio=(
                    edge_ratio
                ),

                reason=(
                    "Large region containing "
                    "structural graphical "
                    "content."
                ),
            )

        # ====================================================
        # RULE 6
        # GRAPHIC
        # ====================================================

        if (
            visual_ink_ratio
            >= self.visual_threshold
        ):

            return VisualClassification(

                region_id=region_id,

                classification="graphic",

                confidence=0.70,

                x=x,
                y=y,
                width=width,
                height=height,

                area=area,

                ocr_overlap_ratio=(
                    ocr_overlap_ratio
                ),

                visual_ink_ratio=(
                    visual_ink_ratio
                ),

                color_ratio=(
                    color_ratio
                ),

                yellow_ratio=(
                    yellow_ratio
                ),

                pink_ratio=(
                    pink_ratio
                ),

                edge_ratio=(
                    edge_ratio
                ),

                reason=(
                    "Meaningful non-text "
                    "visual content detected."
                ),
            )

        # ====================================================
        # RULE 7
        # UNKNOWN
        # ====================================================

        return VisualClassification(

            region_id=region_id,

            classification="unknown",

            confidence=0.50,

            x=x,
            y=y,
            width=width,
            height=height,

            area=area,

            ocr_overlap_ratio=(
                ocr_overlap_ratio
            ),

            visual_ink_ratio=(
                visual_ink_ratio
            ),

            color_ratio=(
                color_ratio
            ),

            yellow_ratio=(
                yellow_ratio
            ),

            pink_ratio=(
                pink_ratio
            ),

            edge_ratio=(
                edge_ratio
            ),

            reason=(
                "Insufficient evidence "
                "for stronger classification."
            ),
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