from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import cv2
import numpy as np


# ============================================================
# REGION DATA MODEL
# ============================================================

@dataclass
class Region:
    """
    Represents a detected region on a PDF page.

    Coordinates:
        x, y = top-left corner
        width, height = region dimensions
    """

    region_id: int
    region_type: str

    x: int
    y: int
    width: int
    height: int

    confidence: float = 0.0

    source: str = "unknown"

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
# REGION DETECTOR
# ============================================================

class RegionDetector:

    """
    Detects non-text visual regions from a PDF page.

    Pipeline:

        PDF image
            ↓
        OCR words
            ↓
        Reliable OCR mask
            ↓
        OpenCV threshold
            ↓
        Remove text
            ↓
        Find graphical contours
            ↓
        Visual candidates

    IMPORTANT:

    Reliable OCR is removed from visual detection.

    Low-confidence OCR is intentionally NOT removed because
    it may actually represent:
        - diagrams
        - chemical symbols
        - formulas
        - arrows
        - labels
        - handwritten annotations
    """

    def __init__(
        self,
        min_area: int = 500,
        min_width: int = 20,
        min_height: int = 20,

        # OCR confidence above which text is considered
        # reliable enough to mask.
        reliable_ocr_confidence: float = 70.0,

        # Extra padding around reliable OCR regions.
        # This prevents letters from leaving tiny contours.
        ocr_mask_padding: int = 6,

        # Morphological settings.
        morphology_kernel_size: int = 3,

        # Contours that are extremely large compared with
        # the page are usually page-level artifacts.
        max_page_area_ratio: float = 0.45,
    ):

        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height

        self.reliable_ocr_confidence = (
            reliable_ocr_confidence
        )

        self.ocr_mask_padding = (
            ocr_mask_padding
        )

        self.morphology_kernel_size = (
            morphology_kernel_size
        )

        self.max_page_area_ratio = (
            max_page_area_ratio
        )

    # ========================================================
    # PIL → OpenCV
    # ========================================================

    @staticmethod
    def pil_to_cv(image):
        """
        Convert a PIL RGB image into OpenCV BGR.
        """

        image_array = np.array(
            image
        )

        return cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

    # ========================================================
    # OCR MASK
    # ========================================================

    def build_reliable_ocr_mask(
        self,
        image,
        ocr_words: List[Dict[str, Any]]
    ):
        """
        Build a binary mask containing reliable OCR text.

        White pixels = reliable text
        Black pixels = everything else

        Only sufficiently reliable OCR is masked.

        Short/isolated OCR tokens are treated carefully because
        chemistry pages frequently contain symbols such as:

            Sn
            Cn
            Ci
            n=
            2

        These may be meaningful diagram labels.
        """

        height = image.height
        width = image.width

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for word in ocr_words:

            text = str(
                word.get(
                    "text",
                    ""
                )
            ).strip()

            if not text:
                continue

            confidence = float(
                word.get(
                    "confidence",
                    0
                )
            )

            if confidence < (
                self.reliable_ocr_confidence
            ):
                continue

            # ------------------------------------------------
            # Coordinates
            # ------------------------------------------------

            x = int(
                word.get(
                    "left",
                    0
                )
            )

            y = int(
                word.get(
                    "top",
                    0
                )
            )

            word_width = int(
                word.get(
                    "width",
                    0
                )
            )

            word_height = int(
                word.get(
                    "height",
                    0
                )
            )

            if word_width <= 0:
                continue

            if word_height <= 0:
                continue

            # ------------------------------------------------
            # Padding
            # ------------------------------------------------

            padding = (
                self.ocr_mask_padding
            )

            x1 = max(
                0,
                x - padding
            )

            y1 = max(
                0,
                y - padding
            )

            x2 = min(
                width,
                x + word_width + padding
            )

            y2 = min(
                height,
                y + word_height + padding
            )

            # ------------------------------------------------
            # Draw text mask
            # ------------------------------------------------

            cv2.rectangle(
                mask,
                (x1, y1),
                (x2, y2),
                255,
                thickness=-1
            )

        return mask

    # ========================================================
    # DETECT VISUAL CONTOURS
    # ========================================================

    def detect_contours(
        self,
        image,
        ocr_words: List[Dict[str, Any]] = None
    ) -> List[Region]:
        """
        Detect visual candidates after masking reliable OCR.

        The critical difference from the previous version is:

            OCR → mask → OpenCV

        rather than:

            OpenCV on the entire page
        """

        if ocr_words is None:
            ocr_words = []

        cv_image = self.pil_to_cv(
            image
        )

        page_height, page_width = (
            cv_image.shape[:2]
        )

        page_area = (
            page_width
            * page_height
        )

        # ----------------------------------------------------
        # Grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            cv_image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Slight blur
        # ----------------------------------------------------

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # ----------------------------------------------------
        # Adaptive threshold
        # ----------------------------------------------------

        threshold = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15
        )

        # ----------------------------------------------------
        # Build reliable OCR mask
        # ----------------------------------------------------

        ocr_mask = (
            self.build_reliable_ocr_mask(
                image,
                ocr_words
            )
        )

        # ----------------------------------------------------
        # Remove reliable OCR from threshold
        #
        # threshold:
        #     white = detected foreground
        #
        # ocr_mask:
        #     white = reliable OCR
        #
        # Result:
        #     white = visual candidates
        # ----------------------------------------------------

        threshold[
            ocr_mask > 0
        ] = 0

        # ----------------------------------------------------
        # Morphological cleanup
        # ----------------------------------------------------

        kernel_size = (
            self.morphology_kernel_size
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                kernel_size,
                kernel_size
            )
        )

        # Remove tiny isolated noise.
        cleaned = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        # ----------------------------------------------------
        # Find contours
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []

        for contour in contours:

            x, y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )

            area = (
                width
                * height
            )

            # ------------------------------------------------
            # Basic filtering
            # ------------------------------------------------

            if area < self.min_area:
                continue

            if width < self.min_width:
                continue

            if height < self.min_height:
                continue

            # ------------------------------------------------
            # Reject page-sized artifacts
            # ------------------------------------------------

            area_ratio = (
                area / page_area
            )

            if (
                area_ratio
                > self.max_page_area_ratio
            ):
                continue

            regions.append(
                Region(
                    region_id=0,
                    region_type="visual_candidate",

                    x=x,
                    y=y,
                    width=width,
                    height=height,

                    confidence=0.0,

                    source="opencv_contour",
                )
            )

        # ----------------------------------------------------
        # Sort top → bottom, left → right
        # ----------------------------------------------------

        regions.sort(
            key=lambda region: (
                region.y,
                region.x
            )
        )

        # ----------------------------------------------------
        # Assign IDs
        # ----------------------------------------------------

        for index, region in enumerate(
            regions,
            start=1
        ):

            region.region_id = index

        return regions

    # ========================================================
    # OCR TEXT REGIONS
    # ========================================================

    def detect_text_regions(
        self,
        ocr_words: List[Dict[str, Any]]
    ) -> List[Region]:
        """
        Convert OCR word bounding boxes into text regions.
        """

        regions = []

        for index, word in enumerate(
            ocr_words,
            start=1
        ):

            text = word.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            regions.append(
                Region(
                    region_id=index,

                    region_type="text",

                    x=int(
                        word["left"]
                    ),

                    y=int(
                        word["top"]
                    ),

                    width=int(
                        word["width"]
                    ),

                    height=int(
                        word["height"]
                    ),

                    confidence=float(
                        word.get(
                            "confidence",
                            0
                        )
                    ),

                    source="tesseract",
                )
            )

        return regions

    # ========================================================
    # LOW CONFIDENCE OCR
    # ========================================================

    def detect_low_confidence_regions(
        self,
        ocr_words: List[Dict[str, Any]],
        threshold: float = 60.0
    ) -> List[Region]:
        """
        Identify low-confidence OCR regions.

        These are NOT removed from visual detection.
        """

        regions = []

        for index, word in enumerate(
            ocr_words,
            start=1
        ):

            confidence = float(
                word.get(
                    "confidence",
                    0
                )
            )

            if confidence < 0:
                continue

            if confidence >= threshold:
                continue

            text = word.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            regions.append(
                Region(
                    region_id=index,

                    region_type="low_confidence",

                    x=int(
                        word["left"]
                    ),

                    y=int(
                        word["top"]
                    ),

                    width=int(
                        word["width"]
                    ),

                    height=int(
                        word["height"]
                    ),

                    confidence=confidence,

                    source="tesseract",
                )
            )

        return regions

    # ========================================================
    # IOU
    # ========================================================

    @staticmethod
    def calculate_iou(
        region_a: Region,
        region_b: Region
    ) -> float:
        """
        Calculate Intersection over Union.
        """

        ax1 = region_a.x
        ay1 = region_a.y

        ax2 = (
            region_a.x
            + region_a.width
        )

        ay2 = (
            region_a.y
            + region_a.height
        )

        bx1 = region_b.x
        by1 = region_b.y

        bx2 = (
            region_b.x
            + region_b.width
        )

        by2 = (
            region_b.y
            + region_b.height
        )

        intersection_x1 = max(
            ax1,
            bx1
        )

        intersection_y1 = max(
            ay1,
            by1
        )

        intersection_x2 = min(
            ax2,
            bx2
        )

        intersection_y2 = min(
            ay2,
            by2
        )

        intersection_width = max(
            0,
            intersection_x2
            - intersection_x1
        )

        intersection_height = max(
            0,
            intersection_y2
            - intersection_y1
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        area_a = (
            region_a.width
            * region_a.height
        )

        area_b = (
            region_b.width
            * region_b.height
        )

        union_area = (
            area_a
            + area_b
            - intersection_area
        )

        if union_area == 0:
            return 0.0

        return (
            intersection_area
            / union_area
        )

    # ========================================================
    # COMPLETE PAGE ANALYSIS
    # ========================================================

    def analyze_page(
        self,
        image,
        ocr_words: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a complete PDF page.

        Important pipeline:

            OCR
             ↓
            text regions
             ↓
            reliable OCR mask
             ↓
            OpenCV
             ↓
            visual candidates
        """

        if ocr_words is None:
            ocr_words = []

        # ----------------------------------------------------
        # Text regions
        # ----------------------------------------------------

        text_regions = (
            self.detect_text_regions(
                ocr_words
            )
        )

        # ----------------------------------------------------
        # Low-confidence OCR
        # ----------------------------------------------------

        low_confidence_regions = (
            self.detect_low_confidence_regions(
                ocr_words
            )
        )

        # ----------------------------------------------------
        # Visual candidates
        #
        # IMPORTANT:
        # OCR words are now passed into detect_contours().
        # ----------------------------------------------------

        visual_candidates = (
            self.detect_contours(
                image,
                ocr_words
            )
        )

        # ----------------------------------------------------
        # Return analysis
        # ----------------------------------------------------

        return {

            "image_width": image.width,

            "image_height": image.height,

            "text_regions": [
                region.to_dict()
                for region in text_regions
            ],

            "low_confidence_regions": [
                region.to_dict()
                for region in low_confidence_regions
            ],

            "visual_candidates": [
                region.to_dict()
                for region in visual_candidates
            ],

            "counts": {

                "text_regions": len(
                    text_regions
                ),

                "low_confidence_regions": len(
                    low_confidence_regions
                ),

                "visual_candidates": len(
                    visual_candidates
                ),
            },
        }