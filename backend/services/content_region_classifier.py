from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import re


# ============================================================
# CLASSIFIED REGION
# ============================================================

@dataclass
class ClassifiedRegion:

    region_id: int

    region_type: str

    text: str

    x: int
    y: int
    width: int
    height: int

    confidence: float

    source: str

    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# CONTENT REGION CLASSIFIER
# ============================================================

class ContentRegionClassifier:

    """
    Conservative classifier for OCR regions.

    Classification:

        reliable_text
        uncertain_text
        symbol_or_visual
        visual_candidate

    Important principle:

        If OCR is ambiguous, preserve the original
        visual region instead of inventing content.
    """

    def __init__(
        self,
        reliable_confidence: float = 70.0,
        min_reliable_words: int = 2,
        min_single_word_width: int = 40,
        min_single_word_height: int = 25,
    ):

        self.reliable_confidence = (
            reliable_confidence
        )

        self.min_reliable_words = (
            min_reliable_words
        )

        self.min_single_word_width = (
            min_single_word_width
        )

        self.min_single_word_height = (
            min_single_word_height
        )

    # ========================================================
    # TEXT CHARACTER ANALYSIS
    # ========================================================

    @staticmethod
    def clean_text(text: str) -> str:

        return " ".join(
            text.strip().split()
        )

    @staticmethod
    def word_count(text: str) -> int:

        return len(
            text.split()
        )

    @staticmethod
    def contains_letters(text: str) -> bool:

        return bool(
            re.search(
                r"[A-Za-z]",
                text
            )
        )

    @staticmethod
    def contains_multiple_letters(text: str) -> bool:

        letters = re.findall(
            r"[A-Za-z]",
            text
        )

        return len(letters) >= 2

    @staticmethod
    def looks_like_symbol(text: str) -> bool:

        text = text.strip()

        if not text:
            return True

        # Very short OCR outputs are often symbols,
        # diagram labels, or OCR noise.
        if len(text) <= 2:
            return True

        # Pure punctuation/symbols.
        if not re.search(
            r"[A-Za-z0-9]",
            text
        ):
            return True

        return False

    # ========================================================
    # CLASSIFY OCR BLOCK
    # ========================================================

    def classify_text_block(
        self,
        block
    ) -> ClassifiedRegion:

        text = self.clean_text(
            block.text
        )

        confidence = float(
            block.average_confidence
        )

        words = self.word_count(
            text
        )

        # ----------------------------------------------------
        # Empty OCR
        # ----------------------------------------------------

        if not text:

            return ClassifiedRegion(
                region_id=block.block_id,
                region_type="symbol_or_visual",
                text=text,
                x=block.x,
                y=block.y,
                width=block.width,
                height=block.height,
                confidence=confidence,
                source="tesseract",
                reason="Empty OCR result.",
            )

        # ----------------------------------------------------
        # Low confidence
        # ----------------------------------------------------

        if confidence < self.reliable_confidence:

            return ClassifiedRegion(
                region_id=block.block_id,
                region_type="uncertain_text",
                text=text,
                x=block.x,
                y=block.y,
                width=block.width,
                height=block.height,
                confidence=confidence,
                source="tesseract",
                reason=(
                    "OCR confidence is below "
                    "the reliable threshold."
                ),
            )

        # ----------------------------------------------------
        # Very short output
        # ----------------------------------------------------

        if self.looks_like_symbol(text):

            return ClassifiedRegion(
                region_id=block.block_id,
                region_type="symbol_or_visual",
                text=text,
                x=block.x,
                y=block.y,
                width=block.width,
                height=block.height,
                confidence=confidence,
                source="tesseract",
                reason=(
                    "Very short OCR result; "
                    "likely a symbol, label, "
                    "or diagram component."
                ),
            )

        # ----------------------------------------------------
        # Single-word OCR
        # ----------------------------------------------------

        if words == 1:

            # Short single words are treated conservatively.
            if (
                len(text) <= 3
                and not self.contains_multiple_letters(text)
            ):

                return ClassifiedRegion(
                    region_id=block.block_id,
                    region_type="symbol_or_visual",
                    text=text,
                    x=block.x,
                    y=block.y,
                    width=block.width,
                    height=block.height,
                    confidence=confidence,
                    source="tesseract",
                    reason=(
                        "Short isolated OCR token; "
                        "likely a symbol or label."
                    ),
                )

        # ----------------------------------------------------
        # Reliable text
        # ----------------------------------------------------

        return ClassifiedRegion(
            region_id=block.block_id,
            region_type="reliable_text",
            text=text,
            x=block.x,
            y=block.y,
            width=block.width,
            height=block.height,
            confidence=confidence,
            source="tesseract",
            reason=(
                "OCR confidence and text structure "
                "indicate usable text."
            ),
        )

    # ========================================================
    # CLASSIFY VISUAL CANDIDATE
    # ========================================================

    def classify_visual_candidate(
        self,
        region: Dict[str, Any]
    ) -> ClassifiedRegion:

        return ClassifiedRegion(
            region_id=region["region_id"],
            region_type="visual_candidate",
            text="",
            x=region["x"],
            y=region["y"],
            width=region["width"],
            height=region["height"],
            confidence=0.0,
            source="opencv",
            reason=(
                "Detected by visual analysis. "
                "Requires region merging and "
                "visual classification."
            ),
        )

    # ========================================================
    # COMPLETE PAGE
    # ========================================================

    def classify_page(
        self,
        text_blocks: List[Any],
        visual_candidates: List[Dict[str, Any]]
    ):

        classified = []

        # ----------------------------------------------------
        # OCR regions
        # ----------------------------------------------------

        for block in text_blocks:

            classified.append(
                self.classify_text_block(
                    block
                )
            )

        # ----------------------------------------------------
        # Visual regions
        # ----------------------------------------------------

        for index, region in enumerate(
            visual_candidates,
            start=1000
        ):

            region_copy = dict(
                region
            )

            region_copy["region_id"] = (
                index
            )

            classified.append(
                self.classify_visual_candidate(
                    region_copy
                )
            )

        # ----------------------------------------------------
        # Page order
        # ----------------------------------------------------

        classified.sort(
            key=lambda region: (
                region.y,
                region.x
            )
        )

        return classified