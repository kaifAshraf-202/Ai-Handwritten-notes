from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple


# ============================================================
# TEXT BLOCK
# ============================================================

@dataclass
class TextBlock:

    block_id: int

    text: str

    x: int
    y: int
    width: int
    height: int

    average_confidence: float

    word_count: int

    tesseract_block: int
    tesseract_paragraph: int
    tesseract_line: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# TEXT BLOCK MERGER
# ============================================================

class TextBlockMerger:

    """
    Converts Tesseract OCR words into spatially meaningful
    text blocks.

    Tesseract provides the initial layout information, but
    we additionally use physical distance between words.

    This prevents unrelated content from different areas of
    the page being merged into one text block.
    """

    def __init__(
        self,
        max_horizontal_gap: int = 140,
        max_vertical_difference: int = 45,
    ):

        self.max_horizontal_gap = (
            max_horizontal_gap
        )

        self.max_vertical_difference = (
            max_vertical_difference
        )

    # ========================================================
    # GEOMETRY
    # ========================================================

    @staticmethod
    def right(word):
        return (
            int(word["left"])
            + int(word["width"])
        )

    @staticmethod
    def bottom(word):
        return (
            int(word["top"])
            + int(word["height"])
        )

    @staticmethod
    def center_y(word):
        return (
            int(word["top"])
            + int(word["height"]) / 2
        )

    # ========================================================
    # TESSERACT GROUPING
    # ========================================================

    @staticmethod
    def get_line_key(word):

        return (
            int(word.get("block_num", 0)),
            int(word.get("par_num", 0)),
            int(word.get("line_num", 0)),
        )

    def group_by_tesseract_line(
        self,
        words: List[Dict[str, Any]]
    ):

        groups = {}

        for word in words:

            text = word.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            key = self.get_line_key(
                word
            )

            if key not in groups:
                groups[key] = []

            groups[key].append(
                word
            )

        # Left → right.
        for key in groups:

            groups[key].sort(
                key=lambda word:
                int(word["left"])
            )

        return groups

    # ========================================================
    # SPLIT A TESSERACT LINE SPATIALLY
    # ========================================================

    def split_line(
        self,
        words: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:

        if not words:
            return []

        words = sorted(
            words,
            key=lambda word:
            int(word["left"])
        )

        sub_lines = [
            [words[0]]
        ]

        for current in words[1:]:

            previous = (
                sub_lines[-1][-1]
            )

            # -----------------------------------------------
            # Horizontal gap
            # -----------------------------------------------

            gap = (
                int(current["left"])
                - self.right(previous)
            )

            # -----------------------------------------------
            # Vertical difference
            # -----------------------------------------------

            vertical_difference = abs(
                self.center_y(current)
                - self.center_y(previous)
            )

            # -----------------------------------------------
            # Decide whether to split
            # -----------------------------------------------

            should_split = (
                gap > self.max_horizontal_gap
                or
                vertical_difference
                > self.max_vertical_difference
            )

            if should_split:

                sub_lines.append(
                    [current]
                )

            else:

                sub_lines[-1].append(
                    current
                )

        return sub_lines

    # ========================================================
    # CREATE TEXT BLOCK
    # ========================================================

    def create_block(
        self,
        words,
        block_id
    ):

        x1 = min(
            int(word["left"])
            for word in words
        )

        y1 = min(
            int(word["top"])
            for word in words
        )

        x2 = max(
            self.right(word)
            for word in words
        )

        y2 = max(
            self.bottom(word)
            for word in words
        )

        text = " ".join(
            word["text"].strip()
            for word in words
            if word["text"].strip()
        )

        confidences = [
            float(word["confidence"])
            for word in words
            if float(word["confidence"]) >= 0
        ]

        average_confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        first_word = words[0]

        return TextBlock(
            block_id=block_id,

            text=text,

            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,

            average_confidence=average_confidence,

            word_count=len(words),

            tesseract_block=int(
                first_word.get(
                    "block_num",
                    0
                )
            ),

            tesseract_paragraph=int(
                first_word.get(
                    "par_num",
                    0
                )
            ),

            tesseract_line=int(
                first_word.get(
                    "line_num",
                    0
                )
            ),
        )

    # ========================================================
    # MAIN
    # ========================================================

    def merge(
        self,
        words: List[Dict[str, Any]]
    ) -> List[TextBlock]:

        tesseract_lines = (
            self.group_by_tesseract_line(
                words
            )
        )

        blocks = []

        for words_in_line in (
            tesseract_lines.values()
        ):

            # -----------------------------------------------
            # Split suspiciously large spatial gaps.
            # -----------------------------------------------

            sub_lines = self.split_line(
                words_in_line
            )

            for sub_line in sub_lines:

                if not sub_line:
                    continue

                block = self.create_block(
                    sub_line,
                    len(blocks) + 1
                )

                blocks.append(
                    block
                )

        # ----------------------------------------------------
        # Sort top → bottom, then left → right.
        # ----------------------------------------------------

        blocks.sort(
            key=lambda block: (
                block.y,
                block.x
            )
        )

        # ----------------------------------------------------
        # Reassign IDs.
        # ----------------------------------------------------

        for index, block in enumerate(
            blocks,
            start=1
        ):

            block.block_id = index

        return blocks