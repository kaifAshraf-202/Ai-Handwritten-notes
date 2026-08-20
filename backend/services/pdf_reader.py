from pathlib import Path
import fitz


class PDFReader:
    """
    Handles basic PDF operations:
    - Open PDF
    - Validate PDF
    - Get metadata
    - Read individual pages
    - Render pages as images
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        try:
            self.document = fitz.open(self.pdf_path)
        except Exception as error:
            raise ValueError(
                f"Unable to open PDF: {error}"
            )

        if self.document.page_count == 0:
            raise ValueError("The PDF contains no pages.")

    # -----------------------------------------
    # Basic information
    # -----------------------------------------

    def get_page_count(self) -> int:
        return self.document.page_count

    def get_metadata(self) -> dict:
        metadata = self.document.metadata or {}

        return {
            "filename": self.pdf_path.name,
            "page_count": self.document.page_count,
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
        }

    # -----------------------------------------
    # Page handling
    # -----------------------------------------

    def get_page(self, page_number: int):
        """
        page_number is 1-based.

        Example:
        get_page(1) -> first PDF page
        """

        if page_number < 1 or page_number > self.document.page_count:
            raise ValueError(
                f"Page must be between 1 and "
                f"{self.document.page_count}"
            )

        return self.document.load_page(page_number - 1)

    # -----------------------------------------
    # Direct text extraction
    # -----------------------------------------

    def extract_text(self, page_number: int) -> str:
        page = self.get_page(page_number)

        text = page.get_text("text")

        return text.strip()

    # -----------------------------------------
    # Render page
    # -----------------------------------------

    def render_page(
        self,
        page_number: int,
        dpi: int = 200
    ) -> bytes:

        page = self.get_page(page_number)

        pixmap = page.get_pixmap(
            dpi=dpi,
            alpha=False
        )

        return pixmap.tobytes("png")

    # -----------------------------------------
    # Detect whether page probably needs OCR
    # -----------------------------------------

    def needs_ocr(
        self,
        page_number: int,
        minimum_characters: int = 30
    ) -> bool:

        text = self.extract_text(page_number)

        return len(text.strip()) < minimum_characters

    # -----------------------------------------
    # Analyze page
    # -----------------------------------------

    def analyze_page(self, page_number: int) -> dict:

        text = self.extract_text(page_number)

        requires_ocr = len(text.strip()) < 30

        return {
            "page_number": page_number,
            "direct_text": text,
            "character_count": len(text),
            "requires_ocr": requires_ocr,
        }

    # -----------------------------------------
    # Close PDF
    # -----------------------------------------

    def close(self):
        if self.document:
            self.document.close()

    # -----------------------------------------
    # Context manager support
    # -----------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()