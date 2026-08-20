from backend.services.pdf_reader import PDFReader


PDF_PATH = "storage/uploads/test.pdf"


with PDFReader(PDF_PATH) as pdf:

    print("\n--- PDF INFORMATION ---")

    print(pdf.get_metadata())

    print("\nTotal pages:")
    print(pdf.get_page_count())

    print("\n--- FIRST PAGE ---")

    result = pdf.analyze_page(1)

    print("Page:", result["page_number"])
    print("Characters:", result["character_count"])
    print("Needs OCR:", result["requires_ocr"])

    print("\nDirect text:")
    print(result["direct_text"][:1000])