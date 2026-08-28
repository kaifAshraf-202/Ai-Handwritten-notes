from pathlib import Path

from backend.services.pipeline import (
    HandNotePagePipeline,
)


PDF_PATH = Path(
    "storage/uploads/test.pdf"
)


def main():

    print()
    print("=" * 60)
    print("           HANDNOTE AI - FULL PIPELINE")
    print("=" * 60)

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    pipeline = HandNotePagePipeline()

    print()
    print("Analyzing PDF...")

    results = pipeline.analyze_pdf(
        pdf_path=PDF_PATH,
        dpi=200,
        language="eng",
        psm=3,
    )

    print()
    print("=" * 60)
    print("              PIPELINE RESULTS")
    print("=" * 60)

    print(
        f"\nPages analyzed: "
        f"{len(results)}"
    )

    for result in results:

        page = result.page

        print()
        print(
            f"PAGE {page.page_number}"
        )

        print(
            "-" * 60
        )

        print(
            f"OCR words:               "
            f"{len(page.ocr_words)}"
        )

        print(
            f"Text blocks:             "
            f"{len(page.text_blocks)}"
        )

        print(
            f"Text classifications:    "
            f"{len(page.text_classifications)}"
        )

        print(
            f"Raw visual regions:      "
            f"{len(result.raw_visual_regions)}"
        )

        print(
            f"Merged visual regions:   "
            f"{len(result.merged_visual_regions)}"
        )

        print(
            f"Split visual regions:    "
            f"{len(result.split_visual_regions)}"
        )

        print(
            f"Visual classifications:  "
            f"{len(result.visual_classifications)}"
        )

        semantic_regions = (
            result.semantic_result.get(
                "regions",
                []
            )
        )

        print(
            f"Semantic regions:        "
            f"{len(semantic_regions)}"
        )

        print(
            f"Layout blocks:           "
            f"{len(result.layout_blocks)}"
        )

        print()
        print(
            "PageModel: OK"
        )

        # ----------------------------------------------------
        # Show text
        # ----------------------------------------------------

        print()
        print(
            "TEXT BLOCKS"
        )

        for block in page.text_blocks:

            data = (
                block
                if isinstance(
                    block,
                    dict
                )
                else {}
            )

            text = data.get(
                "text",
                ""
            )

            if text:

                print(
                    f"  - {text}"
                )

    print()
    print("=" * 60)
    print(
        "          FULL PIPELINE COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()