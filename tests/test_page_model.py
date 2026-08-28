from backend.services.page_model import (
    PageModel,
    PageModelBuilder,
    build_page_model,
)


def main():

    print()
    print("=" * 56)
    print("             HANDNOTE AI - PAGE MODEL")
    print("=" * 56)

    # --------------------------------------------------------
    # Test with representative pipeline data
    # --------------------------------------------------------

    ocr_words = [
        {
            "text": "OPTICAL",
            "confidence": 90.0,
            "left": 1196,
            "top": 78,
            "width": 316,
            "height": 56,
        },
        {
            "text": "ISOMERISM",
            "confidence": 90.0,
            "left": 1534,
            "top": 78,
            "width": 421,
            "height": 56,
        },
    ]

    text_blocks = [
        {
            "block_id": 1,
            "text": "OPTICAL ISOMERISM",
            "confidence": 90.0,
        }
    ]

    visual_regions = [
        {
            "region_id": 1,
            "x": 1182,
            "y": 62,
            "width": 780,
            "height": 88,
        }
    ]

    visual_classifications = [
        {
            "region_id": 1,
            "classification": "diagram",
            "confidence": 0.84,
        }
    ]

    semantic_regions = [
        {
            "region_id": 1,
            "region_type": "diagram",
            "parent_region_id": 1,
        }
    ]

    layout_blocks = [
        {
            "block_id": 1,
            "x": 45,
            "y": 62,
            "width": 3213,
            "height": 1123,
            "region_ids": [1],
            "block_type": "visual",
            "confidence": 0.93,
        }
    ]

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    page = build_page_model(
        page_number=1,
        image_width=3556,
        image_height=2000,
        ocr_words=ocr_words,
        text_blocks=text_blocks,
        text_classifications=[],
        visual_regions=visual_regions,
        visual_classifications=visual_classifications,
        semantic_regions=semantic_regions,
        layout_blocks=layout_blocks,
        metadata={
            "source": "test.pdf",
        },
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print()
    print("Page model created.")

    print()
    print("=" * 56)
    print("                 PAGE SUMMARY")
    print("=" * 56)

    summary = page.summary()

    for key, value in summary.items():

        print(
            f"{key:<28}: {value}"
        )

    # --------------------------------------------------------
    # Dictionary conversion
    # --------------------------------------------------------

    data = page.to_dict()

    print()
    print(
        "Dictionary conversion: OK"
    )

    print(
        f"Top-level fields: "
        f"{len(data)}"
    )

    # --------------------------------------------------------
    # Type verification
    # --------------------------------------------------------

    assert isinstance(
        page,
        PageModel
    )

    assert page.page_number == 1

    assert page.image_width == 3556

    assert page.image_height == 2000

    assert len(
        page.ocr_words
    ) == 2

    assert len(
        page.visual_regions
    ) == 1

    assert len(
        page.visual_classifications
    ) == 1

    assert len(
        page.semantic_regions
    ) == 1

    assert len(
        page.layout_blocks
    ) == 1

    print()
    print(
        "All assertions passed."
    )

    print()
    print("=" * 56)
    print(
        "        PAGE MODEL COMPLETED"
    )
    print("=" * 56)


if __name__ == "__main__":
    main()