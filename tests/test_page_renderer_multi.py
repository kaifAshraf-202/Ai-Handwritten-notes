from __future__ import annotations

import importlib
import pkgutil
import time
from pathlib import Path
from typing import Any, Dict, Type

import fitz
import numpy as np
from PIL import Image

from backend.services.page_renderer import PageRenderer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "storage"
    / "output"
    / "multi_page_test"
)

DPI = 200

LANGUAGE = "eng"

PSM = 3


# ------------------------------------------------------------
# Pages to test.
#
# These are 1-based page numbers.
# ------------------------------------------------------------

TEST_PAGES = [
    1,
    10,
    25,
    50,
    75,
    100,
]


# ============================================================
# PDF DISCOVERY
# ============================================================

def find_test_pdf() -> Path:
    project_root = Path(__file__).resolve().parents[1]

    preferred_names = [
        "test.pdf",
        "test(1).pdf",
        "GeoOptConSKm(1).pdf",
        "GeoOptConSKm.pdf",
    ]

    # 1. Check project root
    for name in preferred_names:
        candidate = project_root / name
        if candidate.exists():
            return candidate

    # 2. Check storage/uploads
    uploads_dir = project_root / "storage" / "uploads"

    if uploads_dir.exists():
        for name in preferred_names:
            candidate = uploads_dir / name
            if candidate.exists():
                return candidate

        pdfs = sorted(uploads_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]

    # 3. Check storage/input
    input_dir = project_root / "storage" / "input"

    if input_dir.exists():
        pdfs = sorted(input_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]

    # 4. Search entire project
    pdfs = sorted(project_root.glob("**/*.pdf"))

    if pdfs:
        return pdfs[0]

    raise FileNotFoundError(
        "No PDF file was found anywhere in the project.\n\n"
        "Place a test PDF in:\n"
        f"    {uploads_dir}\n"
    )
    """
    Find the PDF used for the multi-page renderer test.

    Priority:

        1. test.pdf
        2. test(1).pdf
        3. GeoOptConSKm(1).pdf
        4. first PDF in project root
        5. first PDF inside storage/input
    """

    preferred_names = [
        "test.pdf",
        "test(1).pdf",
        "GeoOptConSKm(1).pdf",
        "GeoOptConSKm.pdf",
    ]

    # --------------------------------------------------------
    # Check preferred filenames.
    # --------------------------------------------------------

    for filename in preferred_names:

        candidate = (
            PROJECT_ROOT
            / filename
        )

        if candidate.exists():

            return candidate

    # --------------------------------------------------------
    # Check storage/input.
    # --------------------------------------------------------

    input_dir = (
        PROJECT_ROOT
        / "storage"
        / "input"
    )

    if input_dir.exists():

        pdfs = sorted(
            input_dir.glob("*.pdf")
        )

        if pdfs:

            return pdfs[0]

    # --------------------------------------------------------
    # Check project root.
    # --------------------------------------------------------

    pdfs = sorted(
        PROJECT_ROOT.glob("*.pdf")
    )

    if pdfs:

        return pdfs[0]

    raise FileNotFoundError(
        "No PDF file was found.\n\n"
        "Place your test PDF in the project root, "
        "for example:\n\n"
        "    D:\\KAIF ASHRAF\\Crazy Projects\\handnote-ai\\test.pdf\n\n"
        "Then run:\n\n"
        "    python -m tests.test_page_renderer_multi"
    )


# ============================================================
# PIPELINE DISCOVERY
# ============================================================

def find_handnote_pipeline_class() -> Type[Any]:
    """
    Automatically find HandNotePagePipeline inside
    backend.services.

    This means we do not need to guess the pipeline
    module filename.
    """

    import backend.services

    services_path = list(
        backend.services.__path__
    )

    candidate_modules = []

    # --------------------------------------------------------
    # Prefer likely pipeline/page modules.
    # --------------------------------------------------------

    for module_info in pkgutil.iter_modules(
        services_path
    ):

        module_name = module_info.name

        if (
            "pipeline" in module_name.lower()
            or "page" in module_name.lower()
        ):

            candidate_modules.append(
                module_name
            )

    # --------------------------------------------------------
    # Add remaining modules.
    # --------------------------------------------------------

    for module_info in pkgutil.iter_modules(
        services_path
    ):

        module_name = module_info.name

        if module_name not in candidate_modules:

            candidate_modules.append(
                module_name
            )

    # --------------------------------------------------------
    # Search for HandNotePagePipeline.
    # --------------------------------------------------------

    for module_name in candidate_modules:

        try:

            module = importlib.import_module(
                f"backend.services.{module_name}"
            )

        except Exception:

            continue

        pipeline_class = getattr(
            module,
            "HandNotePagePipeline",
            None,
        )

        if pipeline_class is not None:

            print(
                "Pipeline module found:"
            )

            print(
                f"    backend.services.{module_name}"
            )

            return pipeline_class

    raise ImportError(
        "HandNotePagePipeline could not be found "
        "inside backend.services."
    )


# ============================================================
# IMAGE METRICS
# ============================================================

def calculate_image_metrics(
    source: Image.Image,
    rendered: Image.Image,
) -> Dict[str, float]:
    """
    Compare original source image against reconstructed
    renderer output.

    Metrics:

        MAE
        Exact pixel match
        Within ±5
        Within ±20
        Significant difference >20
    """

    source_rgb = source.convert("RGB")

    rendered_rgb = rendered.convert("RGB")

    # --------------------------------------------------------
    # Ensure identical dimensions.
    # --------------------------------------------------------

    if source_rgb.size != rendered_rgb.size:

        rendered_rgb = rendered_rgb.resize(
            source_rgb.size,
            Image.Resampling.LANCZOS,
        )

    source_array = np.asarray(
        source_rgb,
        dtype=np.int16,
    )

    rendered_array = np.asarray(
        rendered_rgb,
        dtype=np.int16,
    )

    # --------------------------------------------------------
    # Absolute RGB difference.
    # --------------------------------------------------------

    difference = np.abs(
        source_array
        - rendered_array
    )

    # --------------------------------------------------------
    # MAE.
    # --------------------------------------------------------

    mae = float(
        difference.mean()
    )

    # --------------------------------------------------------
    # Per-pixel conditions.
    # --------------------------------------------------------

    exact = np.all(
        difference == 0,
        axis=2,
    )

    within_5 = np.all(
        difference <= 5,
        axis=2,
    )

    within_20 = np.all(
        difference <= 20,
        axis=2,
    )

    significant = np.any(
        difference > 20,
        axis=2,
    )

    total_pixels = exact.size

    return {
        "mae": mae,

        "exact_pct": (
            100.0
            * float(exact.sum())
            / float(total_pixels)
        ),

        "within_5_pct": (
            100.0
            * float(within_5.sum())
            / float(total_pixels)
        ),

        "within_20_pct": (
            100.0
            * float(within_20.sum())
            / float(total_pixels)
        ),

        "significant_pct": (
            100.0
            * float(significant.sum())
            / float(total_pixels)
        ),
    }


# ============================================================
# SAFE LENGTH
# ============================================================

def collection_length(
    value: Any,
) -> int:

    if value is None:
        return 0

    try:

        return len(value)

    except TypeError:

        return 0


# ============================================================
# PAGE MODEL STATISTICS
# ============================================================

def get_model_statistics(
    page_model: Any,
) -> Dict[str, int]:
    """
    Extract useful PageModel statistics.
    """

    def get_value(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if obj is None:
            return default

        if isinstance(obj, dict):

            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    return {
        "text_blocks": collection_length(
            get_value(
                page_model,
                "text_blocks",
                [],
            )
        ),

        "visual_regions": collection_length(
            get_value(
                page_model,
                "visual_regions",
                [],
            )
        ),

        "semantic_regions": collection_length(
            get_value(
                page_model,
                "semantic_regions",
                [],
            )
        ),

        "layout_blocks": collection_length(
            get_value(
                page_model,
                "layout_blocks",
                [],
            )
        ),
    }


# ============================================================
# MAIN TEST
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print(
        "        HANDNOTE AI - V3 MULTI-PAGE RENDERER TEST"
    )
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Locate PDF.
    # --------------------------------------------------------

    pdf_path = find_test_pdf()

    print(
        f"PDF selected:"
    )

    print(
        f"    {pdf_path}"
    )

    print()

    # --------------------------------------------------------
    # Create output directory.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find pipeline.
    # --------------------------------------------------------

    print(
        "Searching for HandNotePagePipeline..."
    )

    PipelineClass = (
        find_handnote_pipeline_class()
    )

    print()

    # --------------------------------------------------------
    # Create pipeline.
    # --------------------------------------------------------

    pipeline = PipelineClass()

    # --------------------------------------------------------
    # Create V3 renderer.
    # --------------------------------------------------------

    renderer = PageRenderer(
        debug=False,
    )

    # --------------------------------------------------------
    # Open PDF.
    # --------------------------------------------------------

    document = fitz.open(
        pdf_path
    )

    total_pages = len(
        document
    )

    print(
        f"Total PDF pages: {total_pages}"
    )

    print(
        f"Requested test pages: {TEST_PAGES}"
    )

    print(
        f"DPI: {DPI}"
    )

    print()

    results = []

    try:

        # ====================================================
        # PROCESS SELECTED PAGES
        # ====================================================

        for page_number in TEST_PAGES:

            print()
            print("-" * 72)
            print(
                f"TESTING PAGE {page_number}"
            )
            print("-" * 72)

            # ------------------------------------------------
            # Validate page.
            # ------------------------------------------------

            if (
                page_number < 1
                or page_number > total_pages
            ):

                print(
                    f"[SKIPPED] Page "
                    f"{page_number} does not exist."
                )

                continue

            start_time = (
                time.perf_counter()
            )

            page_index = (
                page_number - 1
            )

            # ------------------------------------------------
            # Render source image.
            # ------------------------------------------------

            print(
                "Rendering source page..."
            )

            source_image = (
                pipeline.render_page(
                    document=document,
                    page_number=page_index,
                    dpi=DPI,
                )
            )

            print(
                f"Source size: "
                f"{source_image.width} x "
                f"{source_image.height}"
            )

            # ------------------------------------------------
            # Run COMPLETE HandNote pipeline.
            #
            # PDF
            # ↓
            # OCR
            # ↓
            # Region Detection
            # ↓
            # Visual Merger
            # ↓
            # Visual Splitter
            # ↓
            # Visual Classification
            # ↓
            # Semantic Grouping
            # ↓
            # Layout Analysis
            # ↓
            # Text Block Merging
            # ↓
            # Content Classification
            # ↓
            # PageModel
            # ------------------------------------------------

            print(
                "Running complete HandNote pipeline..."
            )

            pipeline_result = (
                pipeline.analyze_pdf_page(
                    pdf_path=pdf_path,
                    page_number=page_index,
                    dpi=DPI,
                    language=LANGUAGE,
                    psm=PSM,
                )
            )

            print(
                "Pipeline completed."
            )

            # ------------------------------------------------
            # Extract PageModel.
            # ------------------------------------------------

            page_model = (
                pipeline_result.page
            )

            print(
                "PageModel: OK"
            )

            # ------------------------------------------------
            # PageModel statistics.
            # ------------------------------------------------

            stats = (
                get_model_statistics(
                    page_model
                )
            )

            print(
                f"Text blocks: "
                f"{stats['text_blocks']}"
            )

            print(
                f"Visual regions: "
                f"{stats['visual_regions']}"
            )

            print(
                f"Semantic regions: "
                f"{stats['semantic_regions']}"
            )

            print(
                f"Layout blocks: "
                f"{stats['layout_blocks']}"
            )

            # ------------------------------------------------
            # Render using PageRenderer V3.
            # ------------------------------------------------

            print(
                "Rendering with PageRenderer V3..."
            )

            output_path = (
                OUTPUT_DIR
                / (
                    f"page_"
                    f"{page_number:03d}.png"
                )
            )

            rendered_image = (
                renderer.render(
                    page_model=page_model,
                    source_image=source_image,
                    output_path=output_path,
                )
            )

            # ------------------------------------------------
            # Verify output.
            # ------------------------------------------------

            if not output_path.exists():

                raise RuntimeError(
                    "Renderer did not create output file:\n"
                    f"{output_path}"
                )

            # ------------------------------------------------
            # Calculate visual metrics.
            # ------------------------------------------------

            metrics = (
                calculate_image_metrics(
                    source=source_image,
                    rendered=rendered_image,
                )
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            # ------------------------------------------------
            # Store result.
            # ------------------------------------------------

            result = {
                "page": page_number,

                "mae": metrics[
                    "mae"
                ],

                "exact_pct": metrics[
                    "exact_pct"
                ],

                "within_5_pct": metrics[
                    "within_5_pct"
                ],

                "within_20_pct": metrics[
                    "within_20_pct"
                ],

                "significant_pct": metrics[
                    "significant_pct"
                ],

                "text_blocks": stats[
                    "text_blocks"
                ],

                "visual_regions": stats[
                    "visual_regions"
                ],

                "semantic_regions": stats[
                    "semantic_regions"
                ],

                "layout_blocks": stats[
                    "layout_blocks"
                ],

                "seconds": elapsed,

                "output": str(
                    output_path
                ),
            }

            results.append(
                result
            )

            # ------------------------------------------------
            # Print page result.
            # ------------------------------------------------

            print()
            print(
                "RESULT"
            )

            print(
                f"Output: "
                f"{output_path}"
            )

            print(
                f"MAE: "
                f"{metrics['mae']:.4f}"
            )

            print(
                f"Exact pixel match: "
                f"{metrics['exact_pct']:.2f}%"
            )

            print(
                f"Within ±5: "
                f"{metrics['within_5_pct']:.2f}%"
            )

            print(
                f"Within ±20: "
                f"{metrics['within_20_pct']:.2f}%"
            )

            print(
                f"Significant difference >20: "
                f"{metrics['significant_pct']:.2f}%"
            )

            print(
                f"Processing time: "
                f"{elapsed:.2f}s"
            )

    finally:

        document.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 72)
    print(
        "                    FINAL SUMMARY"
    )
    print("=" * 72)
    print()

    if not results:

        print(
            "No pages were successfully tested."
        )

        return

    # --------------------------------------------------------
    # Table header.
    # --------------------------------------------------------

    print(
        "Page | "
        "MAE    | "
        "Exact    | "
        "±5       | "
        "±20      | "
        ">20"
    )

    print(
        "-" * 72
    )

    # --------------------------------------------------------
    # Individual results.
    # --------------------------------------------------------

    for result in results:

        print(
            f"{result['page']:4d} | "
            f"{result['mae']:6.2f} | "
            f"{result['exact_pct']:7.2f}% | "
            f"{result['within_5_pct']:7.2f}% | "
            f"{result['within_20_pct']:7.2f}% | "
            f"{result['significant_pct']:6.2f}%"
        )

    # --------------------------------------------------------
    # Calculate averages.
    # --------------------------------------------------------

    average_mae = (
        sum(
            result["mae"]
            for result in results
        )
        / len(results)
    )

    average_exact = (
        sum(
            result["exact_pct"]
            for result in results
        )
        / len(results)
    )

    average_within_5 = (
        sum(
            result["within_5_pct"]
            for result in results
        )
        / len(results)
    )

    average_within_20 = (
        sum(
            result["within_20_pct"]
            for result in results
        )
        / len(results)
    )

    average_significant = (
        sum(
            result["significant_pct"]
            for result in results
        )
        / len(results)
    )

    average_time = (
        sum(
            result["seconds"]
            for result in results
        )
        / len(results)
    )

    # --------------------------------------------------------
    # Print averages.
    # --------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "AVERAGE RESULTS"
    )

    print("=" * 72)

    print(
        f"Average MAE: "
        f"{average_mae:.4f}"
    )

    print(
        f"Average exact pixel match: "
        f"{average_exact:.2f}%"
    )

    print(
        f"Average within ±5: "
        f"{average_within_5:.2f}%"
    )

    print(
        f"Average within ±20: "
        f"{average_within_20:.2f}%"
    )

    print(
        f"Average significant difference >20: "
        f"{average_significant:.2f}%"
    )

    print(
        f"Average processing time: "
        f"{average_time:.2f}s"
    )

    # --------------------------------------------------------
    # Best page.
    # --------------------------------------------------------

    best_page = min(
        results,
        key=lambda result: result["mae"],
    )

    # --------------------------------------------------------
    # Worst page.
    # --------------------------------------------------------

    worst_page = max(
        results,
        key=lambda result: result["mae"],
    )

    print()

    print(
        f"Best page by MAE: "
        f"Page {best_page['page']} "
        f"({best_page['mae']:.4f})"
    )

    print(
        f"Worst page by MAE: "
        f"Page {worst_page['page']} "
        f"({worst_page['mae']:.4f})"
    )

    # --------------------------------------------------------
    # Output files.
    # --------------------------------------------------------

    print()

    print(
        "Generated files:"
    )

    for result in results:

        print(
            f"  Page {result['page']}: "
            f"{result['output']}"
        )

    print()
    print("=" * 72)
    print(
        "             MULTI-PAGE TEST COMPLETED"
    )
    print("=" * 72)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()