"""
HandNote AI
Targeted Full Pipeline + PageRenderer V3 Stress Test

FIRST RUN:
    Pages 80-90 only.

AFTER 80-90 PASS:
    Change:

        PAGE_START = 1
        PAGE_END = None

    to test the entire PDF.

Run from project root:

    python -m tests.test_full_pipeline
"""

from __future__ import annotations

import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# CONFIGURATION
# ============================================================

# FIRST TEST
PAGE_START = 80
PAGE_END = 90

# After the targeted test passes, change to:
#
# PAGE_START = 1
# PAGE_END = None

DPI = 200
LANGUAGE = "eng"
PSM = 3

# During the targeted test, save every page.
SAVE_ALL_RENDERED = True

# Pages for which lightweight image comparison is performed.
METRIC_PAGES = {
    80,
    85,
    86,
    90,
}

OUTPUT_DIR = Path(
    "storage/output/full_pipeline_test"
)

RENDER_DIR = (
    OUTPUT_DIR / "rendered_pages"
)

PROGRESS_FILE = (
    OUTPUT_DIR / "progress.json"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "summary.json"
)


# ============================================================
# OPTIONAL MEMORY MONITOR
# ============================================================

try:
    import psutil
except ImportError:
    psutil = None


def get_memory_mb() -> Optional[float]:
    """
    Return current Python process RSS in MB.

    psutil is optional.
    """
    if psutil is None:
        return None

    try:
        process = psutil.Process(
            os.getpid()
        )

        return round(
            process.memory_info().rss
            / (1024 * 1024),
            1,
        )

    except Exception:
        return None


def print_memory(label: str) -> None:
    value = get_memory_mb()

    if value is None:
        print(
            f"  Memory {label}: "
            "psutil not installed",
            flush=True,
        )
    else:
        print(
            f"  Memory {label}: "
            f"{value:.1f} MB",
            flush=True,
        )


# ============================================================
# FIND PDF
# ============================================================

def find_pdf(project_root: Path) -> Path:
    """
    Locate the test PDF.

    Priority:
    1. storage/uploads/test.pdf
    2. storage/input/*.pdf
    3. project root/*.pdf
    4. recursive search as a final fallback
    """

    preferred = project_root / "storage" / "uploads" / "test.pdf"

    if preferred.exists():
        return preferred

    candidates = [
        project_root / "storage" / "input",
        project_root,
    ]

    for folder in candidates:
        if folder.exists():
            pdfs = sorted(folder.glob("*.pdf"))
            if pdfs:
                return pdfs[0]

    # Final fallback
    pdfs = sorted(project_root.rglob("*.pdf"))

    if pdfs:
        return pdfs[0]

    raise FileNotFoundError(
        "No PDF found.\n\n"
        "Expected test PDF at:\n"
        f"  {preferred}\n"
        "\n"
        "or another PDF inside the project."
    )
# ============================================================
# SAFE OBJECT ACCESS
# ============================================================

def get_value(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


def safe_len(
    value: Any,
) -> int:

    if value is None:
        return 0

    try:
        return len(value)
    except Exception:
        return 0


# ============================================================
# PIPELINE COUNTS
# ============================================================

def get_semantic_count(
    result: Any,
) -> int:

    semantic = get_value(
        result,
        "semantic_result",
        None,
    )

    if semantic is None:
        page = get_value(
            result,
            "page",
            None,
        )

        return safe_len(
            get_value(
                page,
                "semantic_regions",
                [],
            )
        )

    if isinstance(
        semantic,
        dict,
    ):

        for key in (
            "semantic_regions",
            "regions",
            "groups",
            "semantic_groups",
        ):
            value = semantic.get(
                key
            )

            if value is not None:
                return safe_len(value)

    for key in (
        "semantic_regions",
        "regions",
        "groups",
        "semantic_groups",
    ):

        value = get_value(
            semantic,
            key,
            None,
        )

        if value is not None:
            return safe_len(value)

    return safe_len(
        semantic
    )


def get_pipeline_counts(
    result: Any,
) -> Dict[str, int]:

    return {
        "text_blocks": safe_len(
            get_value(
                result,
                "text_blocks",
                [],
            )
        ),

        "visual_regions": safe_len(
            get_value(
                result,
                "split_visual_regions",
                [],
            )
        ),

        "semantic_regions": (
            get_semantic_count(result)
        ),

        "layout_blocks": safe_len(
            get_value(
                result,
                "layout_blocks",
                [],
            )
        ),

        "text_classifications": safe_len(
            get_value(
                result,
                "text_classifications",
                [],
            )
        ),

        "visual_classifications": safe_len(
            get_value(
                result,
                "visual_classifications",
                [],
            )
        ),
    }


# ============================================================
# LIGHTWEIGHT IMAGE METRICS
# ============================================================

def calculate_metrics(
    source_image,
    rendered_image,
    max_dimension: int = 1200,
) -> Dict[str, Any]:

    from PIL import Image
    import numpy as np

    source = source_image.convert(
        "RGB"
    )

    rendered = rendered_image.convert(
        "RGB"
    )

    if source.size != rendered.size:
        rendered = rendered.resize(
            source.size,
            Image.Resampling.LANCZOS,
        )

    scale = min(
        1.0,
        max_dimension
        / max(
            source.width,
            source.height,
        ),
    )

    if scale < 1.0:

        new_size = (
            max(
                1,
                int(
                    source.width
                    * scale
                ),
            ),
            max(
                1,
                int(
                    source.height
                    * scale
                ),
            ),
        )

        source = source.resize(
            new_size,
            Image.Resampling.LANCZOS,
        )

        rendered = rendered.resize(
            new_size,
            Image.Resampling.LANCZOS,
        )

    a = np.asarray(
        source,
        dtype=np.int16,
    )

    b = np.asarray(
        rendered,
        dtype=np.int16,
    )

    diff = np.abs(
        a - b
    )

    pixel_count = (
        diff.shape[0]
        * diff.shape[1]
    )

    return {
        "comparison_size": [
            diff.shape[1],
            diff.shape[0],
        ],

        "mae": round(
            float(diff.mean()),
            4,
        ),

        "exact_pixel_percent": round(
            float(
                np.all(
                    diff == 0,
                    axis=2,
                ).sum()
            )
            / pixel_count
            * 100,
            2,
        ),

        "within_5_percent": round(
            float(
                np.all(
                    diff <= 5,
                    axis=2,
                ).sum()
            )
            / pixel_count
            * 100,
            2,
        ),

        "within_20_percent": round(
            float(
                np.all(
                    diff <= 20,
                    axis=2,
                ).sum()
            )
            / pixel_count
            * 100,
            2,
        ),

        "significant_difference_percent": round(
            float(
                np.any(
                    diff > 20,
                    axis=2,
                ).sum()
            )
            / pixel_count
            * 100,
            2,
        ),

        "max_difference": int(
            diff.max()
        ),
    }


# ============================================================
# WRITE JSON CHECKPOINT
# ============================================================

def write_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary.replace(
        path
    )


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(
    pipeline,
    renderer,
    document,
    page_number: int,
) -> Dict[str, Any]:

    from PIL import Image

    page_index = (
        page_number - 1
    )

    start_time = time.perf_counter()

    source_image = None
    pipeline_result = None
    rendered_image = None
    page_model = None

    result = {
        "page": page_number,
        "status": "started",
        "memory_before_mb": (
            get_memory_mb()
        ),
    }

    print()
    print("=" * 72)
    print(
        f"PAGE {page_number}"
    )
    print("=" * 72)

    try:

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        print(
            "Step 1/3 - "
            "Rendering source page...",
            flush=True,
        )

        source_image = pipeline.render_page(
            document=document,
            page_number=page_index,
            dpi=DPI,
        )

        if not isinstance(
            source_image,
            Image.Image,
        ):
            raise TypeError(
                "render_page() did not "
                "return a PIL Image"
            )

        result["source_size"] = [
            source_image.width,
            source_image.height,
        ]

        print(
            f"  Source: "
            f"{source_image.width} x "
            f"{source_image.height}",
            flush=True,
        )

        print_memory(
            "after source render"
        )

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        print(
            "Step 2/3 - "
            "Running complete HandNote pipeline...",
            flush=True,
        )

        pipeline_result = (
            pipeline.analyze_image(
                image=source_image,
                page_number=page_number,
                language=LANGUAGE,
                psm=PSM,
            )
        )

        page_model = get_value(
            pipeline_result,
            "page",
            None,
        )

        if page_model is None:
            raise RuntimeError(
                "Pipeline returned no PageModel"
            )

        counts = get_pipeline_counts(
            pipeline_result
        )

        result["page_model"] = "OK"
        result["counts"] = counts

        print(
            "  Pipeline completed.",
            flush=True,
        )

        print(
            f"  Text blocks: "
            f"{counts['text_blocks']}",
            flush=True,
        )

        print(
            f"  Visual regions: "
            f"{counts['visual_regions']}",
            flush=True,
        )

        print(
            f"  Semantic regions: "
            f"{counts['semantic_regions']}",
            flush=True,
        )

        print(
            f"  Layout blocks: "
            f"{counts['layout_blocks']}",
            flush=True,
        )

        print_memory(
            "after pipeline"
        )

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        print(
            "Step 3/3 - "
            "Rendering with PageRenderer V3...",
            flush=True,
        )

        should_save = (
            SAVE_ALL_RENDERED
            or page_number in METRIC_PAGES
        )

        output_path = None

        if should_save:

            output_path = (
                RENDER_DIR
                / f"page_{page_number:03d}.png"
            )

        rendered_image = renderer.render(
            page_model=page_model,
            source_image=source_image,
            output_path=output_path,
        )

        if not isinstance(
            rendered_image,
            Image.Image,
        ):
            raise TypeError(
                "PageRenderer.render() "
                "did not return a PIL Image"
            )

        result["rendered_size"] = [
            rendered_image.width,
            rendered_image.height,
        ]

        if output_path is not None:

            result["output_path"] = str(
                output_path.resolve()
            )

            print(
                f"  Saved: {output_path}",
                flush=True,
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        if page_number in METRIC_PAGES:

            print(
                "  Calculating lightweight "
                "fidelity metrics...",
                flush=True,
            )

            result["metrics"] = (
                calculate_metrics(
                    source_image,
                    rendered_image,
                )
            )

            metrics = result["metrics"]

            print(
                f"  MAE: "
                f"{metrics['mae']}",
                flush=True,
            )

            print(
                f"  Exact: "
                f"{metrics['exact_pixel_percent']}%",
                flush=True,
            )

            print(
                f"  Within ±5: "
                f"{metrics['within_5_percent']}%",
                flush=True,
            )

            print(
                f"  Within ±20: "
                f"{metrics['within_20_percent']}%",
                flush=True,
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result["status"] = "passed"

        result["seconds"] = round(
            elapsed,
            3,
        )

        print()
        print(
            f"PAGE {page_number} "
            f"PASSED "
            f"in {elapsed:.2f}s",
            flush=True,
        )

        print_memory(
            "before cleanup"
        )

        return result

    except MemoryError as error:

        result["status"] = (
            "memory_error"
        )

        result["error"] = repr(
            error
        )

        result["traceback"] = (
            traceback.format_exc()
        )

        print()
        print(
            f"!!! PAGE {page_number} "
            "MEMORY ERROR !!!",
            flush=True,
        )

        print(
            traceback.format_exc(),
            flush=True,
        )

        return result

    except Exception as error:

        result["status"] = "failed"

        result["error"] = repr(
            error
        )

        result["traceback"] = (
            traceback.format_exc()
        )

        print()
        print(
            f"!!! PAGE {page_number} "
            "FAILED !!!",
            flush=True,
        )

        print(
            traceback.format_exc(),
            flush=True,
        )

        return result

    finally:

        # ----------------------------------------------------
        # IMPORTANT MEMORY CLEANUP
        # ----------------------------------------------------

        try:
            if rendered_image is not None:
                rendered_image.close()
        except Exception:
            pass

        try:
            if source_image is not None:
                source_image.close()
        except Exception:
            pass

        rendered_image = None
        source_image = None
        pipeline_result = None
        page_model = None

        gc.collect()

        print_memory(
            "after cleanup"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 72)
    print(
        "       HANDNOTE AI - TARGETED V3 TEST"
    )
    print("=" * 72)
    print()

    project_root = (
        Path.cwd().resolve()
    )

    print(
        f"Project root : "
        f"{project_root}"
    )

    print(
        f"Page range   : "
        f"{PAGE_START} -> "
        f"{PAGE_END if PAGE_END is not None else 'END'}"
    )

    print(
        f"DPI          : "
        f"{DPI}"
    )

    print(
        f"Language     : "
        f"{LANGUAGE}"
    )

    print(
        f"PSM          : "
        f"{PSM}"
    )

    print()

    # --------------------------------------------------------
    # IMPORTS
    # --------------------------------------------------------

    try:
        import fitz
    except ImportError:

        print(
            "ERROR: PyMuPDF/fitz is not installed."
        )

        return 1

    try:

        from backend.services.pipeline import (
            HandNotePagePipeline,
        )

    except Exception:

        print(
            "ERROR: Could not import "
            "HandNotePagePipeline."
        )

        traceback.print_exc()

        return 1

    try:

        from backend.services.page_renderer import (
            PageRenderer,
        )

    except Exception:

        print(
            "ERROR: Could not import "
            "PageRenderer."
        )

        traceback.print_exc()

        return 1

    # --------------------------------------------------------
    # FIND PDF
    # --------------------------------------------------------

    try:

        pdf_path = find_pdf(
            project_root
        )

    except Exception:

        traceback.print_exc()

        return 1

    print(
        f"PDF          : "
        f"{pdf_path}"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RENDER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # OPEN PDF ONLY ONCE
    # --------------------------------------------------------

    document = None

    try:

        document = fitz.open(
            pdf_path
        )

        total_pages = len(
            document
        )

        print(
            f"PDF pages   : "
            f"{total_pages}"
        )

        if total_pages == 0:

            print(
                "ERROR: PDF has no pages."
            )

            return 1

        start_page = max(
            1,
            int(PAGE_START),
        )

        if PAGE_END is None:

            end_page = total_pages

        else:

            end_page = min(
                total_pages,
                int(PAGE_END),
            )

        if start_page > end_page:

            print(
                "ERROR: Invalid page range."
            )

            return 1

        print(
            f"Testing      : "
            f"pages {start_page}-{end_page}"
        )

        print()

        # ----------------------------------------------------
        # CREATE OBJECTS ONCE
        # ----------------------------------------------------

        pipeline = (
            HandNotePagePipeline()
        )

        renderer = (
            PageRenderer()
        )

        results: List[
            Dict[str, Any]
        ] = []

        checkpoint = {
            "pdf": str(pdf_path),
            "total_pdf_pages": total_pages,
            "page_start": start_page,
            "page_end": end_page,
            "dpi": DPI,
            "language": LANGUAGE,
            "psm": PSM,
            "results": results,
        }

        write_json(
            PROGRESS_FILE,
            checkpoint,
        )

        overall_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # PROCESS PAGES
        # ----------------------------------------------------

        for page_number in range(
            start_page,
            end_page + 1,
        ):

            page_result = process_page(
                pipeline=pipeline,
                renderer=renderer,
                document=document,
                page_number=page_number,
            )

            results.append(
                page_result
            )

            # Save checkpoint immediately.
            checkpoint["results"] = (
                results
            )

            checkpoint[
                "last_processed_page"
            ] = page_number

            write_json(
                PROGRESS_FILE,
                checkpoint,
            )

            gc.collect()

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        total_seconds = (
            time.perf_counter()
            - overall_start
        )

        passed = [
            item
            for item in results
            if item["status"]
            == "passed"
        ]

        failed = [
            item
            for item in results
            if item["status"]
            != "passed"
        ]

        summary = {
            "pdf": str(pdf_path),
            "total_pdf_pages": total_pages,
            "tested_pages": len(results),
            "page_start": start_page,
            "page_end": end_page,
            "passed": len(passed),
            "failed": len(failed),
            "total_seconds": round(
                total_seconds,
                3,
            ),
            "average_seconds_per_page": round(
                total_seconds
                / max(
                    1,
                    len(results),
                ),
                3,
            ),
            "final_memory_mb": (
                get_memory_mb()
            ),
            "results": results,
        }

        write_json(
            SUMMARY_FILE,
            summary,
        )

        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        print()
        print("=" * 72)
        print(
            "                       SUMMARY"
        )
        print("=" * 72)

        print(
            f"PDF              : "
            f"{pdf_path.name}"
        )

        print(
            f"Pages tested     : "
            f"{len(results)}"
        )

        print(
            f"Pages passed     : "
            f"{len(passed)}"
        )

        print(
            f"Pages failed     : "
            f"{len(failed)}"
        )

        print(
            f"Total time       : "
            f"{total_seconds:.2f}s"
        )

        if failed:

            print()
            print(
                "FAILED PAGES:"
            )

            for item in failed:

                print(
                    f"  Page "
                    f"{item.get('page')}: "
                    f"{item.get('status')} - "
                    f"{item.get('error')}"
                )

        print()
        print(
            f"Progress file    : "
            f"{PROGRESS_FILE}"
        )

        print(
            f"Summary file     : "
            f"{SUMMARY_FILE}"
        )

        print(
            f"Rendered pages   : "
            f"{RENDER_DIR}"
        )

        print("=" * 72)

        if failed:
            return 2

        return 0

    except KeyboardInterrupt:

        print()
        print(
            "CTRL+C received."
        )

        print(
            "Test stopped safely."
        )

        print(
            f"Progress is saved in:"
            f"\n{PROGRESS_FILE}"
        )

        return 130

    except Exception:

        print()
        print(
            "FATAL TEST ERROR"
        )

        traceback.print_exc()

        return 1

    finally:

        if document is not None:

            try:
                document.close()
            except Exception:
                pass

        gc.collect()

        print_memory(
            "final"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )