from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


BBox = Tuple[int, int, int, int]


# ============================================================
# HELPERS
# ============================================================

def _value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _bbox(obj: Any) -> BBox:
    """Return x, y, width, height from a region/block-like object."""
    return (
        int(_value(obj, "x", 0) or 0),
        int(_value(obj, "y", 0) or 0),
        int(_value(obj, "width", 0) or 0),
        int(_value(obj, "height", 0) or 0),
    )


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x, y, w, h = bbox

    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(x1, min(width, x + max(0, w)))
    y2 = max(y1, min(height, y + max(0, h)))

    return x1, y1, x2 - x1, y2 - y1


def _intersection_area(a: BBox, b: BBox) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if right <= left or bottom <= top:
        return 0

    return (right - left) * (bottom - top)


def _area(bbox: BBox) -> int:
    return max(0, bbox[2]) * max(0, bbox[3])


# ============================================================
# PAGE RENDERER
# ============================================================

class PageRenderer:
    """
    Reconstruct a HandNote AI PageModel onto a clean canvas.

    Architecture:

        OCR
          -> text blocks
        visual detection
          -> visual regions
        visual classification
        semantic grouping
        layout analysis
          -> PageModel
        PageRenderer
          -> reconstructed image

    Renderer V2 goals:

    1. Never paste a complete source crop.
    2. Preserve visual foreground from the source image.
    3. Preserve original text pixels when possible instead of
       redrawing OCR text with a generic font.
    4. Fall back to OCR text rendering only when source pixels
       cannot be used.
    5. Keep compatibility with the existing PageRenderer API.
    """

    def __init__(
        self,
        background: Union[str, Tuple[int, int, int]] = "auto",
        foreground_threshold: int = 18,
        alpha_threshold: int = 8,
        visual_padding: int = 0,
        text_padding: int = 3,
        render_text: bool = True,
        preserve_highlights: bool = True,
        preserve_handwriting: bool = True,
        preserve_graphics: bool = True,
        preserve_diagrams: bool = True,
        preserve_source_text: bool = True,
        debug: bool = False,
    ):
        self.background = background
        self.foreground_threshold = max(1, int(foreground_threshold))
        self.alpha_threshold = max(1, int(alpha_threshold))

        self.visual_padding = max(0, int(visual_padding))
        self.text_padding = max(0, int(text_padding))

        self.render_text = bool(render_text)

        self.preserve_highlights = bool(preserve_highlights)
        self.preserve_handwriting = bool(preserve_handwriting)
        self.preserve_graphics = bool(preserve_graphics)
        self.preserve_diagrams = bool(preserve_diagrams)

        # V2: preserve source text pixels before using font rendering.
        self.preserve_source_text = bool(preserve_source_text)

        # Existing tests pass debug=False.
        self.debug = bool(debug)

        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

    # ========================================================
    # PAGE MODEL ACCESS
    # ========================================================

    @staticmethod
    def get_page_model(page_model: Any) -> Any:
        """
        Accept either:

            PageModel
            PagePipelineResult.page
        """
        if page_model is None:
            raise ValueError("page_model cannot be None")

        page = _value(page_model, "page", None)

        if page is not None:
            return page

        return page_model

    # ========================================================
    # BACKGROUND DETECTION
    # ========================================================

    @staticmethod
    def _corner_pixels(
        image: Image.Image,
        size: int = 80,
    ) -> np.ndarray:
        """Collect pixels from the four page corners."""
        rgb = image.convert("RGB")
        array = np.asarray(rgb)

        h, w = array.shape[:2]

        s = max(
            1,
            min(size, h // 8, w // 8),
        )

        pieces = [
            array[:s, :s],
            array[:s, w - s:w],
            array[h - s:h, :s],
            array[h - s:h, w - s:w],
        ]

        return np.concatenate(
            [p.reshape(-1, 3) for p in pieces],
            axis=0,
        )

    def estimate_background(
        self,
        image: Image.Image,
    ) -> Tuple[int, int, int]:
        """
        Estimate page background.

        Supports dark pages and light pages.
        """
        if isinstance(self.background, tuple):
            return tuple(
                int(max(0, min(255, v)))
                for v in self.background
            )

        mode = str(self.background).lower().strip()

        if mode in {"white", "light"}:
            return (255, 255, 255)

        if mode in {"black", "dark"}:
            return (0, 0, 0)

        pixels = self._corner_pixels(image)

        array = np.asarray(image.convert("RGB"))

        h, w = array.shape[:2]

        border = max(
            1,
            min(12, h // 10, w // 10),
        )

        border_parts = [
            array[:border, :].reshape(-1, 3),
            array[h - border:h, :].reshape(-1, 3),
            array[:, :border].reshape(-1, 3),
            array[:, w - border:w].reshape(-1, 3),
        ]

        pixels = np.concatenate(
            [pixels, *border_parts],
            axis=0,
        )

        median = np.median(pixels, axis=0)

        if float(np.mean(median)) < 35:
            return (0, 0, 0)

        if float(np.mean(median)) > 220:
            return (255, 255, 255)

        return tuple(int(round(v)) for v in median)

    # ========================================================
    # MASKS
    # ========================================================

    def create_foreground_mask(
        self,
        crop: Image.Image,
        background_rgb: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Create a soft foreground alpha mask based on color distance
        from the supplied background.

        This preserves:
            - white ink on black
            - black ink on white
            - yellow/orange highlights
            - colored handwriting
            - diagrams
            - graphics
        """
        rgb = np.asarray(
            crop.convert("RGB"),
            dtype=np.float32,
        )

        bg = np.asarray(
            background_rgb,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        distance = np.linalg.norm(rgb - bg, axis=2)

        threshold = float(self.foreground_threshold)

        alpha = np.clip(
            (distance - threshold)
            / max(1.0, 255.0 - threshold),
            0.0,
            1.0,
        )

        strong = distance >= threshold * 2.2

        alpha[strong] = np.maximum(
            alpha[strong],
            0.75,
        )

        alpha[distance < self.alpha_threshold] = 0.0

        return (alpha * 255.0).astype(np.uint8)

    def extract_foreground(
        self,
        source_image: Image.Image,
        bbox: BBox,
        background_rgb: Tuple[int, int, int],
        padding: int = 0,
    ) -> Optional[Image.Image]:
        """
        Extract only foreground pixels from a source region.

        The returned image is RGBA with the estimated background
        made transparent.
        """
        x, y, w, h = _clip_bbox(
            (
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding * 2,
                bbox[3] + padding * 2,
            ),
            source_image.width,
            source_image.height,
        )

        if w <= 0 or h <= 0:
            return None

        crop = source_image.crop(
            (x, y, x + w, y + h)
        ).convert("RGB")

        alpha = self.create_foreground_mask(
            crop,
            background_rgb,
        )

        alpha_image = Image.fromarray(
            alpha,
            mode="L",
        )

        alpha_image = alpha_image.filter(
            ImageFilter.GaussianBlur(radius=0.20)
        )

        rgba = crop.convert("RGBA")
        rgba.putalpha(alpha_image)

        return rgba

    # ========================================================
    # SOURCE TEXT PRESERVATION
    # ========================================================

    def extract_source_text(
        self,
        source_image: Image.Image,
        bbox: BBox,
        background_rgb: Tuple[int, int, int],
    ) -> Optional[Image.Image]:
        """
        V2 text extraction.

        Instead of recreating OCR text with Arial/DejaVu, use the
        original text pixels from the source image and make only the
        page background transparent.

        Because the output canvas uses the same background color,
        compositing these pixels preserves the original typography,
        antialiasing and stroke thickness much more accurately.
        """
        x, y, w, h = _clip_bbox(
            bbox,
            source_image.width,
            source_image.height,
        )

        if w <= 0 or h <= 0:
            return None

        crop = source_image.crop(
            (x, y, x + w, y + h)
        ).convert("RGB")

        alpha = self.create_foreground_mask(
            crop,
            background_rgb,
        )

        # Text needs a slightly more faithful mask than visual
        # regions. Keep antialiased edge pixels.
        alpha_image = Image.fromarray(
            alpha,
            mode="L",
        )

        rgba = crop.convert("RGBA")
        rgba.putalpha(alpha_image)

        return rgba

    def _source_text_ink_ratio(
        self,
        source_image: Image.Image,
        bbox: BBox,
        background_rgb: Tuple[int, int, int],
    ) -> float:
        """
        Estimate whether a text bbox actually contains useful source
        pixels.

        This prevents empty/incorrect OCR boxes from producing a
        visible artifact.
        """
        x, y, w, h = _clip_bbox(
            bbox,
            source_image.width,
            source_image.height,
        )

        if w <= 0 or h <= 0:
            return 0.0

        crop = source_image.crop(
            (x, y, x + w, y + h)
        ).convert("RGB")

        alpha = self.create_foreground_mask(
            crop,
            background_rgb,
        )

        return float(np.count_nonzero(alpha > 20)) / float(max(1, w * h))

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    @staticmethod
    def build_visual_classification_map(
        classifications: Optional[Iterable[Any]],
    ) -> Dict[int, str]:
        result: Dict[int, str] = {}

        for item in classifications or []:
            region_id = _value(item, "region_id", 0)

            try:
                region_id = int(region_id)
            except (ValueError, TypeError):
                continue

            label = str(
                _value(
                    item,
                    "classification",
                    "",
                )
            ).lower().strip()

            if region_id:
                result[region_id] = label

        return result

    def should_render_visual(
        self,
        classification: str,
    ) -> bool:
        label = str(
            classification or ""
        ).lower().strip()

        if label == "highlight":
            return self.preserve_highlights

        if label == "handwriting":
            return self.preserve_handwriting

        if label == "graphic":
            return self.preserve_graphics

        if label == "diagram":
            return self.preserve_diagrams

        return True

    # ========================================================
    # OCR COVERAGE
    # ========================================================

    @staticmethod
    def ocr_overlap_ratio(
        bbox: BBox,
        ocr_words: Iterable[Any],
    ) -> float:
        region_area = _area(bbox)

        if region_area <= 0:
            return 0.0

        overlap = 0

        for word in ocr_words or []:
            word_bbox = (
                int(_value(word, "left", 0) or 0),
                int(_value(word, "top", 0) or 0),
                int(_value(word, "width", 0) or 0),
                int(_value(word, "height", 0) or 0),
            )

            overlap += _intersection_area(
                bbox,
                word_bbox,
            )

        return min(
            1.0,
            overlap / float(region_area),
        )

    @staticmethod
    def build_visual_coverage(
        image_size: Tuple[int, int],
        regions: Iterable[Any],
    ) -> np.ndarray:
        """
        Build a binary visual-region coverage map.
        """
        width, height = image_size

        coverage = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        for region in regions or []:
            x, y, w, h = _clip_bbox(
                _bbox(region),
                width,
                height,
            )

            if w <= 0 or h <= 0:
                continue

            coverage[
                y:y + h,
                x:x + w,
            ] = 1

        return coverage

    # ========================================================
    # FONT FALLBACK
    # ========================================================

    def _load_font(self, size: int):
        """
        Font fallback used only when source-pixel text preservation
        is not possible.
        """
        size = max(8, int(size))

        if size in self._font_cache:
            return self._font_cache[size]

        candidates = [
            # Windows
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\calibri.ttf",

            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

        font = None

        for path in candidates:
            if Path(path).exists():
                try:
                    font = ImageFont.truetype(
                        path,
                        size=size,
                    )
                    break
                except OSError:
                    pass

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[size] = font

        return font

    # ========================================================
    # TEXT COLOR
    # ========================================================

    @staticmethod
    def choose_text_color(
        background_rgb: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        brightness = (
            0.299 * background_rgb[0]
            + 0.587 * background_rgb[1]
            + 0.114 * background_rgb[2]
        )

        if brightness < 128:
            return (255, 255, 255)

        return (0, 0, 0)

    # ========================================================
    # TEXT RENDERING
    # ========================================================

    def _fallback_render_text(
        self,
        output: Image.Image,
        block: Any,
        text: str,
        bbox: BBox,
        background_rgb: Tuple[int, int, int],
    ) -> None:
        """
        Last-resort OCR text renderer.
        """
        draw = ImageDraw.Draw(output)

        confidence = float(
            _value(block, "confidence", 0.0) or 0.0
        )

        classification = str(
            _value(
                block,
                "classification",
                _value(block, "content_type", ""),
            )
        ).lower().strip()

        if (
            classification == "uncertain_text"
            and confidence < 45.0
        ):
            if bbox[2] < 80 or bbox[3] < 25:
                return

        font_size = max(
            12,
            int(bbox[3] * 0.72),
        )

        font = self._load_font(font_size)

        draw.multiline_text(
            (bbox[0], bbox[1]),
            text,
            fill=self.choose_text_color(background_rgb),
            font=font,
            spacing=max(
                2,
                int(font_size * 0.18),
            ),
        )

    def render_text_blocks(
        self,
        output: Image.Image,
        text_blocks: Iterable[Any],
        visual_regions: Iterable[Any],
        background_rgb: Tuple[int, int, int],
        source_image: Optional[Image.Image] = None,
    ) -> None:
        """
        Render OCR text.

        V2 priority:

            source text pixels
                    ↓
            OCR/font fallback

        This is the main fidelity improvement.
        """
        if not self.render_text:
            return

        visual_boxes = [
            _bbox(region)
            for region in (visual_regions or [])
        ]

        for block in text_blocks or []:
            text = str(
                _value(
                    block,
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            bbox = _bbox(block)

            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            block_area = max(1, _area(bbox))

            covered = sum(
                _intersection_area(
                    bbox,
                    vb,
                )
                for vb in visual_boxes
            )

            coverage = min(
                1.0,
                covered / float(block_area),
            )

            # Visual regions already contain the source pixels.
            if coverage >= 0.35:
                continue

            # ----------------------------------------------------
            # V2: use exact source text pixels
            # ----------------------------------------------------
            if (
                self.preserve_source_text
                and source_image is not None
            ):
                ink_ratio = self._source_text_ink_ratio(
                    source_image,
                    bbox,
                    background_rgb,
                )

                # Text normally occupies only a fraction of its bbox.
                # A tiny ratio means this OCR box is probably wrong.
                if ink_ratio >= 0.005:
                    source_text = self.extract_source_text(
                        source_image,
                        bbox,
                        background_rgb,
                    )

                    if source_text is not None:
                        output.alpha_composite(
                            source_text,
                            (
                                bbox[0],
                                bbox[1],
                            ),
                        )
                        continue

            # ----------------------------------------------------
            # Fallback
            # ----------------------------------------------------
            self._fallback_render_text(
                output=output,
                block=block,
                text=text,
                bbox=bbox,
                background_rgb=background_rgb,
            )

    # ========================================================
    # VISUAL REGION RENDERING
    # ========================================================

    def render_visual_regions(
        self,
        output: Image.Image,
        source_image: Image.Image,
        visual_regions: Iterable[Any],
        visual_classifications: Optional[Iterable[Any]],
        background_rgb: Tuple[int, int, int],
    ) -> None:
        """
        Render visual regions as transparent foreground pixels.

        Complete source crops are never pasted.
        """
        classification_map = (
            self.build_visual_classification_map(
                visual_classifications
            )
        )

        seen = set()

        for region in visual_regions or []:
            bbox = _bbox(region)

            if _area(bbox) <= 0:
                continue

            region_id = int(
                _value(
                    region,
                    "region_id",
                    0,
                ) or 0
            )

            classification = classification_map.get(
                region_id,
                str(
                    _value(
                        region,
                        "classification",
                        "",
                    )
                ).lower().strip(),
            )

            if not self.should_render_visual(
                classification
            ):
                continue

            key = (
                bbox,
                classification,
            )

            if key in seen:
                continue

            seen.add(key)

            # Highlights are page-level foreground.
            if classification == "highlight":
                local_background = background_rgb
            else:
                crop_x, crop_y, crop_w, crop_h = _clip_bbox(
                    bbox,
                    source_image.width,
                    source_image.height,
                )

                if crop_w <= 0 or crop_h <= 0:
                    continue

                local_crop = source_image.crop(
                    (
                        crop_x,
                        crop_y,
                        crop_x + crop_w,
                        crop_y + crop_h,
                    )
                )

                local_background = self.estimate_background(
                    local_crop
                )

            foreground = self.extract_foreground(
                source_image=source_image,
                bbox=bbox,
                background_rgb=local_background,
                padding=self.visual_padding,
            )

            if foreground is None:
                continue

            x = bbox[0] - self.visual_padding
            y = bbox[1] - self.visual_padding

            # Keep compositing inside the page.
            if x < 0 or y < 0:
                # Re-crop if padding crosses the page boundary.
                left_crop = max(0, -x)
                top_crop = max(0, -y)

                foreground = foreground.crop(
                    (
                        left_crop,
                        top_crop,
                        foreground.width,
                        foreground.height,
                    )
                )

                x = max(0, x)
                y = max(0, y)

            if foreground.width <= 0 or foreground.height <= 0:
                continue

            output.alpha_composite(
                foreground,
                (x, y),
            )

    # ========================================================
    # MAIN RENDER
    # ========================================================

    def render(
        self,
        page_model: Any,
        source_image: Image.Image,
        *,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        Render a PageModel.

        Supported:

            renderer.render(page, image)

            renderer.render(
                page,
                image,
                output_path="output.png",
            )
        """
        page = self.get_page_model(page_model)

        if not isinstance(source_image, Image.Image):
            raise TypeError(
                "source_image must be a PIL.Image.Image"
            )

        source = source_image.convert("RGB")

        background_rgb = self.estimate_background(
            source
        )

        width = int(
            _value(
                page,
                "image_width",
                source.width,
            )
            or source.width
        )

        height = int(
            _value(
                page,
                "image_height",
                source.height,
            )
            or source.height
        )

        # Region coordinates are based on the actual source image.
        if width != source.width or height != source.height:
            width = source.width
            height = source.height

        output = Image.new(
            "RGBA",
            (width, height),
            (*background_rgb, 255),
        )

        visual_regions = _value(
            page,
            "visual_regions",
            [],
        ) or []

        visual_classifications = _value(
            page,
            "visual_classifications",
            [],
        ) or []

        text_blocks = _value(
            page,
            "text_blocks",
            [],
        ) or []

        # --------------------------------------------------------
        # Layer 1: visuals
        # --------------------------------------------------------

        self.render_visual_regions(
            output=output,
            source_image=source,
            visual_regions=visual_regions,
            visual_classifications=visual_classifications,
            background_rgb=background_rgb,
        )

        # --------------------------------------------------------
        # Layer 2: text
        # --------------------------------------------------------

        self.render_text_blocks(
            output=output,
            text_blocks=text_blocks,
            visual_regions=visual_regions,
            background_rgb=background_rgb,
            source_image=source,
        )

        # --------------------------------------------------------
        # Final image
        # --------------------------------------------------------

        final_image = output.convert("RGB")

        if output_path is not None:
            self.save(
                final_image,
                output_path,
            )

        return final_image

    # ========================================================
    # COMPATIBILITY ALIASES
    # ========================================================

    def render_page(
        self,
        page_model: Any,
        source_image: Image.Image,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """Compatibility alias for render()."""
        return self.render(
            page_model=page_model,
            source_image=source_image,
            output_path=output_path,
        )

    def render_reconstructed_page(
        self,
        page_model: Any,
        source_image: Image.Image,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """Compatibility alias for render()."""
        return self.render(
            page_model=page_model,
            source_image=source_image,
            output_path=output_path,
        )

    def render_to_file(
        self,
        page_model: Any,
        source_image: Union[Image.Image, str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        Render and save in one operation.

        Preferred:

            renderer.render_to_file(
                page_model,
                source_image,
                output_path,
            )

        Compatibility form also supported:

            renderer.render_to_file(
                rendered_image,
                output_path,
            )

        The second form is useful for older test scripts that first
        call render_page() and then ask the renderer to save the
        returned PIL image.
        """
        if isinstance(page_model, Image.Image):
            rendered_image = page_model

            # Compatibility call:
            # render_to_file(rendered_image, output_path)
            if output_path is None:
                output_path = source_image

            if output_path is None:
                raise ValueError(
                    "output_path is required when saving a rendered image"
                )

            self.save(
                rendered_image,
                output_path,
            )

            return rendered_image

        if output_path is None:
            raise TypeError(
                "render_to_file() requires "
                "(page_model, source_image, output_path)"
            )

        if not isinstance(source_image, Image.Image):
            raise TypeError(
                "source_image must be a PIL.Image.Image"
            )

        return self.render(
            page_model=page_model,
            source_image=source_image,
            output_path=output_path,
        )

    def save(
        self,
        image: Image.Image,
        output_path: Union[str, Path],
    ) -> str:
        """
        Save a rendered image.

        This method intentionally belongs to PageRenderer so both
        old and new test callers work:

            renderer.save(image, path)
        """
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

        image.save(
            output_path,
            format="PNG",
        )

        return str(
            output_path.resolve()
        )


# ============================================================
# FUNCTIONAL API
# ============================================================

def render_page(
    page_model: Any,
    source_image: Image.Image,
    output_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Functional API for callers that do not want to instantiate
    PageRenderer explicitly.
    """
    renderer = PageRenderer()

    return renderer.render(
        page_model=page_model,
        source_image=source_image,
        output_path=output_path,
    )
