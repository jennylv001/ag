from __future__ import annotations

"""Server-side screenshot overlays for debug/inspection (no DOM mutation).

This module provides utilities to composite clickable element highlights onto a
captured screenshot image entirely on the server side. It never touches the
live page DOM and is safe for stealthy runs when enabled explicitly.
"""

import base64
import io
import logging
from typing import Iterable

from browser_use.dom.views import SelectorMap

# Log a single warning if Pillow is unavailable to avoid noisy logs
_PILLOW_MISSING_WARNED = False

logger = logging.getLogger(__name__)


def _iter_elements_with_viewport_coords(selector_map: SelectorMap) -> Iterable[tuple[int, int, dict]]:
    """Yield (label_index, sort_index, coords_dict) for elements that have viewport_coordinates.

    coords_dict schema expected:
      {
        "top_left": {"x": int, "y": int},
        "bottom_right": {"x": int, "y": int},
        "center": {"x": int, "y": int},
        "width": int,
        "height": int,
      }
    """
    for _, node in selector_map.items():
        vc = getattr(node, "viewport_coordinates", None)
        hi = getattr(node, "highlight_index", None)
        if vc is None or hi is None:
            continue
        try:
            data = vc.model_dump()  # pydantic BaseModel -> dict
            yield int(hi), int(hi), data
        except Exception:
            # If not a pydantic model (unlikely), try best-effort dict access
            try:
                data = {
                    "top_left": {"x": vc.top_left.x, "y": vc.top_left.y},
                    "bottom_right": {"x": vc.bottom_right.x, "y": vc.bottom_right.y},
                    "center": {"x": vc.center.x, "y": vc.center.y},
                    "width": int(getattr(vc, "width")),
                    "height": int(getattr(vc, "height")),
                }
                yield int(hi), int(hi), data
            except Exception:
                continue


def overlay_highlights_on_screenshot(
    screenshot_b64: str,
    selector_map: SelectorMap,
    *,
    max_items: int = 60,
    box_color: tuple[int, int, int] = (255, 0, 0),
    box_alpha: int = 120,
    label_bg: tuple[int, int, int] = (0, 0, 0),
    label_fg: tuple[int, int, int] = (255, 255, 255),
    label_padding: int = 4,
    viewport_width: int | None = None,
    viewport_height: int | None = None,
    scroll_x: int | None = None,
    scroll_y: int | None = None,
) -> str:
    """Return a new base64 screenshot with clickable element boxes overlaid.

    - Draws semi-transparent rectangles around elements found in selector_map
      using viewport_coordinates relative to the screenshot viewport.
    - Draws a small label with the element index near the top-left corner of
      each rectangle.
    - If Pillow is not available, returns the original screenshot unchanged.
    """
    if not screenshot_b64:
        return screenshot_b64

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        global _PILLOW_MISSING_WARNED
        if not _PILLOW_MISSING_WARNED:
            try:
                logger.warning("Pillow not available; skipping screenshot overlays (will not log again)")
            except Exception:
                pass
            _PILLOW_MISSING_WARNED = True
        # Continue silently on subsequent calls
        return screenshot_b64

    try:
        img_bytes = base64.b64decode(screenshot_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Choose a reasonable default font; fall back to load_default
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None  # type: ignore

        # Determine DPR scaling (if screenshot image size differs from viewport CSS pixels)
        # If viewport is missing, assume 1:1 CSS px to image px.
        scale_x = 1.0
        scale_y = 1.0
        if viewport_width and viewport_height and viewport_width > 0 and viewport_height > 0:
            try:
                scale_x = image.width / float(viewport_width)
                scale_y = image.height / float(viewport_height)
            except Exception:
                scale_x = scale_y = 1.0

        count = 0
        drawn_any = False
        for label_idx, sort_idx, coords in sorted(_iter_elements_with_viewport_coords(selector_map), key=lambda t: t[1]):
                if count >= max_items:
                    break
                tl = coords.get("top_left", {})
                br = coords.get("bottom_right", {})
                x1, y1 = int(tl.get("x", 0)), int(tl.get("y", 0))
                x2, y2 = int(br.get("x", 0)), int(br.get("y", 0))

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Scale from viewport CSS pixels to image pixels
                x1 = int(round(x1 * scale_x))
                y1 = int(round(y1 * scale_y))
                x2 = int(round(x2 * scale_x))
                y2 = int(round(y2 * scale_y))

                # Skip fully off-screen boxes
                if x2 < 0 or y2 < 0 or x1 > image.width or y1 > image.height:
                    continue

                # Clamp to image bounds
                x1 = max(0, min(x1, image.width - 1))
                y1 = max(0, min(y1, image.height - 1))
                x2 = max(0, min(x2, image.width))
                y2 = max(0, min(y2, image.height))

                # Semi-transparent fill + solid border
                fill = (*box_color, box_alpha)
                outline = (*box_color, 255)
                draw.rectangle([(x1, y1), (x2, y2)], outline=outline, width=2, fill=fill)

                # Label background near top-left within the box
                label_text = str(label_idx)
                try:
                    text_bbox = draw.textbbox((0, 0), label_text, font=font) if font else draw.textbbox((0, 0), label_text)
                    tw = text_bbox[2] - text_bbox[0]
                    th = text_bbox[3] - text_bbox[1]
                except Exception:
                    # Fallback sizes
                    tw, th = 8 * max(len(label_text), 1), 12
                bg_w = tw + 2 * label_padding
                bg_h = th + 2 * label_padding
                bg_x1 = x1 + 2
                bg_y1 = y1 + 2
                bg_x2 = min(bg_x1 + bg_w, image.width - 1)
                bg_y2 = min(bg_y1 + bg_h, image.height - 1)
                draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=(*label_bg, 200))

                # Center text inside the label box
                tx = bg_x1 + label_padding
                ty = bg_y1 + label_padding
                try:
                    draw.text((tx, ty), label_text, fill=label_fg, font=font)
                except Exception:
                    draw.text((tx, ty), label_text, fill=label_fg)

                count += 1
                drawn_any = True

        # Fallback: use page_coordinates if no viewport_coordinates were drawn
        if not drawn_any:
            try:
                sx = int(scroll_x or 0)
                sy = int(scroll_y or 0)
            except Exception:
                sx = sy = 0

            for _, node in selector_map.items():
                hi = getattr(node, "highlight_index", None)
                pc = getattr(node, "page_coordinates", None)
                if hi is None or pc is None:
                    continue
                try:
                    data = pc.model_dump()
                except Exception:
                    try:
                        data = {
                            "top_left": {"x": pc.top_left.x, "y": pc.top_left.y},
                            "bottom_right": {"x": pc.bottom_right.x, "y": pc.bottom_right.y},
                        }
                    except Exception:
                        continue

                tl = data.get("top_left", {})
                br = data.get("bottom_right", {})
                # Convert absolute page coords to viewport coords by subtracting scroll
                x1, y1 = int(tl.get("x", 0)) - sx, int(tl.get("y", 0)) - sy
                x2, y2 = int(br.get("x", 0)) - sx, int(br.get("y", 0)) - sy

                if x2 <= x1 or y2 <= y1:
                    continue

                # Apply DPR scaling
                x1 = int(round(x1 * scale_x))
                y1 = int(round(y1 * scale_y))
                x2 = int(round(x2 * scale_x))
                y2 = int(round(y2 * scale_y))

                if x2 < 0 or y2 < 0 or x1 > image.width or y1 > image.height:
                    continue

                x1 = max(0, min(x1, image.width - 1))
                y1 = max(0, min(y1, image.height - 1))
                x2 = max(0, min(x2, image.width))
                y2 = max(0, min(y2, image.height))

                fill = (*box_color, box_alpha)
                outline = (*box_color, 255)
                draw.rectangle([(x1, y1), (x2, y2)], outline=outline, width=2, fill=fill)

                label_text = str(int(hi))
                try:
                    text_bbox = draw.textbbox((0, 0), label_text, font=font) if font else draw.textbbox((0, 0), label_text)
                    tw = text_bbox[2] - text_bbox[0]
                    th = text_bbox[3] - text_bbox[1]
                except Exception:
                    tw, th = 8 * max(len(label_text), 1), 12
                bg_w = tw + 2 * label_padding
                bg_h = th + 2 * label_padding
                bg_x1 = x1 + 2
                bg_y1 = y1 + 2
                bg_x2 = min(bg_x1 + bg_w, image.width - 1)
                bg_y2 = min(bg_y1 + bg_h, image.height - 1)
                draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=(*label_bg, 200))

                tx = bg_x1 + label_padding
                ty = bg_y1 + label_padding
                try:
                    draw.text((tx, ty), label_text, fill=label_fg, font=font)
                except Exception:
                    draw.text((tx, ty), label_text, fill=label_fg)

                count += 1
                if count >= max_items:
                    break

        if count == 0:
            return screenshot_b64

        composited = Image.alpha_composite(image, overlay).convert("RGB")
        out = io.BytesIO()
        composited.save(out, format="PNG", optimize=False)
        try:
            logger.debug(f"🧩 Overlaid {count} highlight box(es) on screenshot")
        except Exception:
            pass
        return base64.b64encode(out.getvalue()).decode("utf-8")

    except Exception as e:
        logger.debug(f"Failed to composite highlights on screenshot: {type(e).__name__}: {e}")
        return screenshot_b64
