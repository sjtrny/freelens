"""Cairo Visuals source for the FreeLens README pipeline illustration.

The diagram is intentionally data-driven: it rectifies the reviewed corners from
``dataset/evaluation.json`` and regenerates the same tag with
``Tag.from_message``. Render it at 1920 x 820 from the repository checkout.
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import cairo
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from freelens import Tag  # noqa: E402
from scripts.tag_editor import _perspective_coefficients  # noqa: E402

try:
    from cairo_font import DEFAULT_FONT, set_font_from_file
except ImportError as error:  # pragma: no cover - supplied by Cairo Visuals
    raise RuntimeError("Render this diagram with the cairo-visuals project") from error

WIDTH = 1920
HEIGHT = 680

PHOTO_X = 52
PHOTO_Y = 72
PHOTO_SIZE = 535
PHOTO_CROP = (1000, 1200, 2200, 2400)

RECTIFIED_X = 750
RECTIFIED_Y = 183
TAG_SIZE = 315

GENERATED_X = 1150
GENERATED_Y = RECTIFIED_Y

DATA_X = 1530
DATA_Y = RECTIFIED_Y
DATA_W = 338
DATA_H = TAG_SIZE

SAMPLE_INDEX = 2
LOCATION_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


def rgb(value):
    """Convert a six-digit hex colour to a Cairo RGB tuple."""
    value = value.removeprefix("#")
    return tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))


PAPER = rgb("F4F7FB")
WHITE = rgb("FFFFFF")
BLACK = rgb("000000")
INK = rgb("101828")
RED = rgb("E5484D")


def set_source(ctx, colour, alpha=1.0):
    ctx.set_source_rgba(*colour, alpha)


def rounded_rect(ctx, x, y, width, height, radius):
    radius = min(radius, width / 2, height / 2)
    ctx.new_sub_path()
    ctx.arc(x + width - radius, y + radius, radius, -math.pi / 2, 0)
    ctx.arc(x + width - radius, y + height - radius, radius, 0, math.pi / 2)
    ctx.arc(x + radius, y + height - radius, radius, math.pi / 2, math.pi)
    ctx.arc(x + radius, y + radius, radius, math.pi, math.pi * 1.5)
    ctx.close_path()


def fill_round_rect(ctx, x, y, width, height, radius, colour, alpha=1.0):
    rounded_rect(ctx, x, y, width, height, radius)
    set_source(ctx, colour, alpha)
    ctx.fill()


def stroke_round_rect(
    ctx, x, y, width, height, radius, colour, line_width=1.0, alpha=1.0
):
    rounded_rect(ctx, x, y, width, height, radius)
    set_source(ctx, colour, alpha)
    ctx.set_line_width(line_width)
    ctx.stroke()


def regular_font(ctx, size):
    set_font_from_file(ctx, DEFAULT_FONT, size)


def mono_font(ctx, size, weight=cairo.FONT_WEIGHT_NORMAL):
    ctx.select_font_face("DejaVu Sans Mono", cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size)


def show_text(ctx, value, x, baseline, size, colour=INK, *, mono=False, bold=False):
    if mono:
        weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
        mono_font(ctx, size, weight)
    else:
        regular_font(ctx, size)
    set_source(ctx, colour)
    ctx.move_to(x, baseline)
    ctx.show_text(value)


def pil_surface(image):
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    output.seek(0)
    return cairo.ImageSurface.create_from_png(output)


def load_sample():
    manifest = json.loads((PROJECT_ROOT / "dataset" / "evaluation.json").read_text())
    case = manifest[SAMPLE_INDEX]
    tag_data = case["tags"][0]
    corners = np.asarray(
        [tag_data["location"][key] for key in LOCATION_KEYS], dtype=float
    )

    image_path = PROJECT_ROOT / "dataset" / case["image"]
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        scene = source.crop(PHOTO_CROP).resize(
            (PHOTO_SIZE, PHOTO_SIZE), Image.Resampling.LANCZOS
        )
        coefficients = _perspective_coefficients(corners)
        rectified = source.transform(
            (TAG_SIZE, TAG_SIZE),
            Image.Transform.PERSPECTIVE,
            coefficients,
            Image.Resampling.BICUBIC,
        )

    message_hex = tag_data["message"]
    message_bits = f"{int(message_hex, 16):024b}"
    tag = Tag.from_message(message_bits, n=5)
    generated = tag.to_image(cell_size=TAG_SIZE // 7, quiet_pad_size=0)

    assert generated.size == (TAG_SIZE, TAG_SIZE)
    assert tag.crc_valid is True

    return {
        "scene": pil_surface(scene),
        "rectified": pil_surface(rectified),
        "generated": pil_surface(generated),
        "corners": corners,
        "message_hex": message_hex,
    }


def draw_image_card(
    ctx, surface, x, y, size, radius=18, border_colour=WHITE, border_width=3
):
    fill_round_rect(ctx, x, y + 9, size, size, radius, INK, 0.11)
    ctx.save()
    rounded_rect(ctx, x, y, size, size, radius)
    ctx.clip()
    ctx.set_source_surface(surface, x, y)
    ctx.paint()
    ctx.restore()
    stroke_round_rect(ctx, x, y, size, size, radius, border_colour, border_width)


def scene_point(point):
    crop_left, crop_top, crop_right, crop_bottom = PHOTO_CROP
    scale_x = PHOTO_SIZE / (crop_right - crop_left)
    scale_y = PHOTO_SIZE / (crop_bottom - crop_top)
    return (
        PHOTO_X + (point[0] - crop_left) * scale_x,
        PHOTO_Y + (point[1] - crop_top) * scale_y,
    )


def draw_projection(ctx, corners):
    source_points = [scene_point(point) for point in corners]
    target_points = (
        (RECTIFIED_X, RECTIFIED_Y),
        (RECTIFIED_X + TAG_SIZE, RECTIFIED_Y),
        (RECTIFIED_X + TAG_SIZE, RECTIFIED_Y + TAG_SIZE),
        (RECTIFIED_X, RECTIFIED_Y + TAG_SIZE),
    )

    # A dark under-stroke keeps the red projection guides legible over the photograph.
    for source, target in zip(source_points, target_points, strict=True):
        ctx.move_to(*source)
        ctx.line_to(*target)
        set_source(ctx, INK, 0.22)
        ctx.set_line_width(7)
        ctx.stroke()

        ctx.move_to(*source)
        ctx.line_to(*target)
        set_source(ctx, RED, 0.92)
        ctx.set_line_width(4)
        ctx.stroke()

    # Outline the reviewed source quadrilateral and mark its four ordered corners.
    ctx.move_to(*source_points[0])
    for point in source_points[1:]:
        ctx.line_to(*point)
    ctx.close_path()
    set_source(ctx, RED)
    ctx.set_line_width(5)
    ctx.stroke()

    for point in source_points:
        ctx.arc(*point, 8, 0, math.tau)
        set_source(ctx, WHITE)
        ctx.fill()
        ctx.arc(*point, 5, 0, math.tau)
        set_source(ctx, RED)
        ctx.fill()


def draw_target_corners(ctx):
    target_points = (
        (RECTIFIED_X, RECTIFIED_Y),
        (RECTIFIED_X + TAG_SIZE, RECTIFIED_Y),
        (RECTIFIED_X + TAG_SIZE, RECTIFIED_Y + TAG_SIZE),
        (RECTIFIED_X, RECTIFIED_Y + TAG_SIZE),
    )
    for point in target_points:
        ctx.arc(*point, 7, 0, math.tau)
        set_source(ctx, WHITE)
        ctx.fill()
        ctx.arc(*point, 4.5, 0, math.tau)
        set_source(ctx, RED)
        ctx.fill()


def draw_arrow(ctx, start, end, y):
    ctx.move_to(start, y)
    ctx.line_to(end, y)
    set_source(ctx, INK, 0.72)
    ctx.set_line_width(4)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()

    ctx.move_to(end - 11, y - 8)
    ctx.line_to(end, y)
    ctx.line_to(end - 11, y + 8)
    set_source(ctx, INK, 0.72)
    ctx.set_line_width(4)
    ctx.stroke()


def draw_data_panel(ctx, sample):
    show_text(
        ctx,
        sample["message_hex"],
        DATA_X + 30,
        DATA_Y + 61,
        34,
        BLACK,
        mono=True,
        bold=True,
    )

    divider_y = DATA_Y + 86
    ctx.move_to(DATA_X + 28, divider_y)
    ctx.line_to(DATA_X + DATA_W - 28, divider_y)
    set_source(ctx, BLACK)
    ctx.set_line_width(3)
    ctx.stroke()

    service_lines = (
        "Tram Stop 124A.",
        "Next tram is route 109",
        "low floor tram to box hill",
        "in 1 minute.",
    )
    for index, line in enumerate(service_lines):
        show_text(ctx, line, DATA_X + 30, DATA_Y + 132 + index * 43, 23, BLACK)


def draw(surface_factory, width, height):
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError(f"Render at {WIDTH} × {HEIGHT} pixels")

    sample = load_sample()
    surface = surface_factory(width, height)
    ctx = cairo.Context(surface)

    set_source(ctx, PAPER)
    ctx.paint()

    # The projection is drawn between the source and destination cards so the
    # guides disappear naturally beneath the rectified raster.
    draw_image_card(ctx, sample["scene"], PHOTO_X, PHOTO_Y, PHOTO_SIZE)
    draw_projection(ctx, sample["corners"])
    draw_image_card(
        ctx,
        sample["rectified"],
        RECTIFIED_X,
        RECTIFIED_Y,
        TAG_SIZE,
        14,
        RED,
        5,
    )
    draw_target_corners(ctx)

    draw_image_card(ctx, sample["generated"], GENERATED_X, GENERATED_Y, TAG_SIZE, 14)
    center_y = RECTIFIED_Y + TAG_SIZE / 2
    draw_arrow(
        ctx,
        RECTIFIED_X + TAG_SIZE + 24,
        GENERATED_X - 24,
        center_y,
    )
    draw_arrow(
        ctx,
        GENERATED_X + TAG_SIZE + 18,
        DATA_X - 18,
        center_y,
    )
    draw_data_panel(ctx, sample)

    return surface, width, height
