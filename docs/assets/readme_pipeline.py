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
HEIGHT = 820

PHOTO_X = 52
PHOTO_Y = 215
PHOTO_SIZE = 535
PHOTO_CROP = (1000, 1200, 2200, 2400)

RECTIFIED_X = 760
RECTIFIED_Y = 285
TAG_SIZE = 315

GENERATED_X = 1175
GENERATED_Y = RECTIFIED_Y

DATA_X = 1530
DATA_Y = 215
DATA_W = 338
DATA_H = 535

SAMPLE_INDEX = 2
LOCATION_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


def rgb(value):
    """Convert a six-digit hex colour to a Cairo RGB tuple."""
    value = value.removeprefix("#")
    return tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))


PAPER = rgb("F4F7FB")
WHITE = rgb("FFFFFF")
INK = rgb("101828")
MUTED = rgb("667085")
HAIRLINE = rgb("D8E0EA")
CYAN = rgb("10B7C6")
MAGENTA = rgb("E94691")
YELLOW = rgb("E6B936")
SLATE = rgb("53657A")
PANEL = rgb("111A2C")
PANEL_MUTED = rgb("AAB8CC")
SUCCESS = rgb("58D6A9")
CORNER_COLOURS = (CYAN, MAGENTA, YELLOW, SLATE)


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


def show_right_text(ctx, value, right, baseline, size, colour=INK, *, mono=False):
    if mono:
        mono_font(ctx, size)
    else:
        regular_font(ctx, size)
    extents = ctx.text_extents(value)
    set_source(ctx, colour)
    ctx.move_to(right - extents.width - extents.x_bearing, baseline)
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
        "image_name": Path(case["image"]).name,
        "message_hex": message_hex,
        "message_bits": message_bits,
        "crc_hex": f"{int(tag.crc, 2):04X}",
    }


def draw_image_card(ctx, surface, x, y, size, radius=18):
    fill_round_rect(ctx, x, y + 9, size, size, radius, INK, 0.11)
    ctx.save()
    rounded_rect(ctx, x, y, size, size, radius)
    ctx.clip()
    ctx.set_source_surface(surface, x, y)
    ctx.paint()
    ctx.restore()
    stroke_round_rect(ctx, x, y, size, size, radius, WHITE, 3)


def draw_stage_heading(ctx, number, title, subtitle, x, y):
    fill_round_rect(ctx, x, y - 25, 42, 30, 15, INK)
    show_text(ctx, f"{number:02d}", x + 10, y - 4, 14, WHITE, mono=True, bold=True)
    show_text(ctx, title.upper(), x + 56, y - 3, 21, INK)
    show_text(ctx, subtitle, x + 56, y + 22, 16.5, MUTED, mono=True)


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

    # A dark under-stroke keeps the perspective guides legible over the photograph.
    for source, target, colour in zip(
        source_points, target_points, CORNER_COLOURS, strict=True
    ):
        ctx.move_to(*source)
        ctx.line_to(*target)
        set_source(ctx, INK, 0.22)
        ctx.set_line_width(7)
        ctx.stroke()

        ctx.move_to(*source)
        ctx.line_to(*target)
        set_source(ctx, colour, 0.86)
        ctx.set_line_width(3)
        ctx.stroke()

    # Outline the reviewed source quadrilateral and mark its four ordered corners.
    ctx.move_to(*source_points[0])
    for point in source_points[1:]:
        ctx.line_to(*point)
    ctx.close_path()
    set_source(ctx, WHITE, 0.94)
    ctx.set_line_width(3)
    ctx.stroke()

    for point, colour in zip(source_points, CORNER_COLOURS, strict=True):
        ctx.arc(*point, 8, 0, math.tau)
        set_source(ctx, WHITE)
        ctx.fill()
        ctx.arc(*point, 5, 0, math.tau)
        set_source(ctx, colour)
        ctx.fill()


def draw_target_corners(ctx):
    target_points = (
        (RECTIFIED_X, RECTIFIED_Y),
        (RECTIFIED_X + TAG_SIZE, RECTIFIED_Y),
        (RECTIFIED_X + TAG_SIZE, RECTIFIED_Y + TAG_SIZE),
        (RECTIFIED_X, RECTIFIED_Y + TAG_SIZE),
    )
    for point, colour in zip(target_points, CORNER_COLOURS, strict=True):
        ctx.arc(*point, 7, 0, math.tau)
        set_source(ctx, WHITE)
        ctx.fill()
        ctx.arc(*point, 4.5, 0, math.tau)
        set_source(ctx, colour)
        ctx.fill()


def draw_match_arrow(ctx):
    start = RECTIFIED_X + TAG_SIZE + 30
    end = GENERATED_X - 26
    y = GENERATED_Y + TAG_SIZE / 2

    ctx.move_to(start, y)
    ctx.line_to(end, y)
    set_source(ctx, CYAN)
    ctx.set_line_width(3)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()

    ctx.move_to(end - 12, y - 8)
    ctx.line_to(end, y)
    ctx.line_to(end - 12, y + 8)
    set_source(ctx, CYAN)
    ctx.set_line_width(3)
    ctx.stroke()

    fill_round_rect(ctx, start + 8, y - 49, end - start - 16, 27, 13.5, WHITE)
    stroke_round_rect(ctx, start + 8, y - 49, end - start - 16, 27, 13.5, HAIRLINE)
    show_text(ctx, "same ID", start + 20, y - 30, 13, MUTED)


def draw_code_panel(ctx, sample):
    fill_round_rect(ctx, DATA_X, DATA_Y + 9, DATA_W, DATA_H, 22, INK, 0.13)
    fill_round_rect(ctx, DATA_X, DATA_Y, DATA_W, DATA_H, 22, PANEL)

    show_text(ctx, "Tag.from_message()", DATA_X + 28, DATA_Y + 42, 19, CYAN, mono=True)
    show_text(
        ctx, "# regenerate", DATA_X + 28, DATA_Y + 78, 16.5, PANEL_MUTED, mono=True
    )
    code_lines = (
        "message = (",
        f'    "{sample["message_bits"][:8]}"',
        f'    "{sample["message_bits"][8:16]}"',
        f'    "{sample["message_bits"][16:]}"',
        ")",
        "tag = Tag.from_message(",
        "    message, n=5",
        ")",
        "tag.to_image()",
    )
    baseline = DATA_Y + 110
    for index, line in enumerate(code_lines):
        colour = SUCCESS if line == "tag.to_image()" else WHITE
        show_text(
            ctx, line, DATA_X + 28, baseline + index * 25, 16.5, colour, mono=True
        )

    divider_y = DATA_Y + 334
    ctx.move_to(DATA_X + 28, divider_y)
    ctx.line_to(DATA_X + DATA_W - 28, divider_y)
    set_source(ctx, WHITE, 0.14)
    ctx.set_line_width(1)
    ctx.stroke()

    fields = (
        ("TAG", sample["message_hex"]),
        ("MESSAGE", "24 bits"),
        ("GRID", "5 × 5"),
        ("CRC-16", sample["crc_hex"]),
    )
    for index, (label, value) in enumerate(fields):
        y = divider_y + 35 + index * 35
        show_text(ctx, label, DATA_X + 28, y, 14, PANEL_MUTED, mono=True)
        show_right_text(ctx, value, DATA_X + DATA_W - 28, y, 17, WHITE, mono=True)

    pill_x = DATA_X + 28
    pill_y = DATA_Y + DATA_H - 43
    fill_round_rect(ctx, pill_x, pill_y, DATA_W - 56, 32, 16, SUCCESS, 0.15)
    ctx.arc(pill_x + 19, pill_y + 16, 8, 0, math.tau)
    set_source(ctx, SUCCESS)
    ctx.fill()
    ctx.move_to(pill_x + 15, pill_y + 16)
    ctx.line_to(pill_x + 18, pill_y + 19)
    ctx.line_to(pill_x + 23, pill_y + 13)
    set_source(ctx, PANEL)
    ctx.set_line_width(2)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()
    show_text(
        ctx, "CRC VALID", pill_x + 36, pill_y + 22, 14, SUCCESS, mono=True, bold=True
    )


def draw(surface_factory, width, height):
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError(f"Render at {WIDTH} × {HEIGHT} pixels")

    sample = load_sample()
    surface = surface_factory(width, height)
    ctx = cairo.Context(surface)

    set_source(ctx, PAPER)
    ctx.paint()

    # Header
    fill_round_rect(ctx, 52, 47, 116, 35, 17.5, INK)
    show_text(ctx, "FREELENS", 68, 71, 15, WHITE, mono=True, bold=True)
    show_text(ctx, "one tag · field to data", 188, 74, 32, INK)
    show_right_text(
        ctx, "DETECT · RECTIFY · GENERATE · VERIFY", 1868, 70, 14, MUTED, mono=True
    )
    ctx.move_to(52, 105)
    ctx.line_to(1868, 105)
    set_source(ctx, HAIRLINE)
    ctx.set_line_width(1)
    ctx.stroke()

    # Stage labels
    draw_stage_heading(
        ctx, 1, "Field image", f"dataset / {sample['image_name']}", 52, 158
    )
    draw_stage_heading(ctx, 2, "Rectify", "4 corners → square", 760, 158)
    draw_stage_heading(ctx, 3, "Regenerate", "FreeLens / exact message", 1175, 158)
    draw_stage_heading(ctx, 4, "Read", "code + associated data", 1530, 158)

    # The projection is drawn between the source and destination cards so the
    # guides disappear naturally beneath the rectified raster.
    draw_image_card(ctx, sample["scene"], PHOTO_X, PHOTO_Y, PHOTO_SIZE)
    draw_projection(ctx, sample["corners"])
    draw_image_card(ctx, sample["rectified"], RECTIFIED_X, RECTIFIED_Y, TAG_SIZE, 14)
    draw_target_corners(ctx)

    draw_image_card(ctx, sample["generated"], GENERATED_X, GENERATED_Y, TAG_SIZE, 14)
    draw_match_arrow(ctx)
    draw_code_panel(ctx, sample)

    # Compact provenance and image captions.
    fill_round_rect(
        ctx, PHOTO_X + 18, PHOTO_Y + PHOTO_SIZE - 49, 219, 31, 15.5, INK, 0.80
    )
    show_text(
        ctx,
        "FIELD SAMPLE · CC BY 4.0",
        PHOTO_X + 32,
        PHOTO_Y + PHOTO_SIZE - 28,
        12.5,
        WHITE,
        mono=True,
    )
    show_text(ctx, "OBSERVED", RECTIFIED_X, 638, 13, CYAN, mono=True, bold=True)
    show_text(ctx, "perspective-normalized pixels", RECTIFIED_X, 664, 16, MUTED)
    show_text(ctx, "IDEAL", GENERATED_X, 638, 13, MAGENTA, mono=True, bold=True)
    show_text(ctx, "Tag.to_image() output", GENERATED_X, 664, 16, MUTED)

    return surface, width, height
