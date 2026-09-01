"""Render the vector diagrams used by ``docs/navilens.md``.

The diagrams recreate concepts from the project's PyCon Australia presentation in
the visual style used by the rest of the FreeLens documentation. Run this module
with the Cairo Visuals project on ``PYTHONPATH`` to regenerate both SVG assets.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cairo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "navilens"
sys.path.insert(0, str(PROJECT_ROOT))

from freelens import Tag  # noqa: E402

try:
    from cairo_font import DEFAULT_FONT, set_font_from_file
except ImportError as error:  # pragma: no cover - supplied by Cairo Visuals
    raise RuntimeError(
        "Render these diagrams with the cairo-visuals project"
    ) from error

TAG_ID = 894_562
EXAMPLE_TAG = Tag.from_message(f"{TAG_ID:024b}", n=5)


def rgb(value):
    value = value.removeprefix("#")
    return tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))


PAPER = rgb("F4F7FB")
WHITE = rgb("FFFFFF")
INK = rgb("101828")
MUTED = rgb("667085")
LINE = rgb("CBD5E1")
PALE = rgb("E5EAF0")
BLUE = rgb("2F80ED")
BLUE_PALE = rgb("D9EAFE")
RED = rgb("E5484D")
YELLOW = rgb("E8FF1A")
YELLOW_PALE = rgb("FFF2BF")
PERSONAL = rgb("3448F0")
COMMERCIAL = rgb("EF2C91")
LEASE_LIGHT = rgb("F6A6D3")
LEASE_MID = rgb("CF5995")
LEASE_DARK = rgb("9B1F58")
SERVER = rgb("344054")
DATABASE = rgb("475467")
DATABASE_TOP = rgb("667085")
CELL_COLOURS = {
    "00": rgb("00FFFF"),
    "01": rgb("FF00FF"),
    "10": rgb("FFFF00"),
    "11": rgb("000000"),
}


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


def fill_rect(ctx, x, y, width, height, colour, alpha=1.0):
    set_source(ctx, colour, alpha)
    ctx.rectangle(x, y, width, height)
    ctx.fill()


def stroke_rect(ctx, x, y, width, height, colour, line_width=1.0):
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.rectangle(x, y, width, height)
    ctx.stroke()


def fill_round_rect(ctx, x, y, width, height, radius, colour, alpha=1.0):
    rounded_rect(ctx, x, y, width, height, radius)
    set_source(ctx, colour, alpha)
    ctx.fill()


def stroke_round_rect(ctx, x, y, width, height, radius, colour, line_width=1.0):
    rounded_rect(ctx, x, y, width, height, radius)
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.stroke()


def regular_font(ctx, size):
    set_font_from_file(ctx, DEFAULT_FONT, size)


def mono_font(ctx, size, bold=False):
    weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
    ctx.select_font_face("DejaVu Sans Mono", cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size)


def show_text(ctx, value, x, baseline, size, colour=INK, *, mono=False, bold=False):
    if mono:
        mono_font(ctx, size, bold)
    else:
        regular_font(ctx, size)
    set_source(ctx, colour)
    ctx.move_to(x, baseline)
    ctx.show_text(value)


def show_centered(
    ctx,
    value,
    x,
    y,
    width,
    height,
    size,
    colour=INK,
    *,
    mono=False,
    bold=False,
):
    if mono:
        mono_font(ctx, size, bold)
    else:
        regular_font(ctx, size)
    extents = ctx.text_extents(value)
    text_x = x + (width - extents.width) / 2 - extents.x_bearing
    text_y = y + (height - extents.height) / 2 - extents.y_bearing
    set_source(ctx, colour)
    ctx.move_to(text_x, text_y)
    ctx.show_text(value)


def draw_background(ctx, width, height):
    set_source(ctx, PAPER)
    ctx.paint()
    stroke_round_rect(ctx, 1.5, 1.5, width - 3, height - 3, 18, LINE, 1.5)


def draw_arrow_head(ctx, x, y, angle, colour, size=10, line_width=3):
    for offset in (-0.7, 0.7):
        ctx.move_to(x, y)
        ctx.line_to(
            x - size * math.cos(angle + offset),
            y - size * math.sin(angle + offset),
        )
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.stroke()


def draw_arrow(ctx, start, end, colour=INK, *, both=False, line_width=3):
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    ctx.save()
    ctx.move_to(*start)
    ctx.line_to(*end)
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()
    draw_arrow_head(ctx, *end, angle, colour, line_width=line_width)
    if both:
        draw_arrow_head(ctx, *start, angle + math.pi, colour, line_width=line_width)
    ctx.restore()


def draw_tag(ctx, x, y, cell=11, inner=11, outer=16):
    grid_size = EXAMPLE_TAG.n * cell
    total = grid_size + 2 * inner + 2 * outer
    fill_rect(ctx, x, y, total, total, WHITE)
    stroke_rect(ctx, x, y, total, total, LINE, 2)
    inner_x = x + outer
    inner_y = y + outer
    fill_rect(
        ctx,
        inner_x,
        inner_y,
        grid_size + 2 * inner,
        grid_size + 2 * inner,
        CELL_COLOURS["11"],
    )
    grid_x = inner_x + inner
    grid_y = inner_y + inner
    for index, bits in enumerate(EXAMPLE_TAG.cells):
        row, column = divmod(index, EXAMPLE_TAG.n)
        fill_rect(
            ctx,
            grid_x + column * cell,
            grid_y + row * cell,
            cell,
            cell,
            CELL_COLOURS[bits],
        )
    return total


def draw_phone(ctx, x, y, width, height):
    fill_round_rect(ctx, x, y, width, height, 18, INK)
    fill_round_rect(ctx, x + 8, y + 23, width - 16, height - 53, 6, WHITE)
    fill_round_rect(ctx, x + width / 2 - 22, y + 8, 44, 5, 2.5, MUTED)

    screen_x = x + 16
    screen_y = y + 43
    screen_width = width - 32
    fill_round_rect(ctx, screen_x, screen_y, screen_width, 76, 7, BLUE_PALE)
    show_centered(
        ctx,
        "detected ID",
        screen_x,
        screen_y + 10,
        screen_width,
        22,
        14,
        MUTED,
    )
    show_centered(
        ctx,
        str(TAG_ID),
        screen_x,
        screen_y + 35,
        screen_width,
        31,
        18,
        INK,
        mono=True,
        bold=True,
    )

    center_x = x + width / 2
    wave_y = y + height - 76
    set_source(ctx, BLUE)
    ctx.set_line_width(3)
    for radius in (9, 17, 25):
        ctx.arc(center_x, wave_y, radius, math.pi * 1.15, math.pi * 1.85)
        ctx.stroke()
    ctx.arc(center_x, wave_y, 3.5, 0, math.tau)
    ctx.fill()

    fill_round_rect(ctx, center_x - 22, y + height - 16, 44, 4, 2, MUTED)


def cloud_path(ctx, x, y, width, height):
    ctx.move_to(x + 0.17 * width, y + 0.78 * height)
    ctx.curve_to(
        x + 0.02 * width,
        y + 0.78 * height,
        x,
        y + 0.56 * height,
        x + 0.12 * width,
        y + 0.47 * height,
    )
    ctx.curve_to(
        x + 0.13 * width,
        y + 0.27 * height,
        x + 0.31 * width,
        y + 0.19 * height,
        x + 0.44 * width,
        y + 0.29 * height,
    )
    ctx.curve_to(
        x + 0.55 * width,
        y + 0.04 * height,
        x + 0.82 * width,
        y + 0.13 * height,
        x + 0.84 * width,
        y + 0.39 * height,
    )
    ctx.curve_to(
        x + 1.02 * width,
        y + 0.44 * height,
        x + 1.01 * width,
        y + 0.72 * height,
        x + 0.86 * width,
        y + 0.78 * height,
    )
    ctx.close_path()


def draw_cloud(ctx, x, y, width, height):
    cloud_path(ctx, x, y, width, height)
    set_source(ctx, BLUE_PALE)
    ctx.fill_preserve()
    set_source(ctx, BLUE)
    ctx.set_line_width(2)
    ctx.stroke()
    show_centered(ctx, "Internet", x, y + 35, width, 42, 18, INK)


def ellipse_path(ctx, x, y, width, height):
    ctx.save()
    ctx.translate(x + width / 2, y + height / 2)
    ctx.scale(width / 2, height / 2)
    ctx.arc(0, 0, 1, 0, math.tau)
    ctx.restore()


def draw_database(ctx, x, y, width, height):
    ellipse_height = 34
    fill_rect(ctx, x, y + ellipse_height / 2, width, height - ellipse_height, DATABASE)
    ellipse_path(ctx, x, y + height - ellipse_height, width, ellipse_height)
    set_source(ctx, DATABASE)
    ctx.fill()
    ellipse_path(ctx, x, y, width, ellipse_height)
    set_source(ctx, DATABASE_TOP)
    ctx.fill_preserve()
    set_source(ctx, INK)
    ctx.set_line_width(2)
    ctx.stroke()
    ctx.move_to(x, y + ellipse_height / 2)
    ctx.line_to(x, y + height - ellipse_height / 2)
    ctx.move_to(x + width, y + ellipse_height / 2)
    ctx.line_to(x + width, y + height - ellipse_height / 2)
    ellipse_path(ctx, x, y + height - ellipse_height, width, ellipse_height)
    set_source(ctx, INK)
    ctx.set_line_width(2)
    ctx.stroke()
    show_centered(ctx, "Tag registry", x, y + 47, width, 48, 24, WHITE)
    show_centered(ctx, "ID to information", x, y + 84, width, 30, 15, WHITE)


def draw_registry_record(ctx, x, y, width, height):
    radius = 9
    header_height = 38
    id_width = 95
    owner_width = 120
    fill_round_rect(ctx, x, y, width, height, radius, WHITE)
    stroke_round_rect(ctx, x, y, width, height, radius, LINE, 1.5)

    ctx.save()
    rounded_rect(ctx, x, y, width, height, radius)
    ctx.clip()
    fill_rect(ctx, x, y, width, header_height, PALE)
    ctx.restore()

    for column_x in (x + id_width, x + id_width + owner_width):
        ctx.move_to(column_x, y)
        ctx.line_to(column_x, y + height)
    ctx.move_to(x, y + header_height)
    ctx.line_to(x + width, y + header_height)
    set_source(ctx, LINE)
    ctx.set_line_width(1.5)
    ctx.stroke()

    show_centered(ctx, "ID", x, y, id_width, header_height, 15, INK, bold=True)
    show_centered(
        ctx,
        "Owner",
        x + id_width,
        y,
        owner_width,
        header_height,
        15,
        INK,
        bold=True,
    )
    show_centered(
        ctx,
        "Information",
        x + id_width + owner_width,
        y,
        width - id_width - owner_width,
        header_height,
        15,
        INK,
        bold=True,
    )

    row_y = y + header_height
    row_height = height - header_height
    show_centered(ctx, str(TAG_ID), x, row_y, id_width, row_height, 15, INK, mono=True)
    show_centered(
        ctx, "Yarra Trams", x + id_width, row_y, owner_width, row_height, 15, INK
    )
    data_x = x + id_width + owner_width + 16
    show_text(ctx, "Route 86", data_x, row_y + 27, 15, INK)
    show_text(ctx, "Bundoora RMIT to Docklands", data_x, row_y + 53, 14, MUTED)


def draw_system_overview(ctx, width, height):
    draw_background(ctx, width, height)

    tag_x = 68
    tag_y = 36
    tag_size = 109
    phone_x = 65
    phone_y = 190
    phone_width = 120
    phone_height = 220

    ctx.move_to(phone_x + phone_width / 2, phone_y + 5)
    ctx.line_to(tag_x - 7, tag_y + tag_size + 8)
    ctx.line_to(tag_x + tag_size + 7, tag_y + tag_size + 8)
    ctx.close_path()
    set_source(ctx, YELLOW_PALE, 0.78)
    ctx.fill()

    draw_tag(ctx, tag_x, tag_y)
    show_centered(ctx, "ddTag", tag_x, tag_y + tag_size + 8, tag_size, 28, 17, INK)
    draw_phone(ctx, phone_x, phone_y, phone_width, phone_height)
    show_centered(ctx, "NaviLens app", 45, 420, 160, 34, 18, INK)

    draw_cloud(ctx, 260, 242, 150, 90)
    fill_round_rect(ctx, 485, 220, 215, 135, 12, SERVER)
    show_centered(ctx, "NaviLens", 485, 246, 215, 42, 24, WHITE)
    show_centered(ctx, "service", 485, 282, 215, 42, 24, WHITE)
    show_centered(ctx, "lookup API", 485, 320, 215, 22, 15, PALE)
    draw_database(ctx, 805, 210, 250, 150)

    draw_arrow(ctx, (198, 287), (246, 287), MUTED, both=True, line_width=3)
    draw_arrow(ctx, (424, 287), (471, 287), MUTED, both=True, line_width=3)
    draw_arrow(ctx, (714, 287), (791, 287), MUTED, both=True, line_width=3)

    record_x = 650
    record_y = 390
    record_width = 495
    record_height = 120
    draw_arrow(ctx, (930, 369), (930, 380), INK, line_width=3)
    draw_registry_record(ctx, record_x, record_y, record_width, record_height)


def draw_namespace(ctx, width, height):
    draw_background(ctx, width, height)

    start_x = 48
    gap = 12
    general_width = 190
    personal_width = 180
    commercial_width = 560
    personal_x = start_x + general_width + gap
    commercial_x = personal_x + personal_width + gap
    end_x = commercial_x + commercial_width

    main_y = 120
    main_height = 78
    fill_rect(ctx, start_x, main_y, general_width, main_height, YELLOW)
    fill_rect(ctx, personal_x, main_y, personal_width, main_height, PERSONAL)
    fill_rect(ctx, commercial_x, main_y, commercial_width, main_height, COMMERCIAL)
    show_centered(
        ctx, "General purpose", start_x, main_y, general_width, main_height, 22, INK
    )
    show_centered(
        ctx, "Personal use", personal_x, main_y, personal_width, main_height, 22, WHITE
    )
    show_centered(
        ctx,
        "Commercial",
        commercial_x,
        main_y,
        commercial_width,
        main_height,
        23,
        WHITE,
    )

    lease_y = 40
    lease_height = 56
    lease_width = 165
    lease_positions = (
        (commercial_x, LEASE_LIGHT, "Lease 1", INK),
        (commercial_x + (commercial_width - lease_width) / 2, LEASE_MID, "…", WHITE),
        (end_x - lease_width, LEASE_DARK, "Lease N", WHITE),
    )
    for lease_x, colour, label, text_colour in lease_positions:
        fill_rect(ctx, lease_x, lease_y, lease_width, lease_height, colour)
        show_centered(
            ctx,
            label,
            lease_x,
            lease_y,
            lease_width,
            lease_height,
            21,
            text_colour,
            mono=True,
        )
        center_x = lease_x + lease_width / 2
        ctx.move_to(center_x, lease_y + lease_height)
        ctx.line_to(center_x, main_y - 8)
        set_source(ctx, LINE)
        ctx.set_line_width(2)
        ctx.stroke()

    axis_y = 232
    ctx.move_to(start_x, axis_y)
    ctx.line_to(end_x, axis_y)
    set_source(ctx, INK)
    ctx.set_line_width(2)
    ctx.stroke()
    fill_rect(ctx, start_x - 4, axis_y - 6, 8, 12, INK)
    fill_rect(ctx, end_x - 4, axis_y - 6, 8, 12, INK)
    show_text(ctx, "0", start_x, 271, 20, INK, mono=True)

    end_label = "16,777,215"
    mono_font(ctx, 20)
    end_extents = ctx.text_extents(end_label)
    show_text(ctx, end_label, end_x - end_extents.width, 271, 20, INK, mono=True)
    show_centered(
        ctx,
        "16,777,216 possible IDs",
        start_x,
        250,
        end_x - start_x,
        30,
        15,
        MUTED,
    )


DIAGRAMS = {
    "system-overview": (1180, 545, draw_system_overview),
    "namespace": (1050, 305, draw_namespace),
}
PREVIEW_DIAGRAM = "system-overview"


def draw(surface_factory, width, height):
    expected_width, expected_height, drawer = DIAGRAMS[PREVIEW_DIAGRAM]
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"Render {PREVIEW_DIAGRAM!r} at {expected_width} x {expected_height} pixels"
        )
    surface = surface_factory(width, height)
    drawer(cairo.Context(surface), width, height)
    return surface, width, height


def render_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (width, height, drawer) in DIAGRAMS.items():
        output_path = OUTPUT_DIR / f"{name}.svg"
        surface = cairo.SVGSurface(str(output_path), width, height)
        drawer(cairo.Context(surface), width, height)
        surface.finish()
        print(output_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    render_all()
