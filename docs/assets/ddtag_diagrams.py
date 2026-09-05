"""Render the vector diagrams used by ``docs/ddtag.md``.

The source follows the Cairo Visuals ``draw`` interface and uses its bundled
Atkinson Hyperlegible font. Running the module directly renders every diagram.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cairo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "ddtag"
sys.path.insert(0, str(PROJECT_ROOT))

from freelens import (  # noqa: E402
    Tag,
    get_center_ind,
    get_corner_indices_1d,
    get_crc_inds,
    get_crc_input_inds,
    get_message_inds,
)

try:
    from cairo_font import DEFAULT_FONT, set_font_from_file
except ImportError as error:  # pragma: no cover - supplied by Cairo Visuals
    raise RuntimeError(
        "Render these diagrams with the cairo-visuals project"
    ) from error

MESSAGE_HEX = "4A005C"
MESSAGE_BITS = f"{int(MESSAGE_HEX, 16):024b}"
EXAMPLE_TAG = Tag.from_message(MESSAGE_BITS, n=5)
CONTENT_PADDING = 32
MIN_LABEL_SIZE = 18
# Match image-processing.svg: a 38-pixel line with a 10-pixel arrowhead.
MIN_ARROW_SHAFT_LENGTH = 28
MIN_ARROW_LENGTH = MIN_ARROW_SHAFT_LENGTH + 10


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
ORANGE = rgb("F79009")
ORANGE_PALE = rgb("FDE6C2")
RED = rgb("E5484D")
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


def stroke_round_rect(
    ctx, x, y, width, height, radius, colour, line_width=1.0, alpha=1.0
):
    rounded_rect(ctx, x, y, width, height, radius)
    set_source(ctx, colour, alpha)
    ctx.set_line_width(line_width)
    ctx.stroke()


def stroke_rect(ctx, x, y, width, height, colour, line_width=1.0, alpha=1.0):
    set_source(ctx, colour, alpha)
    ctx.set_line_width(line_width)
    ctx.rectangle(x, y, width, height)
    ctx.stroke()


def regular_font(ctx, size):
    set_font_from_file(ctx, DEFAULT_FONT, max(MIN_LABEL_SIZE, size))


def mono_font(ctx, size, bold=False):
    weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
    ctx.select_font_face("DejaVu Sans Mono", cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(max(MIN_LABEL_SIZE, size))


def show_text(ctx, value, x, baseline, size, colour=INK, *, mono=False, bold=False):
    if mono:
        mono_font(ctx, size, bold)
    else:
        regular_font(ctx, size)
    set_source(ctx, colour)
    ctx.move_to(x, baseline)
    ctx.show_text(value)


def show_centered(
    ctx, value, x, y, width, height, size, colour=INK, *, mono=False, bold=False
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


def draw_step_badge(ctx, step, x, y, *, width=28):
    """Draw a procedure step separately from cell indices and sample counts."""
    height = 28
    rounded_rect(ctx, x, y, width, height, height / 2)
    set_source(ctx, BLUE)
    ctx.fill()
    show_centered(ctx, str(step), x, y, width, height, 18, WHITE, mono=True, bold=True)


def draw_step_title(ctx, step, label, x, y, width, *, size=18):
    draw_step_badge(ctx, step, x, y)
    show_text(ctx, label, x + 38, y + 21, size)


def draw_background(ctx, width, height, *, border=True):
    if not border:
        set_source(ctx, PAPER)
        ctx.paint()
        return

    rounded_rect(ctx, 1.5, 1.5, width - 3, height - 3, 18)
    set_source(ctx, PAPER)
    ctx.fill_preserve()
    set_source(ctx, LINE)
    ctx.set_line_width(1.5)
    ctx.stroke()


def draw_arrow(ctx, start_x, start_y, end_x, end_y, colour=INK, line_width=3):
    angle = math.atan2(end_y - start_y, end_x - start_x)
    head = 11
    ctx.move_to(start_x, start_y)
    ctx.line_to(end_x, end_y)
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()

    for offset in (-0.68, 0.68):
        ctx.move_to(end_x, end_y)
        ctx.line_to(
            end_x - head * math.cos(angle + offset),
            end_y - head * math.sin(angle + offset),
        )
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.stroke()


def draw_polyline_arrow(ctx, points, colour=RED, line_width=3):
    ctx.move_to(*points[0])
    for point in points[1:]:
        ctx.line_to(*point)
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()
    draw_arrow(ctx, *points[-2], *points[-1], colour, line_width)


def draw_actual_grid(ctx, tag, x, y, cell, *, gap=1):
    for index, bits in enumerate(tag.cells):
        row, column = divmod(index, tag.n)
        cell_x = x + column * cell
        cell_y = y + row * cell
        set_source(ctx, CELL_COLOURS[bits])
        ctx.rectangle(cell_x, cell_y, cell - gap, cell - gap)
        ctx.fill()


def draw_full_tag(ctx, tag, x, y, cell, outer):
    inner = cell
    grid_size = tag.n * cell
    total = grid_size + 2 * inner + 2 * outer
    fill_rect(ctx, x + 5, y + 7, total, total, INK, 0.12)
    fill_rect(ctx, x, y, total, total, WHITE)
    stroke_rect(ctx, x, y, total, total, LINE, 2)
    inner_x = x + outer
    inner_y = y + outer
    set_source(ctx, CELL_COLOURS["11"])
    ctx.rectangle(inner_x, inner_y, grid_size + 2 * inner, grid_size + 2 * inner)
    ctx.fill()
    grid_x = inner_x + inner
    grid_y = inner_y + inner
    draw_actual_grid(ctx, tag, grid_x, grid_y, cell, gap=0)
    return {
        "total": total,
        "outer": outer,
        "inner": inner,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_size": grid_size,
        "inner_x": inner_x,
        "inner_y": inner_y,
    }


def draw_example(ctx, width, height):
    draw_full_tag(ctx, EXAMPLE_TAG, 32, 32, 24, 32)


def callout(ctx, start, elbow_x, target_y, title, detail, swatch=None):
    end_x = 690
    ctx.move_to(*start)
    ctx.line_to(elbow_x, target_y)
    ctx.line_to(end_x, target_y)
    set_source(ctx, RED)
    ctx.set_line_width(3)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()
    ctx.arc(*start, 6, 0, math.tau)
    set_source(ctx, RED)
    ctx.fill()

    text_x = 720
    if swatch is not None:
        fill_rect(ctx, text_x, target_y - 23, 42, 42, swatch)
        stroke_rect(ctx, text_x, target_y - 23, 42, 42, LINE, 1.5)
        text_x += 60
    show_text(ctx, title, text_x, target_y - 2, 25, INK)
    show_text(ctx, detail, text_x, target_y + 26, 17, MUTED)


def draw_quiet_zones(ctx, width, height):
    geometry = draw_full_tag(ctx, EXAMPLE_TAG, 70, 35, 40, 55)

    # Three nested borders make the two rings and the grid boundary explicit.
    stroke_rect(
        ctx,
        70,
        35,
        geometry["total"],
        geometry["total"],
        RED,
        3,
    )
    set_source(ctx, RED)
    ctx.set_line_width(3)
    ctx.rectangle(
        geometry["inner_x"],
        geometry["inner_y"],
        geometry["grid_size"] + 2 * geometry["inner"],
        geometry["grid_size"] + 2 * geometry["inner"],
    )
    ctx.stroke()
    ctx.rectangle(
        geometry["grid_x"],
        geometry["grid_y"],
        geometry["grid_size"],
        geometry["grid_size"],
    )
    ctx.stroke()

    callout(
        ctx,
        (70 + geometry["total"] - geometry["outer"] / 2, 76),
        570,
        104,
        "outer quiet zone",
        "solid colour; at least as wide as the inner zone",
        WHITE,
    )
    callout(
        ctx,
        (
            geometry["grid_x"] + geometry["grid_size"] + geometry["inner"] / 2,
            226,
        ),
        600,
        226,
        "inner quiet zone",
        "one cell wide; one of the four tag colours",
        CELL_COLOURS["11"],
    )
    callout(
        ctx,
        (
            geometry["grid_x"] + geometry["grid_size"] - 10,
            geometry["grid_y"] + geometry["grid_size"] - 20,
        ),
        630,
        348,
        "tag grid",
        "the coloured data cells",
        CELL_COLOURS["00"],
    )


def draw_corner_label(ctx, x, y, bits, label, target, from_right=False):
    swatch_x = x if from_right else x + 212
    fill_rect(ctx, swatch_x, y, 54, 54, CELL_COLOURS[bits])
    stroke_rect(ctx, swatch_x, y, 54, 54, LINE, 1.5)
    text_x = x + 68 if from_right else x
    show_text(ctx, label, text_x, y + 23, 19, INK, mono=True)
    start_x = swatch_x + (0 if from_right else 54)
    start_y = y + 27
    ctx.move_to(start_x, start_y)
    ctx.line_to(*target)
    set_source(ctx, RED)
    ctx.set_line_width(2.5)
    ctx.stroke()


def draw_corners(ctx, width, height):
    grid_x = 420
    grid_y = 60
    cell = 72
    corners = {
        0: "00",
        4: "01",
        24: "10",
        20: "11",
    }
    for index in range(25):
        row, column = divmod(index, 5)
        colour = CELL_COLOURS[corners[index]] if index in corners else PALE
        set_source(ctx, colour)
        ctx.rectangle(grid_x + column * cell, grid_y + row * cell, cell - 2, cell - 2)
        ctx.fill()
    stroke_rect(ctx, grid_x - 3, grid_y - 3, 366, 366, INK, 3)

    centres = {
        0: (grid_x + cell / 2, grid_y + cell / 2),
        4: (grid_x + 4.5 * cell, grid_y + cell / 2),
        24: (grid_x + 4.5 * cell, grid_y + 4.5 * cell),
        20: (grid_x + cell / 2, grid_y + 4.5 * cell),
    }
    draw_corner_label(ctx, 78, 76, "00", "00 / cyan", centres[0])
    draw_corner_label(ctx, 868, 76, "01", "01 / magenta", centres[4], True)
    draw_corner_label(ctx, 868, 338, "10", "10 / yellow", centres[24], True)
    draw_corner_label(ctx, 78, 338, "11", "11 / black", centres[20])


def legend_item(ctx, x, y, colour, title, detail):
    fill_rect(ctx, x, y, 48, 48, colour)
    show_text(ctx, title, x + 68, y + 21, 21, INK)
    show_text(ctx, detail, x + 68, y + 46, 16, MUTED)


def draw_crc(ctx, width, height):
    grid_x = 92
    grid_y = 70
    cell = 70
    crc_indices = set(get_crc_inds(5))
    input_indices = set(get_crc_input_inds(5))
    center_index = get_center_ind(5)

    for index in range(25):
        row, column = divmod(index, 5)
        if index in crc_indices:
            colour = ORANGE
        elif index == center_index:
            colour = PALE
        elif index in input_indices:
            colour = BLUE
        else:
            colour = WHITE
        cell_x = grid_x + column * cell
        cell_y = grid_y + row * cell
        set_source(ctx, colour)
        ctx.rectangle(cell_x, cell_y, cell - 2, cell - 2)
        ctx.fill()

    stroke_rect(ctx, grid_x - 3, grid_y - 3, 356, 356, INK, 3)
    legend_item(ctx, 570, 105, BLUE, "CRC input", "32 bits outside the central cross")
    legend_item(ctx, 570, 225, ORANGE, "stored CRC", "eight cells = 16 bits")
    legend_item(ctx, 570, 345, PALE, "size cell", "excluded from the CRC")


def draw_message_order(ctx, width, height, tag=EXAMPLE_TAG):
    grid_x = 70
    grid_y = 70
    cell = 70
    message_indices = get_message_inds(5)
    order = {cell_index: number for number, cell_index in enumerate(message_indices, 1)}

    show_centered(ctx, "read positions (1–12)", grid_x, 24, 350, 28, 20)

    for index in range(25):
        row, column = divmod(index, 5)
        cell_x = grid_x + column * cell
        cell_y = grid_y + row * cell
        bits = tag.cells[index]
        colour = CELL_COLOURS[bits] if index in order else PALE
        set_source(ctx, colour)
        ctx.rectangle(cell_x, cell_y, cell - 2, cell - 2)
        ctx.fill()
        if index in order:
            text_colour = WHITE if bits == "11" else INK
            show_centered(
                ctx,
                str(order[index]),
                cell_x,
                cell_y,
                cell - 2,
                cell - 2,
                22,
                text_colour,
                mono=True,
            )
    stroke_rect(ctx, grid_x - 3, grid_y - 3, 356, 356, INK, 3)

    draw_arrow(ctx, 455, 245, 520, 245, INK, 3)

    chunks = [tag.cells[index] for index in message_indices]
    chip_x = 535
    chip_y = 175
    chip = 48
    gap = 7
    for index, bits in enumerate(chunks):
        x = chip_x + index * (chip + gap)
        fill_rect(ctx, x, chip_y, chip, chip, CELL_COLOURS[bits])
        text_colour = WHITE if bits == "11" else INK
        show_centered(ctx, bits, x, chip_y, chip, chip, 22, text_colour, mono=True)
        show_centered(
            ctx, str(index + 1), x, chip_y + 54, chip, 24, 20, MUTED, mono=True
        )

    ribbon_width = len(chunks) * chip + (len(chunks) - 1) * gap
    ctx.move_to(chip_x, 273)
    ctx.line_to(chip_x + ribbon_width, 273)
    set_source(ctx, INK)
    ctx.set_line_width(2)
    ctx.stroke()
    ctx.move_to(chip_x, 265)
    ctx.line_to(chip_x, 281)
    ctx.move_to(chip_x + ribbon_width, 265)
    ctx.line_to(chip_x + ribbon_width, 281)
    ctx.stroke()
    show_centered(ctx, tag.message, chip_x, 293, ribbon_width, 34, 22, INK, mono=True)
    show_centered(
        ctx,
        f"24 bits · {int(tag.message, 2):06X}",
        chip_x,
        334,
        ribbon_width,
        28,
        20,
        MUTED,
    )


DIAGRAMS = {
    "example": (500, 500, draw_example),
    "quiet-zones": (1200, 480, draw_quiet_zones),
    "corners": (1200, 480, draw_corners),
    "crc": (1200, 500, draw_crc),
    "message-order": (1200, 500, draw_message_order),
}
PREVIEW_DIAGRAM = "example"
SQUARE_DIAGRAMS = {"example"}
BORDERLESS_DIAGRAMS = {"example"}


def layout_content(drawer, layout_width, layout_height, *, square=False):
    """Fit a diagram to its content with the shared margin on all four sides."""
    recording = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    drawer(cairo.Context(recording), layout_width, layout_height)
    content_x, content_y, content_width, content_height = recording.ink_extents()

    if square:
        canvas_width = canvas_height = math.ceil(
            max(content_width, content_height) + 2 * CONTENT_PADDING
        )
        offset_x = (canvas_width - content_width) / 2 - content_x
        offset_y = (canvas_height - content_height) / 2 - content_y
    else:
        canvas_width = math.ceil(content_width + 2 * CONTENT_PADDING)
        canvas_height = math.ceil(content_height + 2 * CONTENT_PADDING)
        offset_x = CONTENT_PADDING - content_x
        offset_y = CONTENT_PADDING - content_y

    return recording, canvas_width, canvas_height, offset_x, offset_y


def render_layout(surface_factory, layout, *, border=True):
    recording, width, height, offset_x, offset_y = layout
    surface = surface_factory(width, height)
    ctx = cairo.Context(surface)
    draw_background(ctx, width, height, border=border)
    ctx.set_source_surface(recording, offset_x, offset_y)
    ctx.paint()
    return surface, width, height


def layout_diagram(name):
    width, height, drawer = DIAGRAMS[name]
    return layout_content(drawer, width, height, square=name in SQUARE_DIAGRAMS)


def render_diagram(surface_factory, name, layout=None):
    if layout is None:
        layout = layout_diagram(name)
    return render_layout(
        surface_factory,
        layout,
        border=name not in BORDERLESS_DIAGRAMS,
    )


def draw(surface_factory, width, height):
    layout = layout_diagram(PREVIEW_DIAGRAM)
    expected_width, expected_height = layout[1:3]
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"Render {PREVIEW_DIAGRAM!r} at {expected_width} x {expected_height} pixels"
        )
    return render_diagram(surface_factory, PREVIEW_DIAGRAM, layout)


def render_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in DIAGRAMS:
        output_path = OUTPUT_DIR / f"{name}.svg"
        surface, width, height = render_diagram(
            lambda width, height: cairo.SVGSurface(str(output_path), width, height),
            name,
        )
        surface.finish()
        print(f"{output_path.relative_to(PROJECT_ROOT)} ({width} x {height})")


if __name__ == "__main__":
    render_all()
