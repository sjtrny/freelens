"""Render the vector diagrams used by ``docs/ddtag-crc.md``.

The source follows the Cairo Visuals ``draw`` interface and uses the same
palette and typography as the other ddTag documentation diagrams.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import cairo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "ddtag-crc"
sys.path.insert(0, str(PROJECT_ROOT))

from docs.assets.ddtag_diagrams import (
    BLUE,
    CELL_COLOURS,
    INK,
    LINE,
    MUTED,
    ORANGE,
    PALE,
    WHITE,
    draw_arrow,
    draw_background,
    draw_message_order,
    fill_rect,
    layout_content,
    mono_font,
    regular_font,
    render_layout,
    rounded_rect,
    set_source,
    show_centered,
    show_text,
    stroke_rect,
    stroke_round_rect,
)
from freelens import (
    Tag,
    get_center_ind,
    get_corner_indices_1d,
    get_crc_inds,
    get_crc_input_inds,
)

MELBOURNE_FULL_TAG_BITS = "00101100010110110011111100001000001110001100110010"
MELBOURNE_TAG = Tag(MELBOURNE_FULL_TAG_BITS, n=5)
MELBOURNE_CELLS = tuple(
    MELBOURNE_FULL_TAG_BITS[offset : offset + 2]
    for offset in range(0, len(MELBOURNE_FULL_TAG_BITS), 2)
)

CORNER_BITS = ("00", "01", "10", "11")


def fill_round_rect(ctx, x, y, width, height, radius, colour):
    rounded_rect(ctx, x, y, width, height, radius)
    set_source(ctx, colour)
    ctx.fill()


def show_centered_text(
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


def draw_card(ctx, x, y, width, height, radius=14):
    fill_round_rect(ctx, x, y, width, height, radius, WHITE)
    stroke_round_rect(ctx, x, y, width, height, radius, LINE, 1.5)


def patent_crc_indices(n):
    center = n // 2
    return tuple(
        index
        for index in range(n * n)
        if (
            (index // n == center or index % n == center) and index != get_center_ind(n)
        )
    )


def draw_grid(ctx, n, x, y, size, colours, labels=None, label_colours=None):
    labels = labels or {}
    label_colours = label_colours or {}
    cell = size / n
    gap = max(1.25, size / 220)

    for index in range(n * n):
        row, column = divmod(index, n)
        cell_x = x + column * cell
        cell_y = y + row * cell
        fill_rect(ctx, cell_x, cell_y, cell - gap, cell - gap, colours[index])
        if index in labels:
            show_centered_text(
                ctx,
                str(labels[index]),
                cell_x,
                cell_y,
                cell - gap,
                cell - gap,
                max(10, min(20, cell * 0.34)),
                label_colours.get(index, INK),
                mono=True,
                bold=True,
            )

    stroke_rect(ctx, x - 2, y - 2, size + 2, size + 2, INK, 2)


def patent_grid_colours(n):
    corners = dict(zip(get_corner_indices_1d(n), CORNER_BITS, strict=True))
    crc_indices = set(patent_crc_indices(n))
    center_index = get_center_ind(n)
    colours = []

    for index in range(n * n):
        if index in corners:
            colours.append(CELL_COLOURS[corners[index]])
        elif index in crc_indices:
            colours.append(ORANGE)
        elif index == center_index:
            colours.append(PALE)
        else:
            colours.append(BLUE)

    return colours


def draw_palette_swatch(ctx, x, y, size):
    half = size / 2
    for column, bits in enumerate(CORNER_BITS[:2]):
        fill_rect(ctx, x + column * half, y, half, half, CELL_COLOURS[bits])
    for column, bits in enumerate(reversed(CORNER_BITS[2:])):
        fill_rect(
            ctx,
            x + column * half,
            y + half,
            half,
            half,
            CELL_COLOURS[bits],
        )


def draw_patent_layout(ctx, width, height):
    grid_size = 176
    grid_y = 58
    panel_width = 230
    panel_xs = (30, 306, 582, 858)

    for n, panel_x in zip((5, 7, 9, 11), panel_xs, strict=True):
        grid_x = panel_x + (panel_width - grid_size) / 2
        show_centered(ctx, f"{n}×{n}", panel_x, 18, panel_width, 27, 20)
        draw_grid(ctx, n, grid_x, grid_y, grid_size, patent_grid_colours(n))
        message_bits = 2 * n * n - 4 * n - 6
        crc_bits = 4 * n - 4
        show_centered(
            ctx,
            f"{message_bits}-bit message",
            panel_x,
            248,
            panel_width,
            24,
            16,
        )
        show_centered(
            ctx,
            f"{crc_bits}-bit CRC",
            panel_x,
            275,
            panel_width,
            22,
            15,
            MUTED,
        )

    legend_y = 326
    swatch_size = 26
    label_gap = 12
    item_gap = 36
    legend_items = (
        (BLUE, "message cells"),
        (ORANGE, "CRC cells"),
        (None, "palette cells"),
        (PALE, "size cell"),
    )
    regular_font(ctx, 17)
    label_extents = [ctx.text_extents(label) for _, label in legend_items]
    legend_width = sum(
        swatch_size + label_gap + extents.width for extents in label_extents
    ) + item_gap * (len(legend_items) - 1)
    legend_x = (width - legend_width) / 2

    for (colour, label), extents in zip(legend_items, label_extents, strict=True):
        if colour is None:
            draw_palette_swatch(ctx, legend_x, legend_y, swatch_size)
        else:
            fill_rect(ctx, legend_x, legend_y, swatch_size, swatch_size, colour)
        show_text(
            ctx,
            label,
            legend_x + swatch_size + label_gap - extents.x_bearing,
            legend_y + 20,
            17,
        )
        legend_x += swatch_size + label_gap + extents.width + item_gap


def draw_patent_order(ctx, width, height):
    n = 5
    grid_size = 260
    grid_x = 34
    grid_y = 32
    crc_indices = patent_crc_indices(n)
    colours = patent_grid_colours(n)
    labels = {cell_index: cell_index for cell_index in range(n * n)}
    label_colours = {}
    for cell_index, background in enumerate(colours):
        if cell_index in crc_indices:
            label_colours[cell_index] = INK
        else:
            foreground = WHITE if background == CELL_COLOURS["11"] else INK
            label_colours[cell_index] = tuple(
                (text_channel + background_channel) / 2
                for text_channel, background_channel in zip(
                    foreground, background, strict=True
                )
            )

    draw_grid(
        ctx,
        n,
        grid_x,
        grid_y,
        grid_size,
        colours,
        labels,
        label_colours,
    )
    draw_arrow(ctx, 320, height / 2, 370, height / 2, MUTED, 2.5)

    chip = 48
    gap = 8
    chip_x = 394
    chip_y = (height - chip) / 2
    show_centered(
        ctx,
        "zero-based cell indices",
        chip_x,
        chip_y - 40,
        len(crc_indices) * chip + (len(crc_indices) - 1) * gap,
        26,
        19,
    )
    for position, cell_index in enumerate(crc_indices):
        x = chip_x + position * (chip + gap)
        fill_rect(ctx, x, chip_y, chip, chip, ORANGE)
        show_centered_text(
            ctx,
            str(cell_index),
            x,
            chip_y,
            chip,
            chip,
            16,
            INK,
            mono=True,
            bold=True,
        )


def actual_input_colours():
    input_indices = set(get_crc_input_inds(5))
    return [
        CELL_COLOURS[MELBOURNE_CELLS[index]] if index in input_indices else PALE
        for index in range(25)
    ]


def actual_label_colours(indices):
    return {
        index: WHITE if MELBOURNE_CELLS[index] == "11" else INK for index in indices
    }


def draw_bit_chip(ctx, bits, x, y, size):
    fill_rect(ctx, x, y, size, size, CELL_COLOURS[bits])
    text_colour = WHITE if bits == "11" else INK
    show_centered_text(
        ctx,
        bits,
        x,
        y,
        size,
        size,
        14,
        text_colour,
        mono=True,
        bold=True,
    )


def draw_observed_input(ctx, width, height):
    input_indices = tuple(get_crc_input_inds(5))
    order = {cell_index: number for number, cell_index in enumerate(input_indices, 1)}
    grid_x = 40
    grid_y = 66
    grid_size = 320

    show_centered(ctx, "read positions (1–16)", grid_x, 24, grid_size, 26, 19)
    draw_grid(
        ctx,
        5,
        grid_x,
        grid_y,
        grid_size,
        actual_input_colours(),
        order,
        actual_label_colours(input_indices),
    )

    rows_x = 476
    row_width = 520
    show_centered(ctx, "four bytes", rows_x, 24, row_width, 26, 19)
    chip = 50
    chip_gap = 8
    chip_x = rows_x + 54
    row_ys = (66, 150, 234, 318)

    for row, row_y in enumerate(row_ys):
        first = row * 4
        last = first + 4
        indices = input_indices[first:last]
        bits = [MELBOURNE_CELLS[index] for index in indices]
        show_centered(
            ctx,
            f"{first + 1}–{last}",
            rows_x,
            row_y,
            46,
            chip,
            14,
            MUTED,
            mono=True,
        )
        for column, cell_bits in enumerate(bits):
            draw_bit_chip(
                ctx,
                cell_bits,
                chip_x + column * (chip + chip_gap),
                row_y,
                chip,
            )

        arrow_start = chip_x + 4 * chip + 3 * chip_gap + 18
        draw_arrow(
            ctx,
            arrow_start,
            row_y + chip / 2,
            arrow_start + 44,
            row_y + chip / 2,
            MUTED,
            2.5,
        )
        byte_value = int("".join(bits), 2)
        byte_x = arrow_start + 68
        draw_card(ctx, byte_x, row_y, 86, chip, 10)
        show_centered_text(
            ctx,
            f"{byte_value:02X}",
            byte_x,
            row_y,
            86,
            chip,
            20,
            INK,
            mono=True,
            bold=True,
        )


def draw_observed_storage(ctx, width, height):
    storage_indices = tuple(get_crc_inds(5))
    pairs = tuple(MELBOURNE_CELLS[index] for index in storage_indices)
    crc_bits = "".join(pairs)
    crc_value = int(crc_bits, 2)
    grouped_bits = " ".join(crc_bits[offset : offset + 4] for offset in range(0, 16, 4))
    grid_x = 32
    grid_y = 76
    grid_size = 270
    show_centered(ctx, "zero-based cell indices", grid_x, 32, grid_size, 26, 19)
    draw_grid(
        ctx,
        5,
        grid_x,
        grid_y,
        grid_size,
        [ORANGE if index in storage_indices else PALE for index in range(25)],
        {index: index for index in range(25)},
        {index: INK if index in storage_indices else MUTED for index in range(25)},
    )

    chip = 44
    cell_gap = 6
    arm_gap = 18
    group_width = 2 * chip + cell_gap
    ribbon_width = 4 * group_width + 3 * arm_gap
    ribbon_x = 458
    center_x = ribbon_x + ribbon_width / 2
    cells_y = 76
    bits_y = 138
    label_x = ribbon_x - 66
    arrow_y = (cells_y + chip + bits_y) / 2

    draw_arrow(
        ctx,
        grid_x + grid_size + 18,
        arrow_y,
        label_x - 16,
        arrow_y,
        MUTED,
        2.5,
    )
    show_centered(ctx, "cell", label_x, cells_y, 54, chip, 15, MUTED)
    show_centered(ctx, "bits", label_x, bits_y, 54, chip, 15, MUTED)

    for position, (cell_index, bits) in enumerate(
        zip(storage_indices, pairs, strict=True)
    ):
        arm, offset = divmod(position, 2)
        x = ribbon_x + arm * (group_width + arm_gap) + offset * (chip + cell_gap)
        fill_rect(ctx, x, bits_y, chip, chip, WHITE)
        stroke_rect(ctx, x, bits_y, chip, chip, LINE, 1.5)
        show_centered_text(
            ctx, bits, x, bits_y, chip, chip, 16, INK, mono=True, bold=True
        )
        fill_rect(ctx, x, cells_y, chip, chip, ORANGE)
        show_centered_text(
            ctx,
            str(cell_index),
            x,
            cells_y,
            chip,
            chip,
            16,
            INK,
            mono=True,
            bold=True,
        )

    for arm, name in enumerate(("left", "upper", "lower", "right")):
        show_centered(
            ctx,
            name,
            ribbon_x + arm * (group_width + arm_gap),
            32,
            group_width,
            26,
            16,
            MUTED,
        )

    draw_arrow(ctx, center_x, 196, center_x, 220, MUTED, 2)
    show_centered(
        ctx, grouped_bits, ribbon_x, 234, ribbon_width, 26, 18, INK, mono=True
    )
    draw_arrow(ctx, center_x, 274, center_x, 298, MUTED, 2)
    show_centered_text(
        ctx,
        f"CRC 0x{crc_value:04X}",
        ribbon_x,
        314,
        ribbon_width,
        32,
        25,
        INK,
        mono=True,
        bold=True,
    )


DIAGRAMS = {
    "patent-layout": (1120, 384, draw_patent_layout),
    "patent-crc-order": (890, 326, draw_patent_order),
    "message-order": (1200, 500, partial(draw_message_order, tag=MELBOURNE_TAG)),
    "observed-crc-input": (1100, 436, draw_observed_input),
    "observed-crc-storage": (920, 378, draw_observed_storage),
}
PREVIEW_DIAGRAM = "patent-layout"


def render_diagram(surface_factory, name):
    width, height, drawer = DIAGRAMS[name]
    return render_layout(surface_factory, layout_content(drawer, width, height))


def draw(surface_factory, width, height):
    layout_width, layout_height, drawer = DIAGRAMS[PREVIEW_DIAGRAM]
    layout = layout_content(drawer, layout_width, layout_height)
    expected_width, expected_height = layout[1:3]
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"Render {PREVIEW_DIAGRAM!r} at {expected_width} x {expected_height} pixels"
        )
    return render_layout(surface_factory, layout)


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
