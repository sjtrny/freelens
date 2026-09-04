"""Render the vector diagrams used by ``docs/ddtag-crc.md``.

The source follows the Cairo Visuals ``draw`` interface and uses the same
palette and typography as the other ddTag documentation diagrams.
"""

from __future__ import annotations

import sys
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
    fill_rect,
    mono_font,
    regular_font,
    rounded_rect,
    set_source,
    show_centered,
    show_text,
    stroke_rect,
    stroke_round_rect,
)
from freelens import (
    get_center_ind,
    get_corner_indices_1d,
    get_crc_inds,
    get_crc_input_inds,
)

MELBOURNE_FULL_TAG_BITS = "00101100010110110011111100001000001110001100110010"
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


def draw_legend_item(ctx, x, y, colour, label):
    fill_rect(ctx, x, y, 26, 26, colour)
    show_text(ctx, label, x + 38, y + 20, 17)


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
    draw_legend_item(ctx, 230, legend_y, BLUE, "message")
    draw_legend_item(ctx, 408, legend_y, ORANGE, "CRC")
    draw_palette_swatch(ctx, 548, legend_y, 26)
    show_text(ctx, "palette", 586, legend_y + 20, 17)
    draw_legend_item(ctx, 724, legend_y, PALE, "size")


def draw_patent_order(ctx, width, height):
    n = 5
    grid_size = 260
    grid_x = 34
    grid_y = 32
    crc_indices = patent_crc_indices(n)
    order = {cell_index: number for number, cell_index in enumerate(crc_indices, 1)}
    label_colours = {cell_index: INK for cell_index in crc_indices}

    draw_grid(
        ctx,
        n,
        grid_x,
        grid_y,
        grid_size,
        patent_grid_colours(n),
        order,
        label_colours,
    )
    draw_arrow(ctx, 320, height / 2, 370, height / 2, MUTED, 2.5)

    chip = 48
    gap = 8
    chip_x = 394
    chip_y = 131
    show_centered(ctx, "cell indices", chip_x, 74, 8 * chip + 7 * gap, 26, 19)
    for ordinal, cell_index in enumerate(crc_indices, 1):
        x = chip_x + (ordinal - 1) * (chip + gap)
        show_centered(
            ctx,
            str(ordinal),
            x,
            chip_y - 31,
            chip,
            22,
            13,
            MUTED,
            mono=True,
        )
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


def actual_storage_colours():
    storage_indices = set(get_crc_inds(5))
    return [
        CELL_COLOURS[MELBOURNE_CELLS[index]] if index in storage_indices else PALE
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

    show_centered(ctx, "input order", grid_x, 24, grid_size, 26, 19)
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
    order = {cell_index: number for number, cell_index in enumerate(storage_indices, 1)}
    grid_x = 40
    grid_y = 48
    grid_size = 300

    show_centered(ctx, "storage order", grid_x, 14, grid_size, 25, 19)
    draw_grid(
        ctx,
        5,
        grid_x,
        grid_y,
        grid_size,
        actual_storage_colours(),
        order,
        actual_label_colours(storage_indices),
    )
    draw_arrow(ctx, 366, 170, 416, 170, MUTED, 2.5)

    chip = 50
    gap = 8
    chip_x = 442
    chip_y = 102
    ribbon_width = 8 * chip + 7 * gap

    for ordinal, cell_index in enumerate(storage_indices, 1):
        x = chip_x + (ordinal - 1) * (chip + gap)
        show_centered(
            ctx,
            str(ordinal),
            x,
            chip_y - 30,
            chip,
            22,
            13,
            MUTED,
            mono=True,
        )
        draw_bit_chip(ctx, MELBOURNE_CELLS[cell_index], x, chip_y, chip)
        show_centered(
            ctx,
            str(cell_index),
            x,
            chip_y + 57,
            chip,
            20,
            13,
            MUTED,
            mono=True,
        )

    crc_bits = "".join(MELBOURNE_CELLS[index] for index in storage_indices)
    grouped_crc = f"{crc_bits[:8]} {crc_bits[8:]}"
    show_centered(
        ctx,
        grouped_crc,
        chip_x,
        218,
        ribbon_width,
        28,
        18,
        INK,
        mono=True,
    )
    show_centered_text(
        ctx,
        f"{int(crc_bits, 2):04X}",
        chip_x,
        264,
        ribbon_width,
        38,
        28,
        INK,
        mono=True,
        bold=True,
    )


def draw_crc_check(ctx, width, height):
    card_y = 44
    card_height = 150
    input_x = 30
    input_width = 224
    engine_x = 326
    engine_width = 268
    calculated_x = 666
    value_width = 170
    stored_x = 900

    draw_card(ctx, input_x, card_y, input_width, card_height)
    show_centered(ctx, "input bytes", input_x, card_y + 20, input_width, 25, 18)
    byte_values = ("13", "A0", "08", "72")
    byte_size = 38
    byte_gap = 8
    bytes_width = 4 * byte_size + 3 * byte_gap
    bytes_x = input_x + (input_width - bytes_width) / 2
    for column, value in enumerate(byte_values):
        x = bytes_x + column * (byte_size + byte_gap)
        fill_rect(ctx, x, card_y + 76, byte_size, byte_size, BLUE)
        show_centered_text(
            ctx,
            value,
            x,
            card_y + 76,
            byte_size,
            byte_size,
            14,
            WHITE,
            mono=True,
            bold=True,
        )

    draw_arrow(
        ctx,
        input_x + input_width + 16,
        card_y + card_height / 2,
        engine_x - 16,
        card_y + card_height / 2,
        MUTED,
        2.5,
    )

    draw_card(ctx, engine_x, card_y, engine_width, card_height)
    show_centered(ctx, "CRC-16", engine_x, card_y + 18, engine_width, 27, 20)
    labels = ("poly", "init", "xorout")
    values = ("0xC867", "0x0000", "0x0000")
    row_y = card_y + 68
    for row, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = row_y + row * 24
        show_text(ctx, label, engine_x + 67, y, 15, MUTED, mono=True)
        show_text(ctx, value, engine_x + 143, y, 15, INK, mono=True)

    draw_arrow(
        ctx,
        engine_x + engine_width + 16,
        card_y + card_height / 2,
        calculated_x - 16,
        card_y + card_height / 2,
        MUTED,
        2.5,
    )

    draw_card(ctx, calculated_x, card_y, value_width, card_height)
    show_centered(
        ctx,
        "calculated",
        calculated_x,
        card_y + 28,
        value_width,
        25,
        17,
        MUTED,
    )
    show_centered_text(
        ctx,
        "FFF2",
        calculated_x,
        card_y + 74,
        value_width,
        42,
        28,
        INK,
        mono=True,
        bold=True,
    )

    show_centered(ctx, "=", 850, card_y, 36, card_height, 28, INK)

    draw_card(ctx, stored_x, card_y, value_width, card_height)
    show_centered(
        ctx,
        "stored",
        stored_x,
        card_y + 28,
        value_width,
        25,
        17,
        MUTED,
    )
    show_centered_text(
        ctx,
        "FFF2",
        stored_x,
        card_y + 74,
        value_width,
        42,
        28,
        INK,
        mono=True,
        bold=True,
    )


DIAGRAMS = {
    "patent-layout": (1120, 384, draw_patent_layout),
    "patent-crc-order": (890, 326, draw_patent_order),
    "observed-crc-input": (1100, 436, draw_observed_input),
    "observed-crc-storage": (1050, 390, draw_observed_storage),
    "crc-check": (1100, 238, draw_crc_check),
}
PREVIEW_DIAGRAM = "patent-layout"


def render_diagram(surface_factory, name):
    width, height, drawer = DIAGRAMS[name]
    surface = surface_factory(width, height)
    ctx = cairo.Context(surface)
    draw_background(ctx, width, height)
    drawer(ctx, width, height)
    return surface, width, height


def draw(surface_factory, width, height):
    expected_width, expected_height, _ = DIAGRAMS[PREVIEW_DIAGRAM]
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"Render {PREVIEW_DIAGRAM!r} at {expected_width} x {expected_height} pixels"
        )
    return render_diagram(surface_factory, PREVIEW_DIAGRAM)


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
