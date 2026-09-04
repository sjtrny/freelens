"""Render the decoding diagrams used by ``docs/ddtag-detection.md``.

The illustrations continue from the frame found by
``ddtag_detection_diagrams.py`` and expose the real intermediate values produced
while decoding the tag in ``dataset/images/0004.jpg``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cairo
import cv2 as cv
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "ddtag-detection"
sys.path.insert(0, str(PROJECT_ROOT))

from docs.assets.ddtag_detection_diagrams import (  # noqa: E402
    BLUE,
    CONTENT_PADDING,
    DETAIL_CROP,
    INK,
    LINE,
    MUTED,
    ORANGE,
    RED,
    WHITE,
    draw_arrow,
    draw_background,
    draw_image,
    draw_path,
    fill_round_rect,
    load_stages,
    map_point,
    rgb,
    set_source,
    show_centered,
    show_text,
    stroke_round_rect,
)
from freelens import (  # noqa: E402
    _correct_colours,
    _decode_sampled_cells,
    _rectify_frame,
    _sample_cells,
    compute_crc_5x5,
)

N = 5
CELL_SIZE = 32
QUIET_ZONE_CELLS = N + 4
FRAME_CELLS = N + 2
BIT_VALUES = ("00", "01", "10", "11")
CELL_COLOURS = (
    rgb("00FFFF"),
    rgb("FF00FF"),
    rgb("FFFF00"),
    rgb("000000"),
)
GREEN = rgb("12B76A")


@dataclass(frozen=True)
class DecodingStages:
    rectified: Image.Image
    frame: Image.Image
    black: np.ndarray
    white: np.ndarray
    cells_rgb: np.ndarray
    corrected_rgb: np.ndarray
    cells_lab: np.ndarray
    rotation: int
    rotated_rgb: np.ndarray
    rotated_lab: np.ndarray
    palette_lab: np.ndarray
    classes: np.ndarray
    tag: object
    computed_crc: int


@cache
def load_decoding_stages():
    detection = load_stages()
    image_rgba = cv.cvtColor(np.asarray(detection.image), cv.COLOR_RGB2RGBA)
    polygon = np.asarray(detection.frame_polygon, dtype=np.float32)

    quiet_zone_pixels = CELL_SIZE * QUIET_ZONE_CELLS
    frame_start = CELL_SIZE
    frame_stop = CELL_SIZE * (N + 3)
    reference_points = np.float32(
        [
            [frame_start, frame_start],
            [frame_stop, frame_start],
            [frame_stop, frame_stop],
            [frame_start, frame_stop],
        ]
    )
    transform = cv.getPerspectiveTransform(polygon, reference_points)
    rectified_rgba = cv.warpPerspective(
        image_rgba,
        transform,
        (quiet_zone_pixels, quiet_zone_pixels),
    )

    frame_rgb, black, white = _rectify_frame(
        image_rgba,
        polygon,
        N,
        CELL_SIZE,
    )
    cells_rgb = _sample_cells(frame_rgb, CELL_SIZE)
    corrected_rgb = np.round(_correct_colours(cells_rgb, black, white) * 255).astype(
        np.uint8
    )
    cells_lab = cv.cvtColor(corrected_rgb, cv.COLOR_RGB2Lab)

    observed_corners = np.array(
        [
            cells_lab[1, 1],
            cells_lab[1, -2],
            cells_lab[-2, -2],
            cells_lab[-2, 1],
        ]
    )
    darkest_corner = int(np.argmin(observed_corners[:, 0]))
    rotation = (darkest_corner + 1) % 4
    rotated_rgb = np.rot90(corrected_rgb, k=rotation)
    rotated_lab = np.rot90(cells_lab, k=rotation)
    palette_lab = np.array(
        [
            rotated_lab[1, 1],
            rotated_lab[1, -2],
            rotated_lab[-2, -2],
            rotated_lab[-2, 1],
        ]
    )
    grid_lab = rotated_lab[1 : N + 1, 1 : N + 1].astype(np.float64)
    distances = np.mean((grid_lab[:, :, None, :] - palette_lab) ** 2, axis=-1)
    classes = np.argmin(distances, axis=-1)
    tag = _decode_sampled_cells(cells_lab, N, validate_crc=True)

    return DecodingStages(
        rectified=Image.fromarray(rectified_rgba[..., :3], mode="RGB"),
        frame=Image.fromarray(frame_rgb, mode="RGB"),
        black=black,
        white=white,
        cells_rgb=cells_rgb,
        corrected_rgb=corrected_rgb,
        cells_lab=cells_lab,
        rotation=rotation,
        rotated_rgb=rotated_rgb,
        rotated_lab=rotated_lab,
        palette_lab=palette_lab,
        classes=classes,
        tag=tag,
        computed_crc=compute_crc_5x5(tag.cells),
    )


def stroke_rect(ctx, x, y, width, height, colour, line_width=1.0, alpha=1.0):
    set_source(ctx, colour, alpha)
    ctx.set_line_width(line_width)
    ctx.rectangle(x, y, width, height)
    ctx.stroke()


def fill_rect(ctx, x, y, width, height, colour, alpha=1.0):
    set_source(ctx, colour, alpha)
    ctx.rectangle(x, y, width, height)
    ctx.fill()


def colour_from_sample(sample):
    return tuple(float(channel) / 255 for channel in sample)


def draw_numbered_vertex(ctx, x, y, number):
    ctx.new_sub_path()
    ctx.arc(x, y, 12, 0, math.tau)
    set_source(ctx, RED)
    ctx.fill_preserve()
    set_source(ctx, WHITE)
    ctx.set_line_width(2)
    ctx.stroke()
    show_centered(ctx, str(number), x - 12, y - 12, 24, 24, 13, WHITE, mono=True)


def draw_projection_line(ctx, source, target):
    ctx.move_to(*source)
    ctx.line_to(*target)
    set_source(ctx, RED, 0.92)
    ctx.set_line_width(3.5)
    ctx.stroke()


def draw_rectification(ctx, width, height):
    detection = load_stages()
    decoding = load_decoding_stages()
    panel_size = 390
    panel_y = 58
    left_x = CONTENT_PADDING
    right_x = width - CONTENT_PADDING - panel_size
    detail = detection.image.crop(DETAIL_CROP)

    show_centered(ctx, "cyclic frame vertices", left_x, 20, panel_size, 27, 20)
    show_centered(ctx, "square rectification", right_x, 20, panel_size, 27, 20)

    draw_image(ctx, detail, left_x, panel_y, panel_size)

    source_points = [
        map_point(point, left_x, panel_y, panel_size, DETAIL_CROP)
        for point in detection.frame_polygon
    ]
    frame_inset = panel_size / QUIET_ZONE_CELLS
    target_points = (
        (right_x + frame_inset, panel_y + frame_inset),
        (right_x + panel_size - frame_inset, panel_y + frame_inset),
        (right_x + panel_size - frame_inset, panel_y + panel_size - frame_inset),
        (right_x + frame_inset, panel_y + panel_size - frame_inset),
    )

    for source, target in zip(source_points, target_points, strict=True):
        draw_projection_line(ctx, source, target)

    draw_image(ctx, decoding.rectified, right_x, panel_y, panel_size)
    draw_path(
        ctx,
        detection.frame_polygon,
        left_x,
        panel_y,
        panel_size,
        RED,
        4,
        crop=DETAIL_CROP,
        under_stroke=True,
    )
    stroke_rect(
        ctx,
        target_points[0][0],
        target_points[0][1],
        target_points[1][0] - target_points[0][0],
        target_points[3][1] - target_points[0][1],
        RED,
        4,
    )

    for index, (source, target) in enumerate(
        zip(source_points, target_points, strict=True),
        1,
    ):
        draw_numbered_vertex(ctx, *source, index)
        draw_numbered_vertex(ctx, *target, index)


def draw_ring_overlay(ctx, x, y, size, outer_cell, colour):
    cell = size / QUIET_ZONE_CELLS
    outer = outer_cell * cell
    inner = (outer_cell + 1) * cell
    fill_rect(ctx, x + outer, y + outer, size - 2 * outer, cell, colour, 0.24)
    fill_rect(
        ctx,
        x + outer,
        y + size - inner,
        size - 2 * outer,
        cell,
        colour,
        0.24,
    )
    fill_rect(ctx, x + outer, y + inner, cell, size - 2 * inner, colour, 0.24)
    fill_rect(
        ctx,
        x + size - inner,
        y + inner,
        cell,
        size - 2 * inner,
        colour,
        0.24,
    )
    stroke_rect(
        ctx,
        x + outer,
        y + outer,
        size - 2 * outer,
        size - 2 * outer,
        colour,
        3,
    )
    stroke_rect(
        ctx,
        x + inner,
        y + inner,
        size - 2 * inner,
        size - 2 * inner,
        colour,
        3,
    )


def draw_reference(ctx, x, y, colour, name, values, highlight):
    swatch = 52
    fill_rect(ctx, x, y, swatch, swatch, colour)
    stroke_rect(ctx, x, y, swatch, swatch, highlight, 3)
    show_text(ctx, name, x + 66, y + 21, 17)
    show_text(
        ctx,
        " ".join(str(round(value)) for value in values),
        x + 66,
        y + 47,
        15,
        MUTED,
        mono=True,
    )


def draw_quiet_zone_references(ctx, width, height):
    decoding = load_decoding_stages()
    panel_size = 360
    panel_y = 53
    left_x = CONTENT_PADDING
    card_x = 414
    card_width = 200
    right_x = width - CONTENT_PADDING - panel_size

    show_centered(ctx, "rectified quiet zones", left_x, 18, panel_size, 27, 20)
    show_centered(ctx, "black-framed tag", right_x, 18, panel_size, 27, 20)
    draw_image(ctx, decoding.rectified, left_x, panel_y, panel_size)
    draw_ring_overlay(ctx, left_x, panel_y, panel_size, 0, BLUE)
    draw_ring_overlay(ctx, left_x, panel_y, panel_size, 1, ORANGE)

    card_y = 105
    card_height = 256
    fill_round_rect(ctx, card_x, card_y, card_width, card_height, 14, WHITE)
    stroke_round_rect(ctx, card_x, card_y, card_width, card_height, 14, LINE, 1.5)
    show_centered(ctx, "median RGB", card_x, card_y + 14, card_width, 28, 18)
    draw_reference(
        ctx,
        card_x + 18,
        card_y + 65,
        colour_from_sample(decoding.white),
        "white",
        decoding.white,
        BLUE,
    )
    draw_reference(
        ctx,
        card_x + 18,
        card_y + 153,
        colour_from_sample(decoding.black),
        "black",
        decoding.black,
        ORANGE,
    )

    center_y = panel_y + panel_size / 2
    draw_arrow(ctx, left_x + panel_size + 9, card_x - 9, center_y)
    draw_arrow(ctx, card_x + card_width + 9, right_x - 9, center_y)
    draw_image(ctx, decoding.frame, right_x, panel_y, panel_size)


def draw_sample_grid(ctx, samples, x, y, size, *, labels=False):
    rows, columns = samples.shape[:2]
    cell = size / columns
    for row in range(rows):
        for column in range(columns):
            value = samples[row, column]
            colour = colour_from_sample(value)
            fill_rect(ctx, x + column * cell, y + row * cell, cell, cell, colour)
            if labels:
                palette_index = int(value)
                text_colour = WHITE if palette_index == 3 else INK
                show_centered(
                    ctx,
                    BIT_VALUES[palette_index],
                    x + column * cell,
                    y + row * cell,
                    cell,
                    cell,
                    13,
                    text_colour,
                    mono=True,
                )
    stroke_rect(ctx, x, y, size, size, LINE, 2)


def draw_class_grid(ctx, classes, x, y, size, *, labels=False):
    rows, columns = classes.shape
    cell = size / columns
    for row in range(rows):
        for column in range(columns):
            palette_index = int(classes[row, column])
            fill_rect(
                ctx,
                x + column * cell,
                y + row * cell,
                cell,
                cell,
                CELL_COLOURS[palette_index],
            )
            if labels:
                text_colour = WHITE if palette_index == 3 else INK
                show_centered(
                    ctx,
                    BIT_VALUES[palette_index],
                    x + column * cell,
                    y + row * cell,
                    cell,
                    cell,
                    13,
                    text_colour,
                    mono=True,
                )
    stroke_rect(ctx, x, y, size, size, INK, 2)


def draw_sample_centres(ctx, x, y, size):
    cell = size / FRAME_CELLS
    for index in range(1, FRAME_CELLS):
        position = index * cell
        ctx.move_to(x + position, y)
        ctx.line_to(x + position, y + size)
        ctx.move_to(x, y + position)
        ctx.line_to(x + size, y + position)
    set_source(ctx, WHITE, 0.42)
    ctx.set_line_width(1)
    ctx.stroke()

    for row in range(FRAME_CELLS):
        for column in range(FRAME_CELLS):
            point_x = x + (column + 0.5) * cell
            point_y = y + (row + 0.5) * cell
            ctx.arc(point_x, point_y, 4.5, 0, math.tau)
            set_source(ctx, WHITE)
            ctx.fill()
            ctx.arc(point_x, point_y, 2.6, 0, math.tau)
            set_source(ctx, RED)
            ctx.fill()


def draw_range_mapping(ctx, x, y, width, decoding):
    height = 236
    fill_round_rect(ctx, x, y, width, height, 14, WHITE)
    stroke_round_rect(ctx, x, y, width, height, 14, LINE, 1.5)
    show_centered(ctx, "map RGB range", x, y + 14, width, 30, 18)

    rows = (
        ("black", decoding.black, "0"),
        ("white", decoding.white, "255"),
    )
    for row_index, (label, values, target) in enumerate(rows):
        row_y = y + 62 + row_index * 69
        swatch_colour = colour_from_sample(values)
        fill_rect(ctx, x + 18, row_y, 38, 38, swatch_colour)
        stroke_rect(ctx, x + 18, row_y, 38, 38, LINE, 1.3)
        show_text(ctx, label, x + 67, row_y + 16, 15)
        show_text(
            ctx,
            " ".join(str(round(value)) for value in values),
            x + 67,
            row_y + 36,
            12,
            MUTED,
            mono=True,
        )
        draw_arrow(
            ctx,
            x + width - 78,
            x + width - 52,
            row_y + 20,
            line_width=1.8,
        )
        show_text(ctx, target, x + width - 44, row_y + 26, 14, INK, mono=True)

    show_centered(ctx, "RGB to CIELab", x, y + 197, width, 25, 16, MUTED)


def draw_colour_sampling(ctx, width, height):
    decoding = load_decoding_stages()
    panel_size = 340
    panel_y = 64
    left_x = CONTENT_PADDING
    card_x = 404
    card_width = 240
    right_x = width - CONTENT_PADDING - panel_size

    show_centered(ctx, "cell centres", left_x, 20, panel_size, 27, 20)
    show_centered(ctx, "49 colour samples", right_x, 20, panel_size, 27, 20)
    draw_image(ctx, decoding.frame, left_x, panel_y, panel_size)
    draw_sample_centres(ctx, left_x, panel_y, panel_size)
    draw_range_mapping(ctx, card_x, 115, card_width, decoding)
    draw_sample_grid(ctx, decoding.corrected_rgb, right_x, panel_y, panel_size)

    center_y = panel_y + panel_size / 2
    draw_arrow(ctx, left_x + panel_size + 9, card_x - 9, center_y)
    draw_arrow(ctx, card_x + card_width + 9, right_x - 9, center_y)


def draw_grid_highlight(ctx, x, y, size, row, column, colour, line_width=4):
    cell = size / N
    stroke_rect(
        ctx,
        x + column * cell + line_width / 2,
        y + row * cell + line_width / 2,
        cell - line_width,
        cell - line_width,
        colour,
        line_width,
    )


def draw_orientation_palette(ctx, width, height):
    decoding = load_decoding_stages()
    panel_size = 260
    panel_y = 62
    panel_xs = (32, 407, 782)
    observed = decoding.corrected_rgb[1 : N + 1, 1 : N + 1]
    oriented = decoding.rotated_rgb[1 : N + 1, 1 : N + 1]

    labels = ("darkest corner", "oriented palette", "nearest palette colour")
    for x, label in zip(panel_xs, labels, strict=True):
        show_centered(ctx, label, x, 19, panel_size, 27, 19)

    draw_sample_grid(ctx, observed, panel_xs[0], panel_y, panel_size)
    draw_grid_highlight(ctx, panel_xs[0], panel_y, panel_size, 4, 4, RED, 5)

    draw_sample_grid(ctx, oriented, panel_xs[1], panel_y, panel_size)
    for row, column in ((0, 0), (0, 4), (4, 4), (4, 0)):
        draw_grid_highlight(
            ctx,
            panel_xs[1],
            panel_y,
            panel_size,
            row,
            column,
            ORANGE,
            4,
        )

    draw_class_grid(
        ctx,
        decoding.classes,
        panel_xs[2],
        panel_y,
        panel_size,
        labels=True,
    )

    center_y = panel_y + panel_size / 2
    draw_arrow(ctx, panel_xs[0] + panel_size + 18, panel_xs[1] - 18, center_y)
    show_centered(
        ctx,
        "rotate 90°",
        panel_xs[0] + panel_size,
        center_y - 43,
        panel_xs[1] - panel_xs[0] - panel_size,
        24,
        16,
        MUTED,
    )
    draw_arrow(ctx, panel_xs[1] + panel_size + 18, panel_xs[2] - 18, center_y)


def draw_check(ctx, center_x, center_y, radius=15):
    ctx.new_sub_path()
    ctx.arc(center_x, center_y, radius, 0, math.tau)
    set_source(ctx, GREEN)
    ctx.fill()
    ctx.move_to(center_x - radius * 0.48, center_y)
    ctx.line_to(center_x - radius * 0.12, center_y + radius * 0.36)
    ctx.line_to(center_x + radius * 0.54, center_y - radius * 0.4)
    set_source(ctx, WHITE)
    ctx.set_line_width(3)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke()


def draw_validation_card(ctx, x, y, width, height, title):
    fill_round_rect(ctx, x, y, width, height, 14, WHITE)
    stroke_round_rect(ctx, x, y, width, height, 14, LINE, 1.5)
    show_text(ctx, title, x + 20, y + 30, 18)


def draw_validation(ctx, width, height):
    decoding = load_decoding_stages()
    grid_size = 300
    grid_x = CONTENT_PADDING
    grid_y = 66
    card_x = 388
    card_width = width - CONTENT_PADDING - card_x

    show_centered(ctx, "decoded cells", grid_x, 20, grid_size, 27, 20)
    draw_class_grid(
        ctx,
        decoding.classes,
        grid_x,
        grid_y,
        grid_size,
        labels=True,
    )
    draw_arrow(ctx, grid_x + grid_size + 14, card_x - 14, grid_y + grid_size / 2)

    corners_y = 32
    draw_validation_card(ctx, card_x, corners_y, card_width, 98, "corner order")
    swatch = 42
    swatch_x = card_x + 190
    for index, (bits, colour) in enumerate(zip(BIT_VALUES, CELL_COLOURS, strict=True)):
        x = swatch_x + index * 66
        fill_rect(ctx, x, corners_y + 27, swatch, swatch, colour)
        text_colour = WHITE if index == 3 else INK
        show_centered(
            ctx,
            bits,
            x,
            corners_y + 27,
            swatch,
            swatch,
            12,
            text_colour,
            mono=True,
        )
    draw_check(ctx, card_x + card_width - 30, corners_y + 49)

    message_y = 146
    draw_validation_card(ctx, card_x, message_y, card_width, 102, "message")
    grouped_message = " ".join(
        decoding.tag.message[index : index + 8]
        for index in range(0, len(decoding.tag.message), 8)
    )
    show_text(ctx, grouped_message, card_x + 128, message_y + 34, 16, MUTED, mono=True)
    show_text(
        ctx,
        f"{int(decoding.tag.message, 2):06X}",
        card_x + 128,
        message_y + 76,
        26,
        INK,
        mono=True,
        bold=True,
    )

    crc_y = 264
    draw_validation_card(ctx, card_x, crc_y, card_width, 102, "CRC")
    stored_crc = int(decoding.tag.crc, 2)
    show_text(
        ctx,
        f"calculated  {decoding.computed_crc:04X}",
        card_x + 128,
        crc_y + 36,
        17,
        MUTED,
        mono=True,
    )
    show_text(
        ctx,
        f"stored      {stored_crc:04X}",
        card_x + 128,
        crc_y + 73,
        17,
        INK,
        mono=True,
    )
    draw_check(ctx, card_x + card_width - 30, crc_y + 51)


DIAGRAMS = {
    "decoding-rectification": (1040, 480, draw_rectification),
    "quiet-zone-references": (1028, 445, draw_quiet_zone_references),
    "colour-sampling": (1048, 436, draw_colour_sampling),
    "orientation-palette": (1074, 354, draw_orientation_palette),
    "validation": (1040, 398, draw_validation),
}
PREVIEW_DIAGRAM = "decoding-rectification"


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
