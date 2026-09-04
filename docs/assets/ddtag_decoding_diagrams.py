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
    corrected_frame: Image.Image
    corrected_frame_lab: np.ndarray
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
    corrected_frame = np.round(_correct_colours(frame_rgb, black, white) * 255).astype(
        np.uint8
    )
    corrected_frame_lab = cv.cvtColor(corrected_frame, cv.COLOR_RGB2Lab)
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
        corrected_frame=Image.fromarray(corrected_frame, mode="RGB"),
        corrected_frame_lab=corrected_frame_lab,
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


def draw_step_title(ctx, step, label, x, y, width, *, size=18):
    badge_size = 28
    badge_x = x
    fill_round_rect(
        ctx,
        badge_x,
        y,
        badge_size,
        badge_size,
        badge_size / 2,
        BLUE,
    )
    show_centered(
        ctx,
        str(step),
        badge_x,
        y,
        badge_size,
        badge_size,
        14,
        WHITE,
        mono=True,
        bold=True,
    )
    show_text(ctx, label, badge_x + badge_size + 10, y + 21, size)


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

    draw_step_title(ctx, 1, "preserve cyclic vertices", left_x, 17, panel_size)
    draw_step_title(ctx, 2, "rectify to square", right_x, 17, panel_size)

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


def draw_reference_chip(ctx, x, y, colour, name, values, highlight):
    swatch = 34
    fill_rect(ctx, x, y, swatch, swatch, colour)
    stroke_rect(ctx, x, y, swatch, swatch, highlight, 2.5)
    show_text(ctx, name, x + 44, y + 14, 13)
    show_text(
        ctx,
        " ".join(str(round(value)) for value in values),
        x + 44,
        y + 33,
        11,
        MUTED,
        mono=True,
    )


def draw_quiet_zone_references(ctx, width, height):
    decoding = load_decoding_stages()
    panel_size = 300
    panel_y = 62
    panel_xs = (32, 402, 772)

    draw_step_title(ctx, 2, "rectified frame", panel_xs[0], 18, panel_size)
    draw_step_title(ctx, 3, "measure quiet zones", panel_xs[1], 18, panel_size)
    draw_step_title(ctx, 4, "crop black frame", panel_xs[2], 18, panel_size)

    draw_image(ctx, decoding.rectified, panel_xs[0], panel_y, panel_size)
    draw_image(ctx, decoding.rectified, panel_xs[1], panel_y, panel_size)
    draw_ring_overlay(ctx, panel_xs[1], panel_y, panel_size, 0, BLUE)
    draw_ring_overlay(ctx, panel_xs[1], panel_y, panel_size, 1, ORANGE)
    draw_image(ctx, decoding.frame, panel_xs[2], panel_y, panel_size)

    draw_reference_chip(
        ctx,
        panel_xs[1],
        374,
        colour_from_sample(decoding.white),
        "white",
        decoding.white,
        BLUE,
    )
    draw_reference_chip(
        ctx,
        panel_xs[1] + 155,
        374,
        colour_from_sample(decoding.black),
        "black",
        decoding.black,
        ORANGE,
    )

    center_y = panel_y + panel_size / 2
    draw_arrow(ctx, panel_xs[0] + panel_size + 12, panel_xs[1] - 12, center_y)
    draw_arrow(ctx, panel_xs[1] + panel_size + 12, panel_xs[2] - 12, center_y)


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


def draw_lab_channels(ctx, lab, x, y, size):
    fill_round_rect(ctx, x, y, size, size, 12, WHITE)
    stroke_round_rect(ctx, x, y, size, size, 12, LINE, 1.5)
    thumbnail = 78
    gap = 12
    start_x = x + (size - 3 * thumbnail - 2 * gap) / 2
    image_y = y + 75
    for index, label in enumerate(("L*", "a*", "b*")):
        channel = Image.fromarray(lab[..., index], mode="L").convert("RGB")
        image_x = start_x + index * (thumbnail + gap)
        draw_image(ctx, channel, image_x, image_y, thumbnail)
        show_centered(ctx, label, image_x, image_y + thumbnail + 8, thumbnail, 24, 16)
    show_centered(
        ctx,
        "three values per pixel",
        x,
        y + 215,
        size,
        28,
        15,
        MUTED,
    )


def draw_colour_sampling(ctx, width, height):
    decoding = load_decoding_stages()
    panel_size = 280
    panel_y = 72
    left_x = 32
    middle_x = 382
    sample_x = 732
    sample_width = 426

    draw_step_title(ctx, 5, "map RGB range", left_x, 20, panel_size)
    draw_step_title(ctx, 6, "convert to CIELab", middle_x, 20, panel_size)
    draw_step_title(ctx, 7, "sample cell centres", sample_x, 20, sample_width)

    draw_image(ctx, decoding.corrected_frame, left_x, panel_y, panel_size)
    show_centered(
        ctx,
        "black = 0     white = 255",
        left_x,
        358,
        panel_size,
        24,
        14,
        MUTED,
        mono=True,
    )
    draw_lab_channels(
        ctx,
        decoding.corrected_frame_lab,
        middle_x,
        panel_y,
        panel_size,
    )

    fill_round_rect(ctx, sample_x, panel_y, sample_width, panel_size, 12, WHITE)
    stroke_round_rect(ctx, sample_x, panel_y, sample_width, panel_size, 12, LINE, 1.5)
    sample_size = 180
    frame_x = sample_x + 12
    grid_x = sample_x + sample_width - sample_size - 12
    image_y = panel_y + 58
    show_centered(ctx, "frame", frame_x, panel_y + 18, sample_size, 24, 15, MUTED)
    show_centered(
        ctx, "7 × 7 samples", grid_x, panel_y + 18, sample_size, 24, 15, MUTED
    )
    draw_image(ctx, decoding.corrected_frame, frame_x, image_y, sample_size)
    draw_sample_centres(ctx, frame_x, image_y, sample_size)
    draw_sample_grid(ctx, decoding.corrected_rgb, grid_x, image_y, sample_size)
    draw_arrow(ctx, frame_x + sample_size + 8, grid_x - 8, image_y + sample_size / 2)

    center_y = panel_y + panel_size / 2
    draw_arrow(ctx, left_x + panel_size + 12, middle_x - 12, center_y)
    draw_arrow(ctx, middle_x + panel_size + 12, sample_x - 12, center_y)


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


def draw_palette_panel(ctx, oriented, x, y, size):
    corners = (
        oriented[0, 0],
        oriented[0, -1],
        oriented[-1, -1],
        oriented[-1, 0],
    )
    positions = ((0, 0), (0, 1), (1, 1), (1, 0))
    cell = size / 2
    for index, ((row, column), sample) in enumerate(
        zip(positions, corners, strict=True)
    ):
        cell_x = x + column * cell
        cell_y = y + row * cell
        fill_rect(ctx, cell_x, cell_y, cell, cell, colour_from_sample(sample))
        text_colour = WHITE if index == 3 else INK
        show_centered(
            ctx,
            BIT_VALUES[index],
            cell_x,
            cell_y,
            cell,
            cell,
            18,
            text_colour,
            mono=True,
            bold=True,
        )
    stroke_rect(ctx, x, y, size, size, INK, 2)


def draw_orientation_palette(ctx, width, height):
    decoding = load_decoding_stages()
    panel_size = 230
    panel_y = 60
    panel_xs = (32, 332, 632, 932)
    observed = decoding.corrected_rgb[1 : N + 1, 1 : N + 1]
    oriented = decoding.rotated_rgb[1 : N + 1, 1 : N + 1]

    titles = (
        (8, "find darkest"),
        (8, "rotate to bottom-left"),
        (9, "read corner palette"),
        (10, "classify cells"),
    )
    for x, (step, label) in zip(panel_xs, titles, strict=True):
        draw_step_title(ctx, step, label, x, 16, panel_size, size=16)

    draw_sample_grid(ctx, observed, panel_xs[0], panel_y, panel_size)
    draw_grid_highlight(ctx, panel_xs[0], panel_y, panel_size, 4, 4, RED, 5)

    draw_sample_grid(ctx, oriented, panel_xs[1], panel_y, panel_size)
    draw_grid_highlight(ctx, panel_xs[1], panel_y, panel_size, 4, 0, RED, 5)

    draw_palette_panel(ctx, oriented, panel_xs[2], panel_y, panel_size)
    draw_class_grid(
        ctx, decoding.classes, panel_xs[3], panel_y, panel_size, labels=True
    )

    center_y = panel_y + panel_size / 2
    for left_x, right_x in zip(panel_xs[:-1], panel_xs[1:], strict=True):
        draw_arrow(ctx, left_x + panel_size + 14, right_x - 14, center_y)
    show_centered(
        ctx,
        "90°",
        panel_xs[0] + panel_size,
        center_y - 39,
        panel_xs[1] - panel_xs[0] - panel_size,
        22,
        15,
        MUTED,
    )


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


def draw_validation_card(ctx, x, y, width, height, step, title):
    fill_round_rect(ctx, x, y, width, height, 14, WHITE)
    stroke_round_rect(ctx, x, y, width, height, 14, LINE, 1.5)
    draw_step_title(ctx, step, title, x + 20, y + 14, width - 40, size=17)


def draw_validation(ctx, width, height):
    decoding = load_decoding_stages()
    grid_size = 300
    grid_x = CONTENT_PADDING
    grid_y = 66
    card_x = 388
    card_width = width - CONTENT_PADDING - card_x

    show_centered(ctx, "palette values", grid_x, 20, grid_size, 27, 20)
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
    draw_validation_card(
        ctx,
        card_x,
        corners_y,
        card_width,
        98,
        11,
        "check corner order",
    )
    swatch = 42
    swatch_x = card_x + 220
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
    draw_validation_card(
        ctx,
        card_x,
        message_y,
        card_width,
        102,
        12,
        "extract message",
    )
    grouped_message = " ".join(
        decoding.tag.message[index : index + 8]
        for index in range(0, len(decoding.tag.message), 8)
    )
    show_text(ctx, grouped_message, card_x + 190, message_y + 34, 16, MUTED, mono=True)
    show_text(
        ctx,
        f"{int(decoding.tag.message, 2):06X}",
        card_x + 190,
        message_y + 76,
        26,
        INK,
        mono=True,
        bold=True,
    )

    crc_y = 264
    draw_validation_card(
        ctx,
        card_x,
        crc_y,
        card_width,
        102,
        12,
        "compare CRC",
    )
    stored_crc = int(decoding.tag.crc, 2)
    show_text(
        ctx,
        f"calculated  {decoding.computed_crc:04X}",
        card_x + 190,
        crc_y + 36,
        17,
        MUTED,
        mono=True,
    )
    show_text(
        ctx,
        f"stored      {stored_crc:04X}",
        card_x + 190,
        crc_y + 73,
        17,
        INK,
        mono=True,
    )
    draw_check(ctx, card_x + card_width - 30, crc_y + 51)


DIAGRAMS = {
    "decoding-rectification": (1040, 480, draw_rectification),
    "quiet-zone-references": (1104, 445, draw_quiet_zone_references),
    "colour-sampling": (1190, 414, draw_colour_sampling),
    "orientation-palette": (1194, 322, draw_orientation_palette),
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
