"""Render the diagrams used by ``docs/ddtag-detection.md``.

Every stage is derived from the real detector and the same crop of
``dataset/images/0004.jpg``. Run this module with Cairo Visuals on
``PYTHONPATH`` to regenerate the SVG assets.
"""

from __future__ import annotations

import io
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

from docs.assets.ddtag_diagrams import (  # noqa: E402
    CONTENT_PADDING,
    MIN_ARROW_LENGTH,
    MIN_LABEL_SIZE,
    draw_step_title,
    layout_content,
    render_layout,
)
from freelens import (  # noqa: E402
    MIN_FRAME_AREA,
    contour_filter_candidates,
    frame_filter_polygons_4vertex,
    frame_filter_polygons_area,
    frame_filter_polygons_convex,
    frame_filter_polygons_squareish,
    reduce_poly_vertices,
)

try:
    from cairo_font import DEFAULT_FONT, set_font_from_file
except ImportError as error:  # pragma: no cover - supplied by Cairo Visuals
    raise RuntimeError(
        "Render these diagrams with the cairo-visuals project"
    ) from error

SOURCE_IMAGE = PROJECT_ROOT / "dataset" / "images" / "0004.jpg"
SOURCE_CROP = (1075, 947, 2167, 2039)
DETAIL_CROP = (250, 240, 840, 830)


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
CYAN = rgb("8FD3FF")
ORANGE = rgb("F79009")
RED = rgb("E5484D")


@dataclass(frozen=True)
class DetectionStages:
    image: Image.Image
    grayscale: np.ndarray
    threshold: np.ndarray
    contours: tuple[np.ndarray, ...]
    candidates: tuple[np.ndarray, ...]
    polygons: tuple[np.ndarray, ...]
    vertex_filtered: tuple[np.ndarray, ...]
    area_filtered: tuple[np.ndarray, ...]
    convex_filtered: tuple[np.ndarray, ...]
    frames: tuple[np.ndarray, ...]
    frame_contour: np.ndarray
    frame_polygon: np.ndarray


@cache
def load_stages():
    with Image.open(SOURCE_IMAGE) as source:
        image = source.convert("RGB").crop(SOURCE_CROP)

    image_array = np.asarray(image)
    grayscale = cv.cvtColor(image_array, cv.COLOR_RGB2GRAY)
    threshold = cv.adaptiveThreshold(
        grayscale,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        101,
        0,
    )
    contours, _ = cv.findContours(
        threshold,
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE,
    )
    candidates = contour_filter_candidates(contours, MIN_FRAME_AREA)
    polygons = reduce_poly_vertices(candidates)
    vertex_filtered = frame_filter_polygons_4vertex(polygons)
    area_filtered = frame_filter_polygons_area(vertex_filtered, MIN_FRAME_AREA)
    convex_filtered = frame_filter_polygons_convex(area_filtered)
    frames = frame_filter_polygons_squareish(convex_filtered)

    if len(frames) != 1:
        raise RuntimeError(
            f"Expected one frame in the documentation crop, got {len(frames)}"
        )

    frame_polygon = frames[0]
    frame_index = next(
        index
        for index, polygon in enumerate(polygons)
        if np.array_equal(polygon, frame_polygon)
    )

    return DetectionStages(
        image=image,
        grayscale=grayscale,
        threshold=threshold,
        contours=tuple(contours),
        candidates=tuple(candidates),
        polygons=tuple(polygons),
        vertex_filtered=tuple(vertex_filtered),
        area_filtered=tuple(area_filtered),
        convex_filtered=tuple(convex_filtered),
        frames=tuple(frames),
        frame_contour=candidates[frame_index],
        frame_polygon=frame_polygon,
    )


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


def stroke_round_rect(ctx, x, y, width, height, radius, colour, line_width=1.0):
    rounded_rect(ctx, x, y, width, height, radius)
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.stroke()


def draw_background(ctx, width, height):
    rounded_rect(ctx, 1.5, 1.5, width - 3, height - 3, 18)
    set_source(ctx, PAPER)
    ctx.fill_preserve()
    set_source(ctx, LINE)
    ctx.set_line_width(1.5)
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


def pil_surface(image):
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    output.seek(0)
    return cairo.ImageSurface.create_from_png(output)


def draw_image(ctx, image, x, y, size, *, border_colour=LINE, border_width=2):
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    surface = pil_surface(resized)
    ctx.save()
    ctx.rectangle(x, y, size, size)
    ctx.clip()
    ctx.set_source_surface(surface, x, y)
    ctx.paint()
    ctx.restore()
    set_source(ctx, border_colour)
    ctx.set_line_width(border_width)
    ctx.rectangle(x, y, size, size)
    ctx.stroke()


def draw_arrow(ctx, start_x, end_x, y, colour=MUTED, line_width=3):
    head = 10
    ctx.move_to(start_x, y)
    ctx.line_to(end_x, y)
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()
    ctx.move_to(end_x - head, y - 7)
    ctx.line_to(end_x, y)
    ctx.line_to(end_x - head, y + 7)
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.stroke()


def image_from_array(array):
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array, mode="RGB")


def contour_image(contours, colour, thickness):
    stages = load_stages()
    height, width = stages.threshold.shape
    background = np.empty((height, width, 3), dtype=np.uint8)
    background[:] = tuple(round(channel * 255) for channel in INK)
    cv.drawContours(
        background,
        list(contours),
        -1,
        tuple(round(channel * 255) for channel in colour),
        thickness,
        lineType=cv.LINE_AA,
    )
    return image_from_array(background)


def map_point(point, panel_x, panel_y, panel_size, crop=None):
    if crop is None:
        crop = (0, 0, load_stages().image.width, load_stages().image.height)
    left, top, right, bottom = crop
    return (
        panel_x + (float(point[0]) - left) * panel_size / (right - left),
        panel_y + (float(point[1]) - top) * panel_size / (bottom - top),
    )


def draw_path(
    ctx,
    points,
    panel_x,
    panel_y,
    panel_size,
    colour,
    line_width,
    *,
    crop=None,
    close=True,
    under_stroke=False,
):
    points = np.asarray(points).reshape(-1, 2)
    mapped = [map_point(point, panel_x, panel_y, panel_size, crop) for point in points]
    if not mapped:
        return
    ctx.move_to(*mapped[0])
    for point in mapped[1:]:
        ctx.line_to(*point)
    if close:
        ctx.close_path()
    if under_stroke:
        set_source(ctx, WHITE, 0.88)
        ctx.set_line_width(line_width + 4)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke_preserve()
    set_source(ctx, colour)
    ctx.set_line_width(line_width)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()


def draw_vertices(ctx, points, panel_x, panel_y, panel_size, *, crop=None):
    for point in np.asarray(points).reshape(-1, 2):
        x, y = map_point(point, panel_x, panel_y, panel_size, crop)
        ctx.arc(x, y, 6.5, 0, math.tau)
        set_source(ctx, WHITE)
        ctx.fill()
        ctx.arc(x, y, 4.2, 0, math.tau)
        set_source(ctx, RED)
        ctx.fill()


def draw_image_processing(ctx, width, height):
    stages = load_stages()
    panel_size = 310
    panel_y = 64
    panel_xs = (32, 404, 776)
    labels = ((None, "source image"), (1, "greyscale"), (2, "adaptive threshold"))
    images = (
        stages.image,
        image_from_array(stages.grayscale),
        image_from_array(stages.threshold),
    )

    for x, (step, label), image in zip(panel_xs, labels, images, strict=True):
        if step is None:
            show_text(ctx, label, x, 44, 18)
        else:
            draw_step_title(ctx, step, label, x, 23, panel_size)
        draw_image(ctx, image, x, panel_y, panel_size)

    center_y = panel_y + panel_size / 2
    draw_arrow(ctx, panel_xs[0] + panel_size + 12, panel_xs[1] - 12, center_y)
    draw_arrow(ctx, panel_xs[1] + panel_size + 12, panel_xs[2] - 12, center_y)


def draw_contour_candidates(ctx, width, height):
    stages = load_stages()
    panel_size = 360
    panel_y = 76
    left_x = 32
    right_x = width - CONTENT_PADDING - panel_size
    raw = contour_image(stages.contours, CYAN, 3)
    candidates = contour_image(stages.candidates, ORANGE, 5)

    draw_step_title(ctx, 3, "detect contours", left_x, 32, panel_size)
    draw_step_title(ctx, 4, "filter contours", right_x, 32, panel_size)
    draw_image(ctx, raw, left_x, panel_y, panel_size, border_colour=INK)
    draw_image(ctx, candidates, right_x, panel_y, panel_size, border_colour=INK)

    card_gap = 12 + MIN_ARROW_LENGTH + 10
    card_x = left_x + panel_size + card_gap
    card_y = panel_y + 100
    card_width = right_x - card_gap - card_x
    card_height = 160
    draw_arrow(ctx, left_x + panel_size + 12, card_x - 10, panel_y + panel_size / 2)
    draw_arrow(ctx, card_x + card_width + 10, right_x - 12, panel_y + panel_size / 2)
    fill_round_rect(ctx, card_x, card_y, card_width, card_height, 14, WHITE)
    stroke_round_rect(ctx, card_x, card_y, card_width, card_height, 14, LINE, 1.5)
    show_centered(ctx, "4 or more points", card_x, card_y + 20, card_width, 30, 18)
    ctx.move_to(card_x + 22, card_y + 72)
    ctx.line_to(card_x + card_width - 22, card_y + 72)
    set_source(ctx, LINE)
    ctx.set_line_width(1.5)
    ctx.stroke()
    show_centered(ctx, "bounding box", card_x, card_y + 85, card_width, 25, 17, MUTED)
    show_centered(
        ctx,
        "≥ 1,500 px²",
        card_x,
        card_y + 112,
        card_width,
        27,
        18,
        mono=True,
    )

    show_centered(
        ctx,
        f"{len(stages.contours):,} contours",
        left_x,
        panel_y + panel_size + 10,
        panel_size,
        28,
        21,
    )
    show_centered(
        ctx,
        f"{len(stages.candidates):,} candidates",
        right_x,
        panel_y + panel_size + 10,
        panel_size,
        28,
        21,
    )


def draw_polygon_fitting(ctx, width, height):
    stages = load_stages()
    panel_size = 380
    panel_y = 66
    left_x = 32
    right_x = width - CONTENT_PADDING - panel_size
    detail = stages.image.crop(DETAIL_CROP)
    threshold = image_from_array(stages.threshold).crop(DETAIL_CROP)

    draw_step_title(
        ctx,
        4,
        f"{len(stages.frame_contour)} sampled boundary points",
        left_x,
        23,
        panel_size,
    )
    draw_step_title(ctx, 5, "4 fitted vertices", right_x, 23, panel_size)
    draw_image(ctx, threshold, left_x, panel_y, panel_size)
    draw_image(ctx, detail, right_x, panel_y, panel_size)

    draw_path(
        ctx,
        stages.frame_contour,
        left_x,
        panel_y,
        panel_size,
        BLUE,
        3,
        crop=DETAIL_CROP,
    )
    for point in stages.frame_contour.reshape(-1, 2):
        point_x, point_y = map_point(
            point,
            left_x,
            panel_y,
            panel_size,
            DETAIL_CROP,
        )
        ctx.arc(point_x, point_y, 2.3, 0, math.tau)
        set_source(ctx, ORANGE)
        ctx.fill()

    draw_path(
        ctx,
        stages.frame_polygon,
        right_x,
        panel_y,
        panel_size,
        RED,
        5,
        crop=DETAIL_CROP,
        under_stroke=True,
    )
    draw_vertices(
        ctx,
        stages.frame_polygon,
        right_x,
        panel_y,
        panel_size,
        crop=DETAIL_CROP,
    )
    draw_arrow(ctx, left_x + panel_size + 18, right_x - 18, panel_y + panel_size / 2)


def draw_filter_row(ctx, x, y, width, label, count, *, divider=True):
    show_text(ctx, label, x + 22, y + 35, 19)
    badge_size = 42
    badge_x = x + width - badge_size - 18
    fill_round_rect(ctx, badge_x, y + 10, badge_size, badge_size, badge_size / 2, PALE)
    show_centered(
        ctx,
        str(count),
        badge_x,
        y + 10,
        badge_size,
        badge_size,
        18,
        INK,
        mono=True,
        bold=True,
    )
    if divider:
        ctx.move_to(x + 20, y + 63)
        ctx.line_to(x + width - 20, y + 63)
        set_source(ctx, LINE)
        ctx.set_line_width(1.5)
        ctx.stroke()


def draw_frame_filters(ctx, width, height):
    stages = load_stages()
    panel_size = 320
    panel_y = 68
    left_x = 32
    filter_x = left_x + panel_size + 10 + MIN_ARROW_LENGTH + 10
    filter_width = 356
    right_x = width - CONTENT_PADDING - panel_size

    polygon_background = Image.new(
        "RGB",
        stages.image.size,
        tuple(round(channel * 255) for channel in INK),
    )
    draw_image(ctx, polygon_background, left_x, panel_y, panel_size, border_colour=INK)
    for polygon in stages.polygons:
        points = np.asarray(polygon)
        if points.size < 4:
            continue
        draw_path(
            ctx,
            points,
            left_x,
            panel_y,
            panel_size,
            ORANGE,
            1.6,
        )

    draw_image(ctx, stages.image, right_x, panel_y, panel_size)
    draw_path(
        ctx,
        stages.frame_polygon,
        right_x,
        panel_y,
        panel_size,
        RED,
        5,
        under_stroke=True,
    )
    draw_vertices(ctx, stages.frame_polygon, right_x, panel_y, panel_size)

    draw_step_title(
        ctx,
        5,
        f"{len(stages.polygons)} fitted polygons",
        left_x,
        23,
        panel_size,
    )
    draw_step_title(ctx, 6, "filter polygons", filter_x, 23, filter_width)
    draw_step_title(
        ctx,
        6,
        f"{len(stages.frames)} possible frame",
        right_x,
        23,
        panel_size,
    )

    card_y = 70
    card_height = 316
    fill_round_rect(ctx, filter_x, card_y, filter_width, card_height, 16, WHITE)
    stroke_round_rect(ctx, filter_x, card_y, filter_width, card_height, 16, LINE, 1.5)
    rows = (
        ("4 vertices", len(stages.vertex_filtered)),
        ("area ≥ 1,500 px²", len(stages.area_filtered)),
        ("convex", len(stages.convex_filtered)),
        ("roughly square", len(stages.frames)),
    )
    for index, (label, count) in enumerate(rows):
        draw_filter_row(
            ctx,
            filter_x,
            card_y + index * 79,
            filter_width,
            label,
            count,
            divider=index < len(rows) - 1,
        )

    center_y = panel_y + panel_size / 2
    draw_arrow(ctx, left_x + panel_size + 10, filter_x - 10, center_y)
    draw_arrow(ctx, filter_x + filter_width + 10, right_x - 10, center_y)


DIAGRAMS = {
    "image-processing": (1118, 406, draw_image_processing),
    "contour-candidates": (1068, 489, draw_contour_candidates),
    "polygon-fitting": (940, 478, draw_polygon_fitting),
    "frame-filters": (1176, 420, draw_frame_filters),
}
PREVIEW_DIAGRAM = "image-processing"


def render_diagram(surface_factory, name):
    width, height, drawer = DIAGRAMS[name]
    layout = layout_content(drawer, width, height)
    return render_layout(surface_factory, layout)


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
