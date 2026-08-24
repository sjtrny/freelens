import cv2 as cv
import numpy as np
from PIL import Image

import freelens
from freelens import (
    Tag,
    _correct_colours,
    _quiet_zone_references,
    _rectify_frame,
    decode_frames,
)

POLYGON = np.array([[0, 0], [223, 0], [223, 223], [0, 223]], dtype=np.float32)
MESSAGE = "010010100000000010001100"


def test_quiet_zone_references_use_rectified_rings():
    white = np.array([170, 180, 190], dtype=np.uint8)
    black = np.array([20, 30, 40], dtype=np.uint8)
    image = np.full((90, 90, 3), white, dtype=np.uint8)
    image[10:-10, 10:-10] = black
    image[20:-20, 20:-20] = (100, 110, 120)
    valid_pixels = np.ones(image.shape[:2], dtype=bool)

    actual_black, actual_white = _quiet_zone_references(
        image, valid_pixels, cell_size=10
    )

    np.testing.assert_array_equal(actual_black, black)
    np.testing.assert_array_equal(actual_white, white)


def test_missing_quiet_zone_uses_identity_references():
    image = np.full((90, 90, 3), 100, dtype=np.uint8)
    valid_pixels = np.zeros(image.shape[:2], dtype=bool)

    black, white = _quiet_zone_references(image, valid_pixels, cell_size=10)

    np.testing.assert_array_equal(black, [0, 0, 0])
    np.testing.assert_array_equal(white, [255, 255, 255])


def test_quiet_zone_references_ignore_pixels_outside_source_image():
    image = np.full((90, 90, 3), (170, 180, 190), dtype=np.uint8)
    image[10:-10, 10:-10] = (20, 30, 40)
    image[20:-20, 20:-20] = (100, 110, 120)
    image[:, :10] = (0, 0, 0)
    valid_pixels = np.ones(image.shape[:2], dtype=bool)
    valid_pixels[:, :10] = False

    black, white = _quiet_zone_references(image, valid_pixels, cell_size=10)

    np.testing.assert_array_equal(black, [20, 30, 40])
    np.testing.assert_array_equal(white, [170, 180, 190])


def test_rectification_samples_perspective_quiet_zone_rings():
    n = 5
    cell_size = 10
    quiet_zone_pixels = cell_size * (n + 4)
    white = np.array([170, 180, 190], dtype=np.uint8)
    black = np.array([20, 30, 40], dtype=np.uint8)
    source = np.full((quiet_zone_pixels, quiet_zone_pixels, 3), white, dtype=np.uint8)
    source[cell_size:-cell_size, cell_size:-cell_size] = black
    source[2 * cell_size : -2 * cell_size, 2 * cell_size : -2 * cell_size] = (
        100,
        110,
        120,
    )
    source_corners = np.float32(
        [
            [0, 0],
            [quiet_zone_pixels, 0],
            [quiet_zone_pixels, quiet_zone_pixels],
            [0, quiet_zone_pixels],
        ]
    )
    destination_corners = np.float32([[20, 15], [135, 30], [120, 125], [10, 105]])
    transform = cv.getPerspectiveTransform(source_corners, destination_corners)
    scene = cv.warpPerspective(source, transform, (150, 140), borderValue=(1, 2, 3))
    frame_corners = np.float32(
        [
            [cell_size, cell_size],
            [quiet_zone_pixels - cell_size, cell_size],
            [quiet_zone_pixels - cell_size, quiet_zone_pixels - cell_size],
            [cell_size, quiet_zone_pixels - cell_size],
        ]
    )
    polygon = cv.perspectiveTransform(frame_corners[None], transform)[0]
    scene_rgba = np.dstack((scene, np.full(scene.shape[:2], 255, dtype=np.uint8)))

    frame, actual_black, actual_white = _rectify_frame(
        scene_rgba, polygon, n, cell_size
    )

    assert frame.shape == (70, 70, 3)
    np.testing.assert_allclose(actual_black, black, atol=1)
    np.testing.assert_allclose(actual_white, white, atol=1)


def test_colour_correction_maps_quiet_zone_references_to_black_and_white():
    black = np.array([10, 20, 30])
    white = np.array([110, 220, 130])
    midpoint = (black + white) // 2
    image = np.array([[black, midpoint, white]], dtype=np.uint8)

    corrected = _correct_colours(image, black, white)

    np.testing.assert_allclose(
        corrected,
        np.array([[[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]]),
    )


def test_colour_correction_leaves_unusable_channels_on_rgb_scale():
    image = np.array([[[64, 100, 192]]], dtype=np.uint8)

    corrected = _correct_colours(
        image,
        black=np.array([0, 50, 255]),
        white=np.array([255, 50, 0]),
    )

    np.testing.assert_allclose(
        corrected,
        np.array([[[64 / 255, 100 / 255, 192 / 255]]]),
    )


def test_strict_decoding_uses_quiet_zone_colour_calibration(monkeypatch):
    image = Tag.from_message(MESSAGE).to_image(quiet_pad_size=0)
    corrected = np.asarray(image, dtype=np.float32) / 255
    monkeypatch.setattr(
        freelens,
        "_correct_colours",
        lambda image, black, white: corrected,
    )

    tags = decode_frames(
        image=Image.new("RGB", (224, 224), "black"),
        polygons=[POLYGON],
        require_valid_crc=True,
    )

    assert [tag.message for tag in tags] == [MESSAGE]


def test_valid_frame_is_still_colour_calibrated(monkeypatch):
    image = Tag.from_message(MESSAGE).to_image(quiet_pad_size=0)
    calls = []
    original = freelens._correct_colours

    def record_call(image, black, white):
        calls.append((black, white))
        return original(image, black, white)

    monkeypatch.setattr(freelens, "_correct_colours", record_call)

    tags = decode_frames(image, [POLYGON], require_valid_crc=True)

    assert [tag.message for tag in tags] == [MESSAGE]
    assert len(calls) == 1


def test_non_strict_decoding_uses_colour_calibration(monkeypatch):
    image = Tag.from_message(MESSAGE).to_image(quiet_pad_size=0)
    calls = []
    original = freelens._correct_colours

    def record_call(image, black, white):
        calls.append((black, white))
        return original(image, black, white)

    monkeypatch.setattr(freelens, "_correct_colours", record_call)

    tags = decode_frames(image, [POLYGON], require_valid_crc=False)

    assert [tag.message for tag in tags] == [MESSAGE]
    assert len(calls) == 1
