import cv2 as cv
import numpy as np
from PIL import Image

import freelens
from freelens import Tag, _correct_colours, _quiet_zone_references, decode_frames

POLYGON = np.array([[0, 0], [223, 0], [223, 223], [0, 223]], dtype=np.float32)
MESSAGE = "010010100000000010001100"


def test_quiet_zone_references_use_rings_around_frame():
    white = np.array([170, 180, 190], dtype=np.uint8)
    black = np.array([20, 30, 40], dtype=np.uint8)
    image = np.full((91, 91, 3), white, dtype=np.uint8)
    polygon = np.array([[10, 10], [80, 10], [80, 80], [10, 80]])
    cv.fillConvexPoly(image, polygon, color=black.tolist())
    cv.fillConvexPoly(
        image,
        np.array([[20, 20], [70, 20], [70, 70], [20, 70]]),
        color=(100, 110, 120),
    )

    actual_black, actual_white = _quiet_zone_references(image, polygon, n=5)

    np.testing.assert_array_equal(actual_black, black)
    np.testing.assert_array_equal(actual_white, white)


def test_missing_quiet_zone_uses_identity_references():
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    polygon = np.array([[0, 0], [9, 0], [9, 9], [0, 9]])

    black, white = _quiet_zone_references(image, polygon, n=5)

    np.testing.assert_array_equal(black, [0, 0, 0])
    np.testing.assert_array_equal(white, [255, 255, 255])


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
