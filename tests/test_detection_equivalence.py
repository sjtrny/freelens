"""End-to-end guard that detection optimisations stay output-preserving.

The detection fast paths (contour pre-filtering, list-mode contour retrieval,
sampling cell centres before colour correction) are only worth keeping if they
decode exactly what the straightforward implementation decoded. These tests pin
the decoded bits for scenes that exercise the whole pipeline.
"""

import cv2 as cv
import numpy as np
import pytest
from PIL import Image

from freelens import Tag, detect_tags

MESSAGES = (
    "010010100000000010001100",
    "101010101011000000001011",
    "000000000000000000000000",
    "111111111111111111111111",
)


def _scene(messages, size=(900, 700), seed=0):
    """Composite tags onto a noisy background under random perspective warps."""
    rng = np.random.default_rng(seed)
    width, height = size
    scene = rng.integers(90, 190, (height, width, 3), dtype=np.uint8)

    # Distractor rectangles give the contour filters something to reject.
    for _ in range(12):
        x, y = rng.integers(0, width - 60), rng.integers(0, height - 60)
        w, h = rng.integers(10, 55), rng.integers(10, 55)
        colour = rng.integers(0, 256, 3).tolist()
        cv.rectangle(scene, (x, y), (x + w, y + h), colour, -1)

    for index, message in enumerate(messages):
        tag = np.asarray(Tag.from_message(message).to_image(quiet_pad_size=32))
        side = tag.shape[0] - 1
        source = np.float32([[0, 0], [side, 0], [side, side], [0, side]])

        origin_x = 40 + index * 300
        jitter = rng.integers(-12, 13, (4, 2))
        destination = np.float32(
            [
                [origin_x, 60],
                [origin_x + 240, 60],
                [origin_x + 240, 300],
                [origin_x, 300],
            ]
        )
        destination += jitter

        transform = cv.getPerspectiveTransform(source, destination)
        warped = cv.warpPerspective(tag, transform, (width, height))
        mask = cv.warpPerspective(
            np.full(tag.shape[:2], 255, np.uint8), transform, (width, height)
        )
        scene[mask > 0] = warped[mask > 0]

    return Image.fromarray(scene)


@pytest.mark.parametrize("message", MESSAGES)
def test_single_tag_decodes_to_its_message(message):
    tags = detect_tags(
        _scene([message]), n=5, validate_crc=True, require_valid_crc=True
    )

    assert [tag.bit_string for tag in tags] == [Tag.from_message(message).bit_string]


def test_multiple_tags_in_one_scene_all_decode():
    messages = MESSAGES[:3]
    expected = sorted(Tag.from_message(m).bit_string for m in messages)

    tags = detect_tags(
        _scene(messages, size=(1100, 700), seed=7),
        n=5,
        validate_crc=True,
        require_valid_crc=True,
    )

    # Sorted: contour retrieval order is not part of the contract.
    assert sorted(tag.bit_string for tag in tags) == expected


def test_background_without_tags_yields_nothing():
    assert (
        detect_tags(_scene([], seed=3), n=5, validate_crc=True, require_valid_crc=True)
        == []
    )
