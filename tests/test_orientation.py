import cv2 as cv
import numpy as np
import pytest
from PIL import Image

from freelens import Tag, decode_frames, detect_tags

MESSAGE = "010010100000000010001100"


@pytest.mark.parametrize(
    "transpose",
    (
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    ),
    ids=("0", "90", "180", "270"),
)
def test_detect_tags_orients_rotated_tags(transpose):
    expected = Tag.from_message(MESSAGE)
    image = expected.to_image()
    if transpose is not None:
        image = image.transpose(transpose)

    tags = detect_tags(image, n=5, validate_crc=True, require_valid_crc=True)

    assert len(tags) == 1
    assert tags[0].bit_string == expected.bit_string
    assert tags[0].crc_valid is True


@pytest.mark.parametrize("starting_corner", range(4))
def test_decode_frames_preserves_cyclic_contour_order(starting_corner):
    expected = Tag.from_message(MESSAGE)
    source_image = np.asarray(expected.to_image(cell_size=32, quiet_pad_size=0))
    source_corners = np.float32([[0, 0], [223, 0], [223, 223], [0, 223]])

    # This reproduces the perspective geometry that made the old point-ordering
    # heuristic swap the right-hand corners of evaluation image 0009.
    destination_corners = np.float32([[261, 20], [718, 177], [553, 400], [20, 217]])
    transform = cv.getPerspectiveTransform(source_corners, destination_corners)
    projected = cv.warpPerspective(
        source_image,
        transform,
        (740, 420),
        borderValue=(255, 255, 255),
    )

    # Contour approximation can begin at any vertex, but preserves adjacency.
    contour = destination_corners[[3, 0, 1, 2]]
    contour = np.roll(contour, -starting_corner, axis=0)
    tags = decode_frames(
        Image.fromarray(projected),
        [contour],
        n=5,
        validate_crc=True,
        require_valid_crc=True,
    )

    assert len(tags) == 1
    assert tags[0].bit_string == expected.bit_string
