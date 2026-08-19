import numpy as np
import pytest
from PIL import Image

from freelens import Tag, detect_tags, message_length_for_N

MESSAGE = "010010100000000010001100"


def test_detects_shadowed_tag_beside_bright_scene():
    tag = (
        Tag.from_message(MESSAGE)
        .to_image()
        .resize((160, 160), Image.Resampling.NEAREST)
    )
    shadowed_tag = np.round(np.array(tag, dtype=np.float32) * 170 / 255).astype(
        np.uint8
    )
    scene = np.full((600, 600, 3), 255, dtype=np.uint8)
    scene[220:380, 220:380] = shadowed_tag

    tags = detect_tags(
        Image.fromarray(scene),
        validate_crc=True,
        require_valid_crc=True,
    )

    assert [tag.message for tag in tags] == [MESSAGE]


@pytest.mark.parametrize("n", (5, 7, 9, 11))
def test_quiet_zone_filter_supports_every_tag_size(n):
    expected = Tag.from_message("0" * message_length_for_N(n), n=n)

    tags = detect_tags(
        expected.to_image(),
        n=n,
        validate_crc=n == 5,
        require_valid_crc=n == 5,
    )

    assert expected.bit_string in [tag.bit_string for tag in tags]
