import pytest
from PIL import Image

from freelens import Tag, detect_tags

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
