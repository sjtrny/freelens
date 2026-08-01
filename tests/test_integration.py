"""Integration of the validate_crc scheme selector into freelens.

These import freelens (hence cv2 / crc) and auto-skip when those are absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("cv2")
pytest.importorskip("crc")
import freelens

_BITS = {0: "00", 1: "01", 2: "10", 3: "11"}
ISSUE1 = [
    [0, 2, 3, 0, 1],
    [1, 2, 3, 0, 3],
    [3, 3, 0, 0, 2],
    [0, 0, 3, 2, 0],
    [3, 0, 3, 0, 2],
]


def grid_to_bits(grid):
    return "".join(_BITS[v] for row in grid for v in row)


ISSUE1_BITS = grid_to_bits(ISSUE1)


def test_tag_affine_validates_real_tag():
    assert freelens.Tag(ISSUE1_BITS, validate_crc="affine").valid is True


def test_tag_patent_rejects_real_tag():
    assert freelens.Tag(ISSUE1_BITS, validate_crc="patent").valid is False


def test_tag_none_skips_validation():
    assert freelens.Tag(ISSUE1_BITS, validate_crc=None).valid is None


def test_tag_default_is_patent():
    # preserves prior behaviour: Tag.valid defaulted to the documented CRC
    assert freelens.Tag(ISSUE1_BITS).valid is False


def test_tag_bad_mode_raises():
    with pytest.raises(ValueError):
        freelens.Tag(ISSUE1_BITS, validate_crc="bogus")


def test_decode_frames_guards_mode():
    # invalid mode rejected even with no polygons to decode
    with pytest.raises(ValueError):
        freelens.decode_frames(None, [], validate_crc="nope")


def test_validate_bit_string_helper():
    assert freelens.validate_bit_string(ISSUE1_BITS, 5, "affine") is True
    assert freelens.validate_bit_string(ISSUE1_BITS, 5, "patent") is False
    assert freelens.validate_bit_string(ISSUE1_BITS, 5, None) is None
