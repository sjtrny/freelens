"""Tests for the deployed 5x5 checksum (realcrc).

Two independent real tags are used as fixtures:
  * B1269C  -- a printed-label tag.
  * the Melbourne-tram tag from issue #1, whose grid the documented CRC fails to
    reproduce.

The module under test needs only numpy, so most of these run without the
detector's CV deps. The tests that compare against the documented CRC import
freelens and auto-skip when ``cv2`` / ``crc`` are unavailable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import realcrc

# value <-> bits, MSB-first
_BITS = {0: "00", 1: "01", 2: "10", 3: "11"}


def grid_to_bits(grid):
    """5x5 grid of cell values (0..3), row-major -> 50-char tag bit string."""
    return "".join(_BITS[v] for row in grid for v in row)


# Full grids, row-major, C=0 M=1 Y=2 K=3. Corners read clockwise from top-left as
# 0,1,2,3 and centre 0, i.e. canonical orientation.
B1269C = [
    [0, 0, 2, 1, 1],
    [2, 1, 1, 2, 3],
    [0, 3, 0, 1, 3],
    [3, 0, 2, 2, 0],
    [3, 2, 2, 1, 2],
]
ISSUE1 = [  # dataset/positives Melbourne tram tag from issue #1
    [0, 2, 3, 0, 1],
    [1, 2, 3, 0, 3],
    [3, 3, 0, 0, 2],
    [0, 0, 3, 2, 0],
    [3, 0, 3, 0, 2],
]


def _message_bits(grid):
    cells = [grid_to_bits(grid)[i:i + 2] for i in range(0, 50, 2)]
    return "".join(cells[i] for i in realcrc.get_message_inds(5))


def test_b1269c_valid():
    assert realcrc.valid_real_crc(grid_to_bits(B1269C)) is True


def test_issue1_valid():
    assert realcrc.valid_real_crc(grid_to_bits(ISSUE1)) is True


def test_b1269c_crc_value():
    bits = realcrc.real_crc_bit_string(_message_bits(B1269C))
    assert int(bits, 2) == 0x39A7


def test_issue1_crc_value():
    bits = realcrc.real_crc_bit_string(_message_bits(ISSUE1))
    assert int(bits, 2) == 0xFFF2


def test_real_crc_cells_match_tag():
    # crc cells the generator emits must equal the cells actually on the tag,
    # in get_crc_inds order.
    for grid in (B1269C, ISSUE1):
        cells = [grid_to_bits(grid)[i:i + 2] for i in range(0, 50, 2)]
        on_tag = [cells[i] for i in realcrc.get_crc_inds(5)]
        assert realcrc.real_crc_cells(_message_bits(grid)) == on_tag


def test_apply_real_crc_builds_valid_tag():
    message = "101010101011000000001011"
    tag_bits = realcrc.apply_real_crc(message)
    assert len(tag_bits) == 50
    assert realcrc.valid_real_crc(tag_bits) is True
    cells = [tag_bits[i:i + 2] for i in range(0, 50, 2)]
    recovered = "".join(cells[i] for i in realcrc.get_message_inds(5))
    assert recovered == message


def test_corrupting_a_message_cell_is_detected():
    bits = list(grid_to_bits(ISSUE1))
    mi = realcrc.get_message_inds(5)[0]
    original = "".join(bits[2 * mi:2 * mi + 2])
    bits[2 * mi:2 * mi + 2] = list("11" if original != "11" else "00")
    assert realcrc.valid_real_crc("".join(bits)) is False


def test_larger_n_not_supported():
    with pytest.raises(NotImplementedError):
        realcrc.valid_real_crc("0" * (7 * 7 * 2), n=7)


def test_index_helpers_match_freelens():
    pytest.importorskip("cv2")  # freelens imports cv2 at module load
    pytest.importorskip("crc")
    import freelens
    assert realcrc.get_message_inds(5) == freelens.get_message_inds(5)
    assert realcrc.get_crc_inds(5) == freelens.get_crc_inds(5)


def test_documented_fails_where_deployed_passes():
    """The crux of issue #1: the documented CRC rejects a real tag the deployed
    checksum accepts."""
    pytest.importorskip("cv2")
    pytest.importorskip("crc")
    import freelens
    bits = grid_to_bits(ISSUE1)
    assert freelens.valid_crc(bits) is False
    assert realcrc.valid_real_crc(bits) is True
