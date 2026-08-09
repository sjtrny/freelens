"""
Tests for valid_crc_deployed.

Run with pytest, or directly with `python test_crc_deployed.py`.
"""

import numpy as np

from freelens import (
    Tag,
    get_crc_inds,
    ind_bit_map,
    message_length_for_N,
    valid_crc_deployed,
)

# Melbourne tram tag from https://github.com/sjtrny/freelens/issues/1
tram_tag = np.array(
    [
        [0, 2, 3, 0, 1],
        [1, 2, 3, 0, 3],
        [3, 3, 0, 0, 2],
        [0, 0, 3, 2, 0],
        [3, 0, 3, 0, 2],
    ],
    dtype=np.uint8,
)


def test_melbourne_tram_tag():
    cells = tram_tag.ravel()
    bit_string = "".join(ind_bit_map[cell] for cell in cells)

    assert valid_crc_deployed(bit_string, n=5)
    assert Tag(bit_string, n=5).valid

    crc_cells = [cells[i] for i in get_crc_inds(5)]
    crc = int("".join(ind_bit_map[cell] for cell in crc_cells), 2)
    assert crc == 0xFFF2, f"expected 0xFFF2, got 0x{crc:04X}"


def test_tag_generation_roundtrip_5x5():
    message = "010010100000000010001100"

    tag = Tag.from_message(message, n=5)

    assert tag.valid
    assert tag.message == message


def test_tag_generation_all_sizes():
    for n in [5, 7, 9, 11]:
        message = "01" * (message_length_for_N(n) // 2)

        tag = Tag.from_message(message, n=n)

        assert tag.valid, f"generated {n}x{n} tag should be valid"
        assert tag.message == message, f"{n}x{n} message should round trip"


if __name__ == "__main__":
    test_melbourne_tram_tag()
    test_tag_generation_roundtrip_5x5()
    test_tag_generation_all_sizes()

    print("All tests passed")
