import pytest

from freelens import (
    Tag,
    _crc_input_bytes,
    get_crc_inds,
    max_int_for_N,
    message_length_for_N,
)

GENERATION_CASES = (
    (
        7,
        bytes(range(8)),
        bytes.fromhex("00 30 10 20 30 40 50 64 1E"),
        0x15C9EA,
        [21, 22, 23, 3, 10, 17, 31, 38, 45, 25, 26, 27],
    ),
    (
        9,
        bytes(range(15)),
        bytes.fromhex("00 03 10 20 30 40 50 60 70 80 90 A0 B0 C0 74 3A"),
        0x00A9E1B7,
        [36, 37, 38, 39, 4, 13, 22, 31, 49, 58, 67, 76, 41, 42, 43, 44],
    ),
    (
        11,
        bytes(range(24)),
        bytes.fromhex(
            "00 00 70 20 30 40 50 60 70 80 90 A0 B0 C0 D0 E0 F1 01 11 21 31 41 54 58 5E"
        ),
        0xE8BB8233ED,
        [
            55,
            56,
            57,
            58,
            59,
            5,
            16,
            27,
            38,
            49,
            71,
            82,
            93,
            104,
            115,
            61,
            62,
            63,
            64,
            65,
        ],
    ),
)


def _bits(data):
    return "".join(f"{byte:08b}" for byte in data)


@pytest.mark.parametrize(
    ("n", "message_bytes", "expected_input", "expected_crc", "expected_crc_indices"),
    GENERATION_CASES,
    ids=("7x7", "9x9", "11x11"),
)
def test_larger_tag_generation_extends_observed_5x5_crc(
    n,
    message_bytes,
    expected_input,
    expected_crc,
    expected_crc_indices,
):
    message = _bits(message_bytes)

    tag = Tag.from_message(message, n=n)

    assert _crc_input_bytes(tag.cells, n) == expected_input
    assert get_crc_inds(n) == expected_crc_indices
    assert tag.message == message
    assert tag.crc == f"{expected_crc:0{4 * n - 4}b}"
    assert tag.crc_valid is None
    assert tag.valid is None
    assert tag.center_valid is True
    assert tag.corners_valid is True
    assert tag.to_image().size == (32 * (n + 6),) * 2


@pytest.mark.parametrize("n", (5, 7, 9, 11))
def test_maximum_message_value_fits_the_message_width(n):
    width = message_length_for_N(n)
    maximum = max_int_for_N(n)
    message = f"{maximum:0{width}b}"

    assert maximum == (2**width) - 1
    assert len(message) == width
    assert Tag.from_message(message, n=n).message == message


@pytest.mark.parametrize("n", (7, 9, 11))
def test_larger_generator_validates_message(n):
    width = message_length_for_N(n)

    with pytest.raises(TypeError, match="message must be a str"):
        Tag.from_message(None, n=n)
    with pytest.raises(ValueError, match=f"{n}x{n} messages"):
        Tag.from_message("0" * (width - 1), n=n)
    with pytest.raises(ValueError, match="only '0' and '1'"):
        Tag.from_message("0" * (width - 1) + "x", n=n)
