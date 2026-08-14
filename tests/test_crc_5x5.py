import pytest

from freelens import (
    CRC_INPUT_INDICES_5X5,
    Tag,
    _crc_input_bytes_5x5,
    compute_crc_5x5,
    compute_patent_crc,
    get_crc_inds,
    get_message_inds,
    get_patent_crc_inds,
    valid_crc,
)

MELBOURNE_FULL_TAG_BITS = "00101100010110110011111100001000001110001100110010"
MELBOURNE_MESSAGE = "010010100000000010001100"
MELBOURNE_INPUT_BITS = "00010011101000000000100001110010"
MELBOURNE_INPUT_BYTES = bytes.fromhex("13 A0 08 72")
MELBOURNE_CRC = 0xFFF2
MELBOURNE_CRC_CELLS = ("11", "11", "11", "11", "11", "11", "00", "10")

B1269C_FULL_TAG_BITS = "00001001011001011011001100011111001010001110100110"
B1269C_MESSAGE = "101100010010011010011100"
B1269C_INPUT_BITS = "00101111000100100110100101110010"
B1269C_INPUT_BYTES = bytes.fromhex("2F 12 69 72")
B1269C_CRC = 0x39A7
B1269C_CRC_CELLS = ("00", "11", "10", "01", "10", "10", "01", "11")


KNOWN_TAGS = (
    (
        MELBOURNE_FULL_TAG_BITS,
        MELBOURNE_MESSAGE,
        MELBOURNE_INPUT_BITS,
        MELBOURNE_INPUT_BYTES,
        MELBOURNE_CRC,
        MELBOURNE_CRC_CELLS,
    ),
    (
        B1269C_FULL_TAG_BITS,
        B1269C_MESSAGE,
        B1269C_INPUT_BITS,
        B1269C_INPUT_BYTES,
        B1269C_CRC,
        B1269C_CRC_CELLS,
    ),
)


def _cells(bit_string):
    return tuple(
        bit_string[offset : offset + 2] for offset in range(0, len(bit_string), 2)
    )


def _replace_cell(bit_string, index, replacement):
    cells = list(_cells(bit_string))
    assert cells[index] != replacement
    cells[index] = replacement
    return "".join(cells)


def test_crc_input_indices_5x5():
    assert CRC_INPUT_INDICES_5X5 == (
        0,
        5,
        15,
        20,
        1,
        6,
        16,
        21,
        3,
        8,
        18,
        23,
        4,
        9,
        19,
        24,
    )


def test_crc_cell_order_5x5():
    assert get_crc_inds(5) == [10, 11, 2, 7, 17, 22, 13, 14]
    assert get_patent_crc_inds(5) == [2, 7, 10, 11, 13, 14, 17, 22]


def test_patent_and_deployed_5x5_crc_differ():
    assert compute_patent_crc(MELBOURNE_MESSAGE, 5) == 0xE751
    assert compute_crc_5x5(_cells(MELBOURNE_FULL_TAG_BITS)) == MELBOURNE_CRC


@pytest.mark.parametrize(
    (
        "full_tag_bits",
        "message",
        "input_bits",
        "input_bytes",
        "expected_crc",
        "expected_crc_cells",
    ),
    KNOWN_TAGS,
    ids=("melbourne", "b1269c"),
)
def test_known_answer_vectors(
    full_tag_bits,
    message,
    input_bits,
    input_bytes,
    expected_crc,
    expected_crc_cells,
):
    cells = _cells(full_tag_bits)

    assert "".join(cells[index] for index in CRC_INPUT_INDICES_5X5) == input_bits
    assert _crc_input_bytes_5x5(cells) == input_bytes
    assert compute_crc_5x5(cells) == expected_crc
    assert tuple(cells[index] for index in get_crc_inds(5)) == expected_crc_cells

    tag = Tag(full_tag_bits)
    assert tag.bit_string == full_tag_bits
    assert tag.message == message
    assert tag.crc == f"{expected_crc:016b}"
    assert tag.crc_valid is True
    assert tag.center_valid is True
    assert tag.corners_valid is True
    assert valid_crc(full_tag_bits) is True


def test_melbourne_byte_packing_is_exact():
    data = _crc_input_bytes_5x5(_cells(MELBOURNE_FULL_TAG_BITS))

    assert data == b"\x13\xa0\x08\x72"


def test_leading_zero_byte_is_preserved():
    cells = ["01"] * 25
    for index in CRC_INPUT_INDICES_5X5[:4]:
        cells[index] = "00"

    data = _crc_input_bytes_5x5(cells)

    assert len(data) == 4
    assert data[0] == 0x00


def test_generator_matches_complete_melbourne_fixture():
    tag = Tag.from_message(MELBOURNE_MESSAGE)

    assert tag.bit_string == MELBOURNE_FULL_TAG_BITS
    assert tag.crc == f"{MELBOURNE_CRC:016b}"
    assert tag.crc_valid is True


def test_message_corruption_is_detected():
    corrupt = _replace_cell(MELBOURNE_FULL_TAG_BITS, get_message_inds(5)[0], "00")

    assert Tag(corrupt).crc_valid is False


def test_corner_corruption_is_detected_and_named_separately():
    corrupt = _replace_cell(MELBOURNE_FULL_TAG_BITS, 0, "01")
    tag = Tag(corrupt)

    assert tag.crc_valid is False
    assert tag.corners_valid is False
    assert tag.center_valid is True


def test_crc_corruption_is_detected():
    corrupt = _replace_cell(MELBOURNE_FULL_TAG_BITS, get_crc_inds(5)[0], "00")

    assert Tag(corrupt).crc_valid is False


def test_center_corruption_does_not_change_crc_result():
    corrupt = _replace_cell(MELBOURNE_FULL_TAG_BITS, 12, "01")
    tag = Tag(corrupt)

    assert tag.crc_valid is True
    assert tag.center_valid is False
    assert tag.corners_valid is True


def test_validation_can_be_skipped():
    corrupt = _replace_cell(MELBOURNE_FULL_TAG_BITS, 0, "01")
    tag = Tag(corrupt, validate_crc=False)

    assert tag.crc_valid is None
    assert tag.valid is None
    assert tag.corners_valid is False


@pytest.mark.parametrize(
    "bit_string",
    (
        "",
        MELBOURNE_FULL_TAG_BITS[:-1],
        MELBOURNE_FULL_TAG_BITS + "0",
        MELBOURNE_FULL_TAG_BITS + "trailing data",
        MELBOURNE_FULL_TAG_BITS[:-1] + "x",
    ),
)
def test_tag_rejects_malformed_bit_strings(bit_string):
    with pytest.raises(ValueError):
        Tag(bit_string)


@pytest.mark.parametrize("bit_string", (None, 123, b"0" * 50))
def test_tag_rejects_nonstring_input(bit_string):
    with pytest.raises(TypeError, match="bit_string must be a str"):
        Tag(bit_string)


@pytest.mark.parametrize("n", (7, 9, 11))
def test_larger_tag_parsing_requires_crc_validation_to_be_disabled(n):
    bit_string = "0" * (n * n * 2)

    with pytest.raises(
        ValueError, match="CRC validation is supported only for 5x5 tags"
    ):
        Tag(bit_string, n=n)

    tag = Tag(bit_string, n=n, validate_crc=False)
    assert tag.n == n
    assert tag.crc_valid is None


@pytest.mark.parametrize("n", (7, 9, 11))
def test_crc_helpers_reject_larger_sizes(n):
    with pytest.raises(
        ValueError, match="CRC validation is supported only for 5x5 tags"
    ):
        valid_crc("0" * (n * n * 2), n=n)


@pytest.mark.parametrize("message", ("", "0" * 23, "0" * 25, "0" * 23 + "x"))
def test_generator_rejects_malformed_messages(message):
    with pytest.raises(ValueError):
        Tag.from_message(message)


@pytest.mark.parametrize("message", (None, 123, b"0" * 24))
def test_generator_rejects_nonstring_messages(message):
    with pytest.raises(TypeError, match="message must be a str"):
        Tag.from_message(message)


def test_crc_input_helper_rejects_wrong_cell_count():
    with pytest.raises(ValueError, match="exactly 25 cells"):
        _crc_input_bytes_5x5(["00"] * 24)


def test_crc_input_helper_rejects_malformed_selected_cell():
    cells = ["00"] * 25
    cells[CRC_INPUT_INDICES_5X5[0]] = "0"

    with pytest.raises(ValueError, match="two-bit binary string"):
        _crc_input_bytes_5x5(cells)


def test_crc_input_helper_does_not_read_crc_cells_during_generation():
    cells = ["00"] * 25
    for index in get_crc_inds(5):
        cells[index] = None

    assert len(_crc_input_bytes_5x5(cells)) == 4


@pytest.mark.parametrize("n", (4, 6, 13, True, "5"))
def test_tag_rejects_unknown_sizes(n):
    with pytest.raises(ValueError, match="n must be one of"):
        Tag(MELBOURNE_FULL_TAG_BITS, n=n)


@pytest.mark.parametrize("validate_crc", (None, 0, 1, "yes"))
def test_tag_requires_boolean_validate_crc(validate_crc):
    with pytest.raises(TypeError, match="validate_crc must be a bool"):
        Tag(MELBOURNE_FULL_TAG_BITS, validate_crc=validate_crc)
