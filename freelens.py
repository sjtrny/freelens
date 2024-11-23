import math

from crc import Calculator, Configuration
from PIL import Image, ImageColor

crc_config_map = {
    5: Configuration(
        width=16,
        polynomial=0xC867,
        init_value=0xFFFF,
        final_xor_value=0x0000,
        reverse_input=False,
        reverse_output=False,
    ),
    7: Configuration(
        width=24,
        polynomial=0x864CFB,
        init_value=0x000000,
        final_xor_value=0x000000,
        reverse_input=False,
        reverse_output=False,
    ),
    9: Configuration(
        width=32,
        polynomial=0x814141AB,
        init_value=0x00000000,
        final_xor_value=0x00000000,
        reverse_input=False,
        reverse_output=False,
    ),
    11: Configuration(
        width=40,
        polynomial=0x0004820009,
        init_value=0x0000000000,
        final_xor_value=0xFFFFFFFFFF,
        reverse_input=False,
        reverse_output=False,
    ),
}


def message_length_for_N(N):
    """
    Returns the number of cells available to store the message for a tag of size N
    """
    total = N**2
    rows = int(math.floor(N / 2))
    crc_length = rows * 4

    # total - corners - middle - CRC
    return total - 4 - 1 - crc_length


def max_int_for_N(N):
    message_len = message_length_for_N(N)
    return 2 ** (message_len * 2)


def get_corner_inds(n):
    return [0, n - 1, n**2 - 1, n**2 - n]


def get_center_ind(n):
    return int(math.floor((n**2) / 2))


def get_crc_inds(n):
    middle_col_x = int(math.floor(n / 2))
    rows = int(math.floor(n / 2))
    center_ind = get_center_ind(n)

    crc_inds = []
    # Set CRC vertical top
    crc_inds.extend(list(range(middle_col_x, middle_col_x + n * rows, n)))
    # Set CRC horizontal
    crc_inds.extend(list(range(rows * n, rows * n + rows, 1)))
    crc_inds.extend(list(range(center_ind + 1, center_ind + 1 + rows, 1)))
    # Set CRC vertical bottom
    crc_inds.extend(list(range(center_ind + n, n * n - middle_col_x, n)))

    return crc_inds


def get_message_inds(n):
    center_ind = get_center_ind(n)
    corner_inds = get_corner_inds(n)
    crc_inds = get_crc_inds(n)
    return set(range(n**2)) - set(crc_inds + [center_ind] + corner_inds)


def tag_colours(id_int, n=5, palette=["cyan", "magenta", "yellow", "black"]):
    """
    palette: a list of strings representing the colours used. Must be supported
    by ImageColor module of Pillow https://pillow.readthedocs.io/en/stable/reference/ImageColor.html
    """
    if id_int < 0 or id_int > max_int_for_N(n):
        raise ValueError(f"message_int must be >= 0 and <= {max_int_for_N(n)}.")

    if n not in [5, 7, 9, 11]:
        raise ValueError("N must be one of 5, 7, 9, 11.")

    if len(set(palette)) != 4:
        raise ValueError("palette must contain 4 distinct elements.")

    n_cells_message = message_length_for_N(n)
    n_bits_message = n_cells_message * 2
    n_bytes_message = int((n_cells_message * 2) / 8)

    try:
        message_bytes = id_int.to_bytes(n_bytes_message)
    except OverflowError:
        raise ValueError(f"message_int too large to be represented in {n}x{n} tag.")

    message_binary_string = format(id_int, f"0{n_bits_message}b")

    crc_calculator = Calculator(crc_config_map[n])
    crc_int = crc_calculator.checksum(message_bytes)
    # n_crc_cells_per_side * 4 sides * 2 bits per cell
    n_bits_crc = int(math.floor(n / 2)) * 4 * 2
    crc_binary_string = format(crc_int, f"0{n_bits_crc}b")

    message_colour_list = [
        palette[int(message_binary_string[i : i + 2], 2)]
        for i in range(0, n_bits_message, 2)
    ]
    crc_colour_list = [
        palette[int(crc_binary_string[i : i + 2], 2)] for i in range(0, n_bits_crc, 2)
    ]

    tag_colour_list = [None] * (n**2)

    # Set corner colours
    corner_inds = get_corner_inds(n)
    for i, ind in enumerate(corner_inds):
        tag_colour_list[ind] = palette[i]

    # Set center colour
    center_ind = get_center_ind(n)
    center_palette_map = {5: 0, 7: 1, 9: 2, 11: 3}
    tag_colour_list[center_ind] = palette[center_palette_map[n]]

    # Set CRC colours
    crc_inds = get_crc_inds(n)
    for i, ind in enumerate(crc_inds):
        tag_colour_list[ind] = crc_colour_list[i]

    message_inds = get_message_inds(n)
    for i, ind in enumerate(message_inds):
        tag_colour_list[ind] = message_colour_list[i]

    return tag_colour_list


def make_tag_image(
    colour_list,
    N=5,
    cell_size=32,
    quiet_pad_size=64,
    inner_pad_colour="black",
    outer_pad_colour="white",
):
    # Create cell grid
    wh = N * cell_size
    grid_img = Image.new("RGB", (wh, wh), inner_pad_colour)
    for i in range(len(colour_list)):
        cell = Image.new("RGB", (cell_size, cell_size), colour_list[i])
        grid_img.paste(cell, (i % N * cell_size, math.floor(i / N) * cell_size))

    # Create inner quiet zone and paste grid image inside
    black_code_img_wh = wh + cell_size * 2
    black_code_img = Image.new(
        "RGB", (black_code_img_wh, black_code_img_wh), inner_pad_colour
    )

    black_code_img.paste(grid_img, (cell_size, cell_size))

    # Create outer quiet zone and paste inner quiet zone inside
    padded_img_wh = black_code_img_wh + quiet_pad_size * 2
    padded_img = Image.new("RGB", (padded_img_wh, padded_img_wh), outer_pad_colour)

    padded_img.paste(black_code_img, (quiet_pad_size, quiet_pad_size))

    return padded_img
