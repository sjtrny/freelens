import math

from crc import Calculator, Configuration
import cv2 as cv
import numpy as np
from PIL import Image, ImageColor
from scipy.spatial import distance as dist

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


def find_square_contour_idxs(contours, hierarchy):
    candidate_contours_idx = []
    for i, c in enumerate(contours):
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(approx)
            area = cv.contourArea(c)
            if area > 600:
                candidate_contours_idx.append(i)
                ar = w / float(h)
                if 0.85 < ar < 1.3:
                    candidate_contours_idx.append(i)

    return candidate_contours_idx


def get_inner_contour_idxs(countour_idxs, hierarchy):
    valid_contour_idxs = []
    for i, idx in enumerate(countour_idxs):
        if hierarchy[idx][2] not in countour_idxs:
            valid_contour_idxs.append(idx)

    return valid_contour_idxs



def order_points(points):
    """
    Sort a 2D array of points into tl, tr, br, bl order.
    """

    # Sort by horizontal position
    x_sort_idx = np.argsort(points[:, 0])

    # Find the left points and then sort by vertical positions
    left_points = points[x_sort_idx[:2], :]
    left_y_sort_idx = np.argsort(left_points[:, 1])
    tl = left_points[left_y_sort_idx[0], :]
    bl = left_points[left_y_sort_idx[1], :]

    # Calculate the distance from tl to right points to find bottom right
    right_points = points[x_sort_idx[2:], :]
    dists = np.linalg.norm(right_points - tl, axis=1)
    right_y_sort_idx = np.argsort(dists)

    return np.array([
        tl,
        right_points[right_y_sort_idx[0], :],
        right_points[right_y_sort_idx[1], :],
        bl,
    ]).astype(np.float32)

def detect_tags(image):
    # Image should be a PIL image format
    image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    image_bw_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2GRAY)

    # Binary threshold with cv2.threshold using Otsu's method
    ret, thresh = cv.threshold(image_bw_cv, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy)

    candidate_contours_idx = find_square_contour_idxs(contours, hierarchy)
    final_contour_idxs = get_inner_contour_idxs(candidate_contours_idx, hierarchy)

    # Mask border
    n_pixels = 224
    ref_pts = np.float32([[0, 0], [n_pixels, 0], [n_pixels, n_pixels], [0, n_pixels]])

    tags = []
    for i, contour_idx in enumerate(final_contour_idxs):
        # Reshape to square
        c = contours[contour_idx]
        peri = cv.arcLength(c, True)
        approx = np.squeeze(cv.approxPolyDP(c, 0.04 * peri, True))
        contour_pts = order_points(approx)
        M = cv.getPerspectiveTransform(contour_pts, ref_pts)
        dst = cv.warpPerspective(image_cv, M, (n_pixels, n_pixels))

        dst_gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        dst_gray = cv.normalize(dst_gray, None, 0, 255, cv.NORM_MINMAX)

        grid_mean = np.zeros((7, 7))
        grid_std = np.zeros((7, 7))

        for r in range(7): # Rows
            for c in range(7): # Cols
                cell = dst_gray[32*r:32*(r + 1), 32*c: 32*(c + 1)]
                grid_mean[r, c] = np.mean(cell)
                grid_std[r, c] = np.std(cell)

        # Bottom left, Top left, Top right, Bottom right
        corner_means = np.array([grid_mean[-2,1], grid_mean[1,1], grid_mean[1, -2], grid_mean[-2,-2]])
        sort_idxs = np.argsort(corner_means)
        sorted_corner_means = corner_means[sort_idxs]

        distinct_corners = True
        for i in range(len(sorted_corner_means)-1):
            if sorted_corner_means[i+1] - sorted_corner_means[i] < 20:
                distinct_corners = False
        if not distinct_corners:
            continue

        # Check bottom left is close enough to border/quiet zone
        # Only use three cells next to the bottom left corner because using the entire quiet
        # zone can be unreliable as the lighting can change across the tag image.
        corner_black = True
        corner_angle_mean = np.mean(np.concatenate((dst_gray[-64:, 0:32].ravel(), dst_gray[-32:, 32:64].ravel())))
        if np.abs(corner_angle_mean - grid_mean[-2,1]) > 25:
            corner_black = False
        if not corner_black:
            continue

        # Check that cell mean is close enough to corner colours
        cell_colours_valid = True
        for r in range(1, 5+1): # Rows
            for c in range(1, 5+1): # Cols
                if np.min(np.abs(sorted_corner_means - grid_mean[r, c])) > 30:
                    cell_colours_valid = False
        if not cell_colours_valid:
            continue

        tags.append(Image.fromarray(cv.cvtColor(dst, cv.COLOR_BGR2RGB)))

    return tags

def decode_tag_image(tag_image, n=5, palette=["cyan", "magenta", "yellow", "black"]):
    palette_rgb_list = [ImageColor.getrgb(c) for c in palette]

    palette_rgb_array = np.vstack(palette_rgb_list)

    cell_size = int(tag_image.width / (n + 2))

    code = []
    for i in range(n):
        for j in range(n):
            window_median = np.median(
                tag_image.crop(
                    (
                        cell_size + cell_size * j,  # Left
                        cell_size + cell_size * i,  # Top
                        cell_size + cell_size * (j + 1),  # Right
                        cell_size + cell_size * (i + 1),  # Bottom
                    )
                ),
                axis=(0, 1),
            )
            ind = np.argmin(np.mean((palette_rgb_array - window_median) ** 2, axis=1))
            code.append(int(ind))

    ind_bit_map = {
        0: "00",
        1: "01",
        2: "10",
        3: "11",
    }

    crc_inds = get_crc_inds(n)
    crc_list = []
    for i, ind in enumerate(crc_inds):
        crc_list.append(code[ind])

    message_inds = get_message_inds(n)
    id_list = []
    for i, ind in enumerate(message_inds):
        id_list.append(code[ind])

    crc_bit_string = "".join([ind_bit_map[ind] for ind in crc_list])
    id_bit_string = "".join([ind_bit_map[ind] for ind in id_list])

    crc_checksum = int(crc_bit_string, 2)
    id_int = int(id_bit_string, 2)

    id_bytes = id_int.to_bytes(int(len(id_bit_string) / 8))
    crc_calculator = Calculator(crc_config_map[n])
    crc_checksum_calculated = crc_calculator.checksum(id_bytes)

    # if crc_checksum != crc_checksum_calculated:
    #     raise ValueError("CRC checksum mismatch, possible data corruption.")

    return id_int
