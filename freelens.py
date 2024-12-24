import math

import cv2 as cv
import numpy as np
from crc import Calculator, Configuration
from PIL import Image

ind_bit_map = {
    0: "00",
    1: "01",
    2: "10",
    3: "11",
}

center_bit_map = {5: "00", 7: "01", 9: "10", 11: "11"}

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

    return np.array(
        [
            tl,
            right_points[right_y_sort_idx[0], :],
            right_points[right_y_sort_idx[1], :],
            bl,
        ]
    ).astype(np.float32)


def reduce_poly_vertices(contours, tolerance=0.1):
    polygons = []
    for c in contours:
        perimeter = cv.arcLength(c, True)
        polygon = np.squeeze(cv.approxPolyDP(c, tolerance * perimeter, True))
        polygons.append(polygon)
    return polygons


def frame_filter_polygons_4vertex(polygons):
    filtered_polygons = []
    for i, p in enumerate(polygons):
        if len(p) == 4:
            filtered_polygons.append(p)

    return filtered_polygons


def frame_filter_polygons_area(polygons, area_threshold=4000):
    filtered_polygons = []
    for i, p in enumerate(polygons):
        if cv.contourArea(p, True) >= area_threshold:
            filtered_polygons.append(p)

    return filtered_polygons


def frame_filter_polygons_convex(polygons):
    filtered_polygons = []
    for i, p in enumerate(polygons):
        if cv.isContourConvex(p):
            filtered_polygons.append(p)

    return filtered_polygons


def frame_filter_polygons_squareish(polygons):
    filtered_polygons = []
    for i, p in enumerate(polygons):
        area = cv.contourArea(p, True)
        perimeter = cv.arcLength(p, True)
        if 1 > np.abs(perimeter / area) > 0:
            filtered_polygons.append(p)

    return filtered_polygons


def expand_polygon(polygon, scale_factor=1 + 4 / 14):
    # Calculate the centroid of the polygon
    centroid = np.mean(polygon, axis=0)

    # Scale each point relative to the centroid
    expanded_polygon = centroid + (polygon - centroid) * scale_factor

    # Convert back to integers for pixel coordinates
    return np.round(expanded_polygon).astype(int)


def frame_filter_white_border(polygons, image_bw):
    filtered_polygons = []

    laplacian = cv.Laplacian(image_bw, cv.CV_64F)

    for i, p in enumerate(polygons):
        p_expanded = expand_polygon(p)

        outer_mask = np.zeros(image_bw.shape).astype(np.uint8)
        cv.fillConvexPoly(outer_mask, p_expanded, color=255)

        inner_mask = np.zeros(image_bw.shape).astype(np.uint8)
        cv.fillConvexPoly(inner_mask, p, color=255)

        mask = outer_mask - inner_mask

        masked_laplacian_inner = laplacian * inner_mask.astype(bool)
        masked_laplacian_inner_values = masked_laplacian_inner[inner_mask > 0]
        inner_laplacian_variance = np.var(masked_laplacian_inner_values)
        masked_laplacian_mask = laplacian * mask.astype(bool)
        masked_laplacian_mask_values = masked_laplacian_mask[mask > 0]
        laplacian_variance = np.var(masked_laplacian_mask_values)

        masked_pixels = image_bw[inner_mask > 0]
        inner_percentile = np.percentile(masked_pixels, 90)
        masked_pixels = image_bw[mask > 0]
        border_median = np.median(masked_pixels)

        if (
            laplacian_variance <= inner_laplacian_variance
            and border_median >= inner_percentile
        ):
            filtered_polygons.append(p)

    return filtered_polygons


def detect_frames(image):
    """
    Based on "Automatic generation and detection of highly reliable fiducial markers under occlusion" Pattern Recognition 2014

    1. Convert image to grayscale
    2. Detect edges by local adaptive thresholding (cv.adaptiveThreshold)
    3. Detect contours by Suzuki's method (cv.findContours)
    4. Fit polygon to contours (cv.approxPolyDP)
    5. Apply filters:
        1. 4-vertex polygons.
        2. Area greater than threshold
        3. Convex polygon
        4. Shape is roughly square (perimeter/area test)
        5. Check that border around frame is white

    TODO: Retain only internal contours (opposite of paper which suggests external)
    """

    image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # 1. Convert image to grayscale
    image_bw_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2GRAY)

    # 2. Detect edges by local adaptive thresholding (cv.adaptiveThreshold)
    threshold_image = cv.adaptiveThreshold(
        image_bw_cv, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 101, 0
    )

    # 3. Detect contours by Suzuki's method (cv.findContours)
    contours, hierarchy = cv.findContours(
        threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    # 4. Fit polygon to contours (cv.approxPolyDP)
    polygons = reduce_poly_vertices(contours)

    # 5. Apply filters
    filters = [
        frame_filter_polygons_4vertex,
        lambda polygons: frame_filter_polygons_area(polygons, 2**11),
        frame_filter_polygons_convex,
        frame_filter_polygons_squareish,
        lambda polygons: frame_filter_white_border(polygons, image_bw_cv),
    ]
    for filter in filters:
        polygons = filter(polygons)

    return polygons


def decode_frames(image, polygons, n=5, validate_crc=False):

    image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2Lab)

    cell_size = 32
    n_pixels = cell_size * (n + 2)
    ref_pts = np.float32([[0, 0], [n_pixels, 0], [n_pixels, n_pixels], [0, n_pixels]])

    tags = []

    for i, polygon in enumerate(polygons):

        polygon_ordered = order_points(polygon)

        M = cv.getPerspectiveTransform(polygon_ordered, ref_pts)
        dst = cv.warpPerspective(image_cv, M, (n_pixels, n_pixels))

        values = np.zeros((n + 2, n + 2, 3))

        for r in range(n + 2):  # Rows
            for c in range(n + 2):  # Cols
                center_x = r * cell_size + cell_size // 2
                center_y = c * cell_size + cell_size // 2
                values[r, c] = dst[center_x, center_y]

        # Get corner colours
        corner_vals = np.array(
            [values[1, 1], values[1, -2], values[-2, -2], values[-2, 1]]
        )

        # Find argmin of each cell to corner colours
        code = []
        for r in range(1, n + 1):  # Rows
            for c in range(1, n + 1):  # Cols
                ind = np.argmin(np.mean((corner_vals - values[r, c]) ** 2, axis=1))
                code.append(int(ind))

        bit_string = "".join([ind_bit_map[ind] for ind in code])

        tag = Tag(bit_string, n=n)
        tags.append(tag)

    return tags


def detect_tags(img, n=5, validate_crc=False):
    frames = detect_frames(img)

    tags = decode_frames(img, frames, n=n, validate_crc=validate_crc)

    return tags


def message_length_for_N(N):
    """
    Returns the number of bits available to store the message for a tag of size N
    """
    total = N**2
    rows = int(math.floor(N / 2))
    crc_length = rows * 4

    # total - corners - middle - CRC
    return (total - 4 - 1 - crc_length) * 2


def max_int_for_N(N):
    message_n_bits = message_length_for_N(N)
    return 2 ** (message_n_bits)


def get_corner_indices_1d(n):
    return [0, n - 1, n**2 - 1, n**2 - n]


def get_center_ind(n):
    return int(math.floor((n**2) / 2))


def get_crc_inds(n):
    indices = np.reshape(np.arange(n**2), (n, n))

    center_ind = int(np.floor(n / 2))

    flat_indices = np.zeros((center_ind * 4,), dtype=np.uint32)

    # Left row
    flat_indices[0:center_ind] = indices[center_ind, 0:center_ind]
    # Top column
    flat_indices[center_ind : 2 * center_ind] = indices[0:center_ind, center_ind]
    # Bottom column
    flat_indices[2 * center_ind : 3 * center_ind] = indices[
        center_ind + 1 :, center_ind
    ]
    # Right row
    flat_indices[3 * center_ind : 4 * center_ind] = indices[
        center_ind, center_ind + 1 :
    ]

    return list(flat_indices)


def get_message_inds(n):
    indices = np.reshape(np.arange(n**2), (n, n))

    center_ind = int(np.floor(n / 2))

    # Clear center row and col
    indices[center_ind, :] = -1
    indices[:, center_ind] = -1

    # Clear corners
    indices[0, 0] = -1
    indices[0, n - 1] = -1
    indices[n - 1, n - 1] = -1
    indices[n - 1, 0] = -1

    # Using fortran order to descend rows before columns to
    # match reading order of NaviLens provided tags
    indices_flat = np.ravel(indices, order="F")
    indices_flat = indices_flat[indices_flat >= 0]

    return list(indices_flat)


def valid_crc(bit_string, n=5):

    cells = [bit_string[i : i + 2] for i in range(0, n**2 * 2, 2)]

    crc_inds = get_crc_inds(n)
    crc_bit_string = "".join([cells[i] for i in crc_inds])

    message_inds = get_message_inds(n)
    message_bit_string = "".join([cells[i] for i in message_inds])

    crc_checksum = int(crc_bit_string, 2)

    message_int = int(message_bit_string, 2)
    message_bytes = message_int.to_bytes(
        int(len(message_bit_string) / 8), byteorder="little"
    )
    crc_calculator = Calculator(crc_config_map[n])
    crc_checksum_calculated = crc_calculator.checksum(message_bytes)

    if crc_checksum == crc_checksum_calculated:
        return True

    return False


class Tag:

    def __init__(self, bit_string, n=5):

        if set(bit_string) != {"0", "1"}:
            raise ValueError("bit_string must only contain '0' and '1'.")

        if len(bit_string) != (n**2 * 2):
            raise ValueError(
                f"bit_string has length {len(bit_string)}, expected length {n**2 * 2} since n={n}."
            )

        if n not in center_bit_map.keys():
            raise ValueError(f"n must be one of {center_bit_map.keys()}.")

        self.n = n
        self.bit_string = bit_string
        self.valid = valid_crc(self.bit_string)

        self.cells = tuple(
            self.bit_string[i : i + 2] for i in range(0, self.n**2 * 2, 2)
        )
        self.message = "".join([self.cells[i] for i in get_message_inds(self.n)])
        self.crc = "".join([self.cells[i] for i in get_crc_inds(self.n)])

    @staticmethod
    def from_message(message_bit_string, n=5):

        n_bits_message = message_length_for_N(n)

        if len(message_bit_string) != n_bits_message:
            raise ValueError(
                f"message_bit_string has length {len(message_bit_string)}, expected length {n_bits_message} since n={n}"
            )

        cells = [None] * (n**2)

        # Set the MESSAGE cell values
        message_cells = [message_bit_string[i : i + 2] for i in range(0, n**2 * 2, 2)]
        for i, index in enumerate(get_message_inds(n)):
            cells[index] = message_cells[i]

        # Calculate the CRC
        message_bytes = int(message_bit_string, 2).to_bytes(int(n_bits_message / 8))

        crc_calculator = Calculator(crc_config_map[n])
        crc_int = crc_calculator.checksum(message_bytes)

        # Convert CRC checksum to binary and set cell values
        n_bits_crc = int(math.floor(n / 2)) * 4 * 2
        crc_binary_string = format(crc_int, f"0{n_bits_crc}b")
        crc_cells = [crc_binary_string[i : i + 2] for i in range(0, n_bits_crc, 2)]
        for i, index in enumerate(get_crc_inds(n)):
            cells[index] = crc_cells[i]

        # Set CORNER cell values
        corner_cells = ["00", "01", "10", "11"]
        for i, index in enumerate(get_corner_indices_1d(n)):
            cells[index] = corner_cells[i]

        # Set the CENTER cell value
        cells[get_center_ind(n)] = center_bit_map[n]

        tag_bit_string = "".join(cells)

        return Tag(bit_string=tag_bit_string, n=n)

    def to_image(
        self,
        cell_size=32,
        quiet_pad_size=64,
        palette=("cyan", "magenta", "yellow", "black"),
        inner_pad_colour="black",
        outer_pad_colour="white",
    ):
        """
        palette: a list of strings representing the colours used. Must be supported
        by ImageColor module of Pillow https://pillow.readthedocs.io/en/stable/reference/ImageColor.html
        """

        if len(set(palette)) != 4:
            raise ValueError("palette must contain 4 distinct elements.")

        colour_list = [palette[int(cell, 2)] for cell in self.cells]

        # Create cell grid
        wh = self.n * cell_size
        grid_img = Image.new("RGB", (wh, wh), inner_pad_colour)
        for i in range(len(colour_list)):
            cell = Image.new("RGB", (cell_size, cell_size), colour_list[i])
            grid_img.paste(
                cell, (i % self.n * cell_size, math.floor(i / self.n) * cell_size)
            )

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

    def __str__(self):
        return "".join(self.message)

    def __repr__(self):
        return f"Tag({self.bit_string}, {self.n})"
