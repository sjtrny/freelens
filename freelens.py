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

SUPPORTED_TAG_SIZES = (5, 7, 9, 11)
MIN_FRAME_AREA = 1500

CRC_TAG_SIZE = 5
CRC_WIDTH = 16
CRC_POLYNOMIAL = 0xC867
CRC_INITIAL_VALUE = 0x0000
CRC_FINAL_XOR = 0x0000
CRC_REVERSE_INPUT = False
CRC_REVERSE_OUTPUT = False

# Column-major traversal of every cell outside the center row and column.
# The four palette corners are deliberately included.
CRC_INPUT_INDICES_5X5 = (
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

# The widths and polynomials are size-specific. All other generation rules are
# extrapolated from deployed 5x5 tags.
_CRC_POLYNOMIALS = {
    5: CRC_POLYNOMIAL,
    7: 0x864CFB,
    9: 0x814141AB,
    11: 0x0004820009,
}


def _validate_n(n):
    if isinstance(n, bool) or not isinstance(n, int) or n not in SUPPORTED_TAG_SIZES:
        raise ValueError(f"n must be one of {SUPPORTED_TAG_SIZES}")


def _validate_crc_options(n, validate_crc, require_valid_crc=False):
    _validate_n(n)

    if not isinstance(validate_crc, bool):
        raise TypeError("validate_crc must be a bool")
    if not isinstance(require_valid_crc, bool):
        raise TypeError("require_valid_crc must be a bool")
    if require_valid_crc and not validate_crc:
        raise ValueError("require_valid_crc=True requires validate_crc=True")
    if validate_crc and n != CRC_TAG_SIZE:
        raise ValueError("CRC validation is supported only for 5x5 tags")


def contour_filter_candidates(contours, area_threshold=4000):
    """Drop contours that cannot survive the frame filters, before polygon fitting.

    Adaptive thresholding of a photograph yields tens of thousands of contours,
    almost all of them specks. Fitting a polygon to each one is the single most
    expensive step in detection, so reject the hopeless ones with a bounding box
    first.

    This cannot discard a frame that would otherwise be found. cv.approxPolyDP
    returns a subset of the points it is given, so the fitted polygon lies inside
    the contour's bounding box and cannot have a larger area, nor more vertices
    than the contour it came from.
    """
    filtered_contours = []
    for c in contours:
        if len(c) < 4:
            continue
        _, _, width, height = cv.boundingRect(c)
        if width * height >= area_threshold:
            filtered_contours.append(c)

    return filtered_contours


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


def _quiet_zone_references(image, valid_pixels, cell_size):
    """Return black and white RGB references from a rectified quiet zone."""
    white_mask = np.ones(image.shape[:2], dtype=bool)
    white_mask[cell_size:-cell_size, cell_size:-cell_size] = False

    black_mask = np.zeros(image.shape[:2], dtype=bool)
    black_mask[cell_size:-cell_size, cell_size:-cell_size] = True
    black_mask[2 * cell_size : -2 * cell_size, 2 * cell_size : -2 * cell_size] = False

    black_pixels = image[black_mask & valid_pixels]
    white_pixels = image[white_mask & valid_pixels]
    if not len(black_pixels) or not len(white_pixels):
        return np.zeros(3), np.full(3, 255)

    return np.median(black_pixels, axis=0), np.median(white_pixels, axis=0)


def _correct_colours(image, black, white):
    """Map measured black and white RGB references onto the full RGB range."""
    image = image.astype(np.float32)
    black = np.asarray(black, dtype=np.float32)
    white = np.asarray(white, dtype=np.float32)
    difference = white - black

    corrected = image / 255
    usable_channels = difference > 0
    corrected[..., usable_channels] = (
        image[..., usable_channels] - black[usable_channels]
    ) / difference[usable_channels]
    return np.clip(corrected, 0, 1)


def _rectify_frame(image_rgba, polygon, n, cell_size):
    """Rectify a frame and its quiet zone, returning the frame and RGB references."""
    quiet_zone_pixels = cell_size * (n + 4)
    frame_start = cell_size
    frame_stop = cell_size * (n + 3)
    reference_points = np.float32(
        [
            [frame_start, frame_start],
            [frame_stop, frame_start],
            [frame_stop, frame_stop],
            [frame_start, frame_stop],
        ]
    )
    transform = cv.getPerspectiveTransform(polygon, reference_points)
    rectified = cv.warpPerspective(
        image_rgba,
        transform,
        (quiet_zone_pixels, quiet_zone_pixels),
    )
    valid_pixels = rectified[..., 3] == 255
    black, white = _quiet_zone_references(rectified[..., :3], valid_pixels, cell_size)
    frame = rectified[
        frame_start:frame_stop,
        frame_start:frame_stop,
        :3,
    ]
    return frame, black, white


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
    4. Discard contours too small or too sparse to become a frame
    5. Fit polygon to contours (cv.approxPolyDP)
    6. Apply filters:
        1. 4-vertex polygons.
        2. Area of at least MIN_FRAME_AREA
        3. Convex polygon
        4. Shape is roughly square (perimeter/area test)

    The quiet zone is not used to reject frame candidates.

    TODO: Retain only internal contours (opposite of paper which suggests external)
    """

    # 1. Convert image to grayscale
    image_bw_cv = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2GRAY)

    # 2. Detect edges by local adaptive thresholding (cv.adaptiveThreshold)
    threshold_image = cv.adaptiveThreshold(
        image_bw_cv, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 101, 0
    )

    # 3. Detect contours by Suzuki's method (cv.findContours)
    contours, hierarchy = cv.findContours(
        threshold_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )

    # 4. Discard contours too small or too sparse to become a frame
    candidates = contour_filter_candidates(contours, MIN_FRAME_AREA)

    # 5. Fit polygon to contours (cv.approxPolyDP)
    polygons = reduce_poly_vertices(candidates)

    # 6. Apply filters
    filters = [
        frame_filter_polygons_4vertex,
        lambda polygons: frame_filter_polygons_area(polygons, MIN_FRAME_AREA),
        frame_filter_polygons_convex,
        frame_filter_polygons_squareish,
    ]
    for filter in filters:
        polygons = filter(polygons)

    return polygons


def decode_frames(
    image,
    polygons,
    n=5,
    *,
    validate_crc=True,
    require_valid_crc=False,
):
    """Decode four-vertex frame polygons supplied in cyclic contour order."""
    _validate_crc_options(n, validate_crc, require_valid_crc)

    image_rgba = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2RGBA)

    cell_size = 32

    tags = []

    for i, polygon in enumerate(polygons):
        # Contour approximation already returns adjacent vertices in cyclic order.
        # Its starting corner only rotates the warp, which is normalised below.
        polygon_points = np.asarray(polygon, dtype=np.float32)
        frame_rgb, black, white = _rectify_frame(
            image_rgba, polygon_points, n, cell_size
        )
        corrected_rgb = np.round(
            _correct_colours(frame_rgb, black, white) * 255
        ).astype(np.uint8)
        corrected_lab = cv.cvtColor(corrected_rgb, cv.COLOR_RGB2Lab)
        tag = _decode_rectified_frame(corrected_lab, n, validate_crc)

        if not require_valid_crc or _strictly_valid_tag(tag):
            tags.append(tag)

    return tags


def _decode_rectified_frame(frame_lab, n, validate_crc):
    """Sample and decode one perspective-corrected CIELab frame."""
    cell_size = 32
    values = np.zeros((n + 2, n + 2, 3))

    for row in range(n + 2):
        for column in range(n + 2):
            y = row * cell_size + cell_size // 2
            x = column * cell_size + cell_size // 2
            values[row, column] = frame_lab[y, x]

    corners = np.array([values[1, 1], values[1, -2], values[-2, -2], values[-2, 1]])
    darkest_corner = int(np.argmin(corners[:, 0]))
    values = np.rot90(values, k=(darkest_corner + 1) % 4)
    corners = np.array([values[1, 1], values[1, -2], values[-2, -2], values[-2, 1]])

    code = []
    for row in range(1, n + 1):
        for column in range(1, n + 1):
            distances = np.mean((corners - values[row, column]) ** 2, axis=1)
            code.append(int(np.argmin(distances)))

    bit_string = "".join(ind_bit_map[index] for index in code)
    return Tag(bit_string, n=n, validate_crc=validate_crc)


def _strictly_valid_tag(tag):
    return tag.crc_valid is True and tag.corners_valid is True


def detect_tags(
    img,
    n=5,
    *,
    validate_crc=True,
    require_valid_crc=False,
):
    _validate_crc_options(n, validate_crc, require_valid_crc)
    frames = detect_frames(img)

    tags = decode_frames(
        img,
        frames,
        n=n,
        validate_crc=validate_crc,
        require_valid_crc=require_valid_crc,
    )

    return tags


def message_length_for_N(N):
    """
    Returns the number of bits available to store the message for a tag of size N
    """
    _validate_n(N)
    total = N**2
    rows = int(math.floor(N / 2))
    crc_length = rows * 4

    # total - corners - middle - CRC
    return (total - 4 - 1 - crc_length) * 2


def max_int_for_N(N):
    message_n_bits = message_length_for_N(N)
    return (2**message_n_bits) - 1


def get_corner_indices_1d(n):
    _validate_n(n)
    return [0, n - 1, n**2 - 1, n**2 - n]


def get_center_ind(n):
    _validate_n(n)
    return int(math.floor((n**2) / 2))


def get_crc_inds(n):
    _validate_n(n)
    center = n // 2

    return (
        [center * n + column for column in range(center)]
        + [row * n + center for row in range(center)]
        + [row * n + center for row in range(center + 1, n)]
        + [center * n + column for column in range(center + 1, n)]
    )


def get_crc_input_inds(n):
    """Return non-cross cells in the order used as CRC input."""
    _validate_n(n)
    center = n // 2

    return [
        row * n + column
        for column in range(n)
        for row in range(n)
        if row != center and column != center
    ]


def get_message_inds(n):
    _validate_n(n)
    center = n // 2
    corners = set(get_corner_indices_1d(n))

    return [
        row * n + column
        for column in range(n)
        for row in range(n)
        if row != center and column != center and row * n + column not in corners
    ]


def _validate_tag_bits(bit_string, n):
    if not isinstance(bit_string, str):
        raise TypeError("bit_string must be a str")

    expected_length = n**2 * 2
    if len(bit_string) != expected_length:
        raise ValueError(
            f"{n}x{n} tags must contain exactly {expected_length} bits; "
            f"got {len(bit_string)}"
        )
    if set(bit_string) - {"0", "1"}:
        raise ValueError("bit_string must contain only '0' and '1'")

    return bit_string


def _validate_message_bits(message, n):
    if not isinstance(message, str):
        raise TypeError("message must be a str")

    expected_length = message_length_for_N(n)
    if len(message) != expected_length:
        raise ValueError(
            f"{n}x{n} messages must contain exactly {expected_length} bits; "
            f"got {len(message)}"
        )
    if set(message) - {"0", "1"}:
        raise ValueError("message must contain only '0' and '1'")

    return message


def _crc_input_bytes(cells, n):
    _validate_n(n)

    try:
        cell_count = len(cells)
    except TypeError as error:
        raise TypeError("cells must be a sequence of two-bit strings") from error

    expected_cell_count = n**2
    if cell_count != expected_cell_count:
        raise ValueError(
            f"{n}x{n} tags must contain exactly {expected_cell_count} cells"
        )

    selected = [cells[index] for index in get_crc_input_inds(n)]
    if any(cell not in {"00", "01", "10", "11"} for cell in selected):
        raise ValueError("Every CRC input cell must be a two-bit binary string")

    bits = "".join(selected)
    expected_bit_count = 2 * (n - 1) ** 2
    if len(bits) != expected_bit_count:
        raise ValueError(
            f"{n}x{n} CRC input must contain exactly {expected_bit_count} bits"
        )
    if len(bits) % 8:
        raise ValueError("CRC input must be byte-aligned")

    return bytes(int(bits[offset : offset + 8], 2) for offset in range(0, len(bits), 8))


def _crc_input_bytes_5x5(cells):
    return _crc_input_bytes(cells, CRC_TAG_SIZE)


def _compute_crc(cells, n):
    """Calculate a CRC using the generation rules extrapolated from 5x5 tags."""
    _validate_n(n)
    configuration = Configuration(
        width=4 * n - 4,
        polynomial=_CRC_POLYNOMIALS[n],
        init_value=CRC_INITIAL_VALUE,
        final_xor_value=CRC_FINAL_XOR,
        reverse_input=CRC_REVERSE_INPUT,
        reverse_output=CRC_REVERSE_OUTPUT,
    )
    return Calculator(configuration).checksum(_crc_input_bytes(cells, n))


def compute_crc_5x5(cells):
    """Return the deployed CRC for a complete set of 5x5 tag cells."""
    return _compute_crc(cells, CRC_TAG_SIZE)


def valid_crc(bit_string, n=5):
    """Validate the deployed CRC carried by a 5x5 NaviLens tag."""
    _validate_n(n)
    if n != CRC_TAG_SIZE:
        raise ValueError("CRC validation is supported only for 5x5 tags")

    bit_string = _validate_tag_bits(bit_string, n)
    cells = tuple(
        bit_string[offset : offset + 2] for offset in range(0, len(bit_string), 2)
    )
    expected_crc = int("".join(cells[index] for index in get_crc_inds(n)), 2)
    return compute_crc_5x5(cells) == expected_crc


class Tag:

    def __init__(self, bit_string, n=5, *, validate_crc=True):
        _validate_n(n)
        if not isinstance(validate_crc, bool):
            raise TypeError("validate_crc must be a bool")

        self.n = n
        self.bit_string = _validate_tag_bits(bit_string, n)
        self.cells = tuple(
            self.bit_string[offset : offset + 2]
            for offset in range(0, len(self.bit_string), 2)
        )
        self.message = "".join(self.cells[index] for index in get_message_inds(n))
        self.crc = "".join(self.cells[index] for index in get_crc_inds(n))
        self.center_valid = self.cells[get_center_ind(n)] == center_bit_map[n]
        self.corners_valid = tuple(
            self.cells[index] for index in get_corner_indices_1d(n)
        ) == ("00", "01", "10", "11")

        if not validate_crc:
            self.crc_valid = None
        elif n != CRC_TAG_SIZE:
            raise ValueError("CRC validation is supported only for 5x5 tags")
        else:
            self.crc_valid = valid_crc(self.bit_string, n=n)

    @property
    def valid(self):
        """Deprecated alias for :attr:`crc_valid`."""
        return self.crc_valid

    @classmethod
    def from_message(cls, message, n=5):
        _validate_n(n)
        message = _validate_message_bits(message, n)

        cells = [None] * (n**2)

        message_cells = [
            message[offset : offset + 2] for offset in range(0, len(message), 2)
        ]
        for index, cell in zip(get_message_inds(n), message_cells):
            cells[index] = cell

        for index, cell in zip(get_corner_indices_1d(n), ("00", "01", "10", "11")):
            cells[index] = cell
        cells[get_center_ind(n)] = center_bit_map[n]

        crc = _compute_crc(cells, n)

        crc_width = 4 * n - 4
        crc_bits = f"{crc:0{crc_width}b}"
        crc_cells = [crc_bits[offset : offset + 2] for offset in range(0, crc_width, 2)]
        for index, cell in zip(get_crc_inds(n), crc_cells):
            cells[index] = cell

        return cls(
            bit_string="".join(cells),
            n=n,
            validate_crc=n == CRC_TAG_SIZE,
        )

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
