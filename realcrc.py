"""
Real (deployed) ddTag checksum for the 5x5 grid.

The patent specifies a CRC-16-CDMA2000 for the 5x5 tag, and ``freelens.valid_crc``
implements it faithfully. That checksum does **not** match the tags NaviLens
actually deploys (see https://github.com/sjtrny/freelens/issues/1): for a real
Melbourne-tram tag the grid carries ``0xfff2`` while the documented CRC computes
``0x1ec4``.

This module implements the checksum the deployed 5x5 tags really carry. It is an
affine map over GF(2), expressed in freelens' own conventions (``ind_bit_map``
two bits MSB-first per cell, ``get_message_inds`` / ``get_crc_inds`` ordering)::

    crc_bits = (A_f @ message_bits) XOR b_f          (mod 2)

so it is a drop-in alongside the documented path. It reproduces real tags exactly
(verified against the printed-label tag ``B1269C`` and the issue #1 tram tag).

WHAT THIS IS
    A verified generator and checker for the deployed 5x5 checksum.

WHAT THIS IS NOT  (open TODO)
    A closed-form function. The map has been shown *not* to be any CRC-16 (an
    exhaustive sweep of all 32768 degree-16 generator polynomials finds no match)
    and not GF(4)-linear; present evidence points to a computer-searched [40,24]
    distance code. No prettier closed form is known yet -- the matrix below *is*
    the canonical description until one is found.

SCOPE
    Only n=5 is solved. The 7x7 / 9x9 / 11x11 deployed checksums are not known
    (deriving them needs real tags of those sizes); for those sizes use the
    documented ``valid_crc``.

See CHECKSUM.md for the documented-vs-deployed write-up, the verified tags, and
provenance.
"""

import numpy as np

# Cell value <-> two bits, MSB-first. Mirrors ``freelens.ind_bit_map`` (kept here
# so this module imports with numpy alone, without the detector's CV deps).
ind_bit_map = {0: "00", 1: "01", 2: "10", 3: "11"}

# Affine GF(2) map for the deployed 5x5 checksum, in freelens coordinates.
#   rows  = crc bits, in get_crc_inds(5) order, two bits MSB-first per cell
#   cols  = message bits, in get_message_inds(5) order, two bits MSB-first per cell
_A_f = """
1 0 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 1 1 1 1 1 1 1
1 1 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0
0 1 0 0 1 1 1 0 0 1 1 0 1 0 1 1 0 1 1 1 1 0 1 1
0 0 1 0 0 1 1 1 0 0 1 1 0 1 0 1 1 0 1 1 1 1 0 1
1 0 0 1 0 0 1 1 1 0 0 1 1 0 1 0 1 1 0 1 0 1 1 0
0 1 1 1 0 0 1 0 1 1 1 1 0 1 0 0 1 0 0 1 0 1 0 0
1 0 1 1 1 0 0 1 0 1 1 1 1 0 1 0 0 1 0 0 1 0 1 0
1 1 0 1 1 1 0 0 1 0 1 1 1 1 0 1 0 0 1 0 1 1 0 1
1 1 1 0 1 1 1 0 0 1 0 1 1 1 1 0 1 0 0 1 1 1 1 0
0 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 0 1 0 0 0 1 1 1
1 0 0 0 1 1 0 0 1 0 1 0 1 1 1 0 0 1 0 1 1 1 0 0
0 1 1 1 1 1 0 1 0 1 1 0 1 1 1 0 1 1 0 1 0 0 0 1
0 0 1 1 0 1 1 0 1 0 1 1 0 1 1 1 0 1 1 0 0 0 0 0
0 0 0 1 0 0 1 1 0 1 0 1 1 0 1 1 1 0 1 1 1 0 0 0
1 0 1 1 1 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 1 1
0 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 1 1 1 0 1 1 1 0
"""
A_f = np.array([[int(c) for c in r.split()] for r in _A_f.strip().splitlines()],
               dtype=np.uint8)
b_f = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.uint8)


# --- cell index helpers (mirror the same-named functions in freelens.py) -------

def get_message_inds(n=5):
    indices = np.reshape(np.arange(n ** 2), (n, n))
    center = n // 2
    indices[center, :] = -1
    indices[:, center] = -1
    indices[0, 0] = indices[0, n - 1] = indices[n - 1, n - 1] = indices[n - 1, 0] = -1
    flat = np.ravel(indices, order="F")
    return list(flat[flat >= 0])


def get_crc_inds(n=5):
    indices = np.reshape(np.arange(n ** 2), (n, n))
    center = n // 2
    out = np.zeros(center * 4, dtype=int)
    out[0:center] = indices[center, 0:center]
    out[center:2 * center] = indices[0:center, center]
    out[2 * center:3 * center] = indices[center + 1:, center]
    out[3 * center:4 * center] = indices[center, center + 1:]
    return list(out)


def get_corner_inds(n=5):
    return [0, n - 1, n ** 2 - 1, n ** 2 - n]


def get_center_ind(n=5):
    return (n ** 2) // 2


# --- public API ----------------------------------------------------------------

def real_crc_bits(message_bit_string):
    """24-char message bit string (get_message_inds order) -> 16-bit np.array."""
    if len(message_bit_string) != 24:
        raise ValueError(
            f"expected 24 message bits, got {len(message_bit_string)}")
    m = np.fromiter((int(c) for c in message_bit_string), dtype=np.uint8, count=24)
    return (A_f.dot(m) + b_f) % 2


def real_crc_bit_string(message_bit_string):
    """24-char message bit string -> 16-char crc bit string (get_crc_inds order)."""
    return "".join(map(str, real_crc_bits(message_bit_string)))


def real_crc_cells(message_bit_string):
    """24-char message bit string -> 8 crc cells ('00'..'11'), get_crc_inds order."""
    bits = real_crc_bit_string(message_bit_string)
    return [bits[i:i + 2] for i in range(0, 16, 2)]


def valid_real_crc(bit_string, n=5):
    """True iff the tag's crc cells match the deployed checksum of its message.

    ``bit_string`` is the full n*n*2 grid bit string (row-major cells, two bits
    MSB-first), the same representation ``freelens.Tag`` uses.
    """
    if n != 5:
        raise NotImplementedError(
            "deployed checksum is only known for n=5; use valid_crc for larger n")
    cells = [bit_string[i:i + 2] for i in range(0, n ** 2 * 2, 2)]
    message_bits = "".join(cells[i] for i in get_message_inds(n))
    crc_bits = "".join(cells[i] for i in get_crc_inds(n))
    return real_crc_bit_string(message_bits) == crc_bits


def apply_real_crc(message_bit_string, n=5):
    """Build a full grid bit string carrying the deployed checksum for ``message``.

    Places the message cells, the deployed crc cells, the four orienting corners
    ('00','01','10','11') and the 5x5 center ('00'); returns an n*n*2 bit string
    suitable for ``freelens.Tag(bit_string, n)``.
    """
    if n != 5:
        raise NotImplementedError(
            "deployed checksum is only known for n=5; use Tag.from_message for larger n")
    message_cells = [message_bit_string[i:i + 2] for i in range(0, 24, 2)]
    cells = [None] * (n ** 2)
    for value, index in zip(message_cells, get_message_inds(n)):
        cells[index] = value
    for value, index in zip(real_crc_cells(message_bit_string), get_crc_inds(n)):
        cells[index] = value
    for value, index in zip(["00", "01", "10", "11"], get_corner_inds(n)):
        cells[index] = value
    cells[get_center_ind(n)] = "00"
    return "".join(cells)
