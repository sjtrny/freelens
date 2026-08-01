"""Reproduce the deployed-checksum claims against known real tags.

Needs only numpy. Checks that the affine checksum in ``realcrc.py`` reproduces
the crc cells actually carried by two independent real tags:

    python scripts/reproduce_claims.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import realcrc

_BITS = {0: "00", 1: "01", 2: "10", 3: "11"}

# Full grids, row-major, C=0 M=1 Y=2 K=3.
TAGS = {
    "B1269C (printed label)": [
        [0, 0, 2, 1, 1],
        [2, 1, 1, 2, 3],
        [0, 3, 0, 1, 3],
        [3, 0, 2, 2, 0],
        [3, 2, 2, 1, 2],
    ],
    "issue #1 (Melbourne tram)": [
        [0, 2, 3, 0, 1],
        [1, 2, 3, 0, 3],
        [3, 3, 0, 0, 2],
        [0, 0, 3, 2, 0],
        [3, 0, 3, 0, 2],
    ],
}


def grid_to_bits(grid):
    return "".join(_BITS[v] for row in grid for v in row)


def message_bits(grid):
    cells = [grid_to_bits(grid)[i:i + 2] for i in range(0, 50, 2)]
    return "".join(cells[i] for i in realcrc.get_message_inds(5))


def main():
    ok = True
    for name, grid in TAGS.items():
        bits = grid_to_bits(grid)
        cells = [bits[i:i + 2] for i in range(0, 50, 2)]
        on_tag = "".join(cells[i] for i in realcrc.get_crc_inds(5))
        computed = realcrc.real_crc_bit_string(message_bits(grid))
        match = realcrc.valid_real_crc(bits)
        ok &= match
        print(f"{name}")
        print(f"  crc on tag : 0x{int(on_tag, 2):04x}")
        print(f"  affine crc : 0x{int(computed, 2):04x}")
        print(f"  MATCH      : {'PASS' if match else 'FAIL'}")
    print("\nALL:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
