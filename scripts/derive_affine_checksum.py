"""Derive (generate) the affine checksum matrix A_f / b_f from tag data.

The deployed 5x5 checksum is affine over GF(2):  crc = A @ message XOR b. Given
enough (message, crc) pairs read off real tags, A and b are recovered by solving
the linear system over GF(2) -- 25 independent messages pin the 24 columns plus
the constant. This is how the shipped matrix in ``realcrc.py`` is generated.

Run with no arguments for a self-contained proof: random messages are pushed
through the shipped map, the matrix is re-derived from those pairs alone, and the
result is checked against the shipped matrix.

    python scripts/derive_affine_checksum.py

Point it at a folder of real tag images to derive from actual tags (needs the CV
deps; uses freelens to decode each tag into a (message, crc) pair):

    python scripts/derive_affine_checksum.py path/to/tags
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import realcrc


def _bits(s):
    return np.fromiter((int(c) for c in s), dtype=np.uint8, count=len(s))


def gf2_solve(M, t):
    """Solve M x = t over GF(2). Return (x, rank, consistent) with free vars = 0."""
    A = np.concatenate([M % 2, (t % 2)[:, None]], axis=1).astype(np.uint8)
    m, ncols = A.shape
    ncoef = ncols - 1
    pivot_cols = []
    row = 0
    for col in range(ncoef):
        piv = next((rr for rr in range(row, m) if A[rr, col]), None)
        if piv is None:
            continue
        A[[row, piv]] = A[[piv, row]]
        for rr in range(m):
            if rr != row and A[rr, col]:
                A[rr] ^= A[row]
        pivot_cols.append(col)
        row += 1
        if row == m:
            break
    consistent = not any(
        (not A[rr, :ncoef].any()) and A[rr, ncoef] for rr in range(m))
    x = np.zeros(ncoef, dtype=np.uint8)
    for i, col in enumerate(pivot_cols):
        x[col] = A[i, ncoef]
    return x, len(pivot_cols), consistent


def derive(pairs):
    """pairs: list of (message_bit_string(24), crc_bit_string(16)).

    Returns (A, b, unique, consistent): A is 16x24, b is length 16.
    ``unique`` is True only when the messages pin every column (rank 25).
    """
    # augment each message with a constant 1 to absorb b
    M = np.array([np.append(_bits(msg), 1) for msg, _ in pairs], dtype=np.uint8)
    C = np.array([_bits(crc) for _, crc in pairs], dtype=np.uint8)
    A = np.zeros((16, 24), dtype=np.uint8)
    b = np.zeros(16, dtype=np.uint8)
    unique = consistent = True
    for r in range(16):
        x, rank, ok = gf2_solve(M, C[:, r])
        A[r], b[r] = x[:24], x[24]
        unique &= (rank == 25)
        consistent &= ok
    return A, b, unique, consistent


def pairs_from_dataset(folder):
    """Decode every tag image in ``folder`` into a (message, crc) pair."""
    import glob

    from PIL import Image

    import freelens

    pairs = []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = sorted(f for e in exts for f in glob.glob(os.path.join(folder, e)))
    for path in files:
        img = Image.open(path).convert("RGB")
        for tag in freelens.detect_tags(img, n=5, validate_crc=None):
            pairs.append((tag.message, tag.crc))
    return pairs, files


def _self_test():
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(40):
        msg = "".join(rng.integers(0, 2, 24).astype(str))
        crc = realcrc.real_crc_bit_string(msg)
        pairs.append((msg, crc))
    A, b, unique, consistent = derive(pairs)
    ok = (unique and consistent
          and np.array_equal(A, realcrc.A_f) and np.array_equal(b, realcrc.b_f))
    print("synthetic derivation from 40 random messages:")
    print(f"  unique={unique}  consistent={consistent}  "
          f"matches shipped A_f/b_f={np.array_equal(A, realcrc.A_f) and np.array_equal(b, realcrc.b_f)}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def _from_dataset(folder):
    pairs, files = pairs_from_dataset(folder)
    print(f"decoded {len(pairs)} tag(s) from {len(files)} image(s) in {folder}")
    if len(pairs) < 25:
        print("need >=25 independent messages to pin the matrix; got",
              len(pairs))
        return False
    A, b, unique, consistent = derive(pairs)
    print(f"  unique={unique}  consistent={consistent}")
    print(f"  matches shipped A_f/b_f="
          f"{np.array_equal(A, realcrc.A_f) and np.array_equal(b, realcrc.b_f)}")
    return consistent


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(0 if _from_dataset(sys.argv[1]) else 1)
    sys.exit(0 if _self_test() else 1)
