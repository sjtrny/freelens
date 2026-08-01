"""The affine matrix can be re-derived from (message, crc) pairs.

Exercises scripts/derive_affine_checksum.py: pushing random messages through the
shipped map and solving the GF(2) system back out must recover A_f / b_f exactly.
numpy-only.
"""

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import realcrc
import derive_affine_checksum as d


def _synthetic_pairs(n, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        msg = "".join(rng.integers(0, 2, 24).astype(str))
        pairs.append((msg, realcrc.real_crc_bit_string(msg)))
    return pairs


def test_derive_recovers_shipped_matrix():
    A, b, unique, consistent = d.derive(_synthetic_pairs(40, seed=1))
    assert unique and consistent
    assert np.array_equal(A, realcrc.A_f)
    assert np.array_equal(b, realcrc.b_f)


def test_derive_is_underdetermined_with_too_few_messages():
    # fewer than 25 independent messages cannot pin the 24 columns + constant
    A, b, unique, consistent = d.derive(_synthetic_pairs(5, seed=2))
    assert consistent
    assert not unique


def test_gf2_solve_round_trip():
    rng = np.random.default_rng(3)
    M = rng.integers(0, 2, (30, 25)).astype(np.uint8)
    x_true = rng.integers(0, 2, 25).astype(np.uint8)
    t = (M.dot(x_true)) % 2
    x, rank, consistent = d.gf2_solve(M, t)
    assert consistent
    assert np.array_equal((M.dot(x)) % 2, t)
