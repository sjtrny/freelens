"""Validate a folder of NaviLens code PDFs (or images) against the affine checksum.

For each file the tag is rendered (PDFs via PyMuPDF), decoded, and checked. If the
filename embeds the 6-hex code NaviLens assigns (e.g. ``Exit_AAB00D_210mm.pdf``),
the decoded payload is cross-checked against it -- a ground-truth check that is
independent of the checksum -- and that tag is used for an optional re-derivation
of the affine matrix.

    python scripts/validate_freekit.py path/to/kit
    python scripts/validate_freekit.py                 # dataset/freekit

Needs the CV deps (cv2, crc); PDFs additionally need pymupdf (fitz).
"""

import glob
import io
import os
import re
import sys

import numpy as np
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import freelens
import realcrc
import derive_affine_checksum as d

# our payload cell positions (row, col) for p1..p12
_P = {1: (3, 4), 2: (1, 4), 3: (4, 3), 4: (3, 3), 5: (1, 3), 6: (0, 3),
      7: (4, 1), 8: (3, 1), 9: (1, 1), 10: (0, 1), 11: (3, 0), 12: (1, 0)}
_CODE_RE = re.compile(r"[0-9A-Fa-f]{6}")


def payload_hex(tag):
    """The 24-bit payload in the printed-label convention (matches kit codes)."""
    g = [[int(tag.cells[r * 5 + c], 2) for c in range(5)] for r in range(5)]
    p = [g[_P[j][0]][_P[j][1]] for j in range(1, 13)]
    return f"{sum(p[j] * (4 ** j) for j in range(12)):06X}"


def code_from_name(path):
    m = _CODE_RE.findall(os.path.splitext(os.path.basename(path))[0])
    return m[-1].upper() if m else None


def load_image(path, zoom=3):
    if path.lower().endswith(".pdf"):
        import fitz
        pix = fitz.open(path)[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return Image.open(path).convert("RGB")


def main(folder):
    exts = ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = sorted(f for e in exts for f in glob.glob(os.path.join(folder, e)))
    if not files:
        print(f"no tag files in {folder}")
        return False

    decoded = affine_ok = code_ok = 0
    have_codes = 0
    pairs = []
    for path in files:
        code = code_from_name(path)
        have_codes += code is not None
        try:
            tags = freelens.detect_tags(load_image(path), n=5, validate_crc="affine")
        except Exception as exc:
            print(f"{os.path.basename(path):45s} ERROR {type(exc).__name__}")
            continue
        # pick the real tag: by filename code if we have it, else any affine-valid
        if code is not None:
            chosen = [t for t in tags if payload_hex(t) == code]
        else:
            chosen = [t for t in tags if t.valid]
        if not chosen:
            print(f"{os.path.basename(path):45s} not decoded ({len(tags)} frames)")
            continue
        decoded += 1
        tag = chosen[0]
        if code is not None and payload_hex(tag) == code:
            code_ok += 1
        if realcrc.valid_real_crc(tag.bit_string):
            affine_ok += 1
        pairs.append((tag.message, tag.crc))

    print("\n--- summary ---")
    print(f"files                     : {len(files)}")
    print(f"decoded                   : {decoded}")
    if have_codes:
        print(f"payload == filename code  : {code_ok} / {have_codes}")
    print(f"affine checksum valid     : {affine_ok}")

    distinct = list({m: c for m, c in pairs}.items())
    print(f"distinct (message, crc)   : {len(distinct)}")
    if len(distinct) >= 25:
        A, b, unique, consistent = d.derive(distinct)
        matches = np.array_equal(A, realcrc.A_f) and np.array_equal(b, realcrc.b_f)
        print(f"re-derivation             : consistent={consistent} unique={unique} "
              f"matches_shipped={matches}")
        if consistent and not unique:
            print("  (kit is consistent with the shipped matrix but its codes share a"
                  " prefix, so they do not vary across enough payload bits to pin it"
                  " uniquely)")
    return affine_ok == decoded and decoded > 0


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_REPO, "dataset", "freekit")
    sys.exit(0 if main(folder) else 1)
