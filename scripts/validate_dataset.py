"""Decode a folder of tag images and report checksum validity at scale.

Decodes every image with ``freelens.detect_tags`` and, for each decoded tag,
reports whether it passes the affine (deployed) checksum and the patent CRC.
Useful for the NaviLens free kit or the bundled ``dataset/positives``.

    python scripts/validate_dataset.py                     # dataset/positives
    python scripts/validate_dataset.py path/to/tags

Needs the CV deps (cv2, crc).
"""

import glob
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import freelens

_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def main(folder):
    files = sorted(f for e in _EXTS for f in glob.glob(os.path.join(folder, e)))
    if not files:
        print(f"no images found in {folder}")
        return False

    images_with_tag = 0
    images_affine_ok = 0
    total_tags = 0
    affine_ok = 0
    patent_ok = 0

    errors = 0
    for path in files:
        img = Image.open(path).convert("RGB")
        try:
            tags = freelens.detect_tags(img, n=5, validate_crc="affine")
        except Exception as exc:  # a degenerate frame can trip Tag's own checks
            errors += 1
            print(f"{os.path.basename(path):40s} ERROR {type(exc).__name__}")
            continue
        decoded = [t for t in tags if t.valid is not None]
        aff = [t for t in tags if t.valid]
        pat = [t for t in tags if freelens.valid_crc(t.bit_string)]
        total_tags += len(tags)
        affine_ok += len(aff)
        patent_ok += len(pat)
        if tags:
            images_with_tag += 1
        if aff:
            images_affine_ok += 1
        print(f"{os.path.basename(path):40s} tags={len(tags):2d} "
              f"affine_ok={len(aff):2d} patent_ok={len(pat):2d}")

    print("\n--- summary ---")
    print(f"images                : {len(files)}")
    print(f"images errored         : {errors}")
    print(f"images with a tag      : {images_with_tag}")
    print(f"images with affine-ok  : {images_affine_ok}")
    print(f"tags decoded           : {total_tags}")
    print(f"tags passing affine    : {affine_ok}")
    print(f"tags passing patent    : {patent_ok}")
    return True


if __name__ == "__main__":
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "dataset", "positives")
    sys.exit(0 if main(folder) else 1)
