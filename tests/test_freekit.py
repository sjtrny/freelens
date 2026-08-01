"""Validate the deployed (affine) checksum across a whole set of real tags.

Point this at the NaviLens free-kit tag images (or any folder of tag photos):

    set FREELENS_FREEKIT=path\\to\\tags        # Windows
    export FREELENS_FREEKIT=path/to/tags       # POSIX

or drop the images in ``dataset/freekit/``. Each image is decoded and its
affine checksum checked. The whole module skips when no images are present, so
CI stays green until the kit is added.
"""

import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("cv2")
pytest.importorskip("crc")
import freelens
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIR = os.environ.get("FREELENS_FREEKIT", os.path.join(_REPO, "dataset", "freekit"))
_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
_IMAGES = sorted(f for ext in _EXTS for f in glob.glob(os.path.join(_DIR, ext)))

pytestmark = pytest.mark.skipif(
    not _IMAGES, reason=f"no free-kit images in {_DIR} (set FREELENS_FREEKIT)"
)


@pytest.mark.parametrize("path", _IMAGES, ids=[os.path.basename(p) for p in _IMAGES])
def test_freekit_tag_passes_affine(path):
    img = Image.open(path).convert("RGB")
    tags = freelens.detect_tags(img, n=5, validate_crc="affine")
    assert tags, f"no tag detected in {os.path.basename(path)}"
    assert any(t.valid for t in tags), (
        f"no affine-valid tag in {os.path.basename(path)}")
