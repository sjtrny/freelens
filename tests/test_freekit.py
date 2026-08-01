"""Validate the deployed (affine) checksum across a whole set of real tags.

Point this at the NaviLens free-kit tags (PDFs or images), or any folder of tag
photos:

    set FREELENS_FREEKIT=path\\to\\tags        # Windows
    export FREELENS_FREEKIT=path/to/tags       # POSIX

or drop the files in ``dataset/freekit/``. Each tag is decoded and its affine
checksum checked. The module skips when the folder is empty, so CI stays green
until a kit is added. PDFs additionally need pymupdf (fitz).
"""

import glob
import io
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
_EXTS = ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
_FILES = sorted(f for ext in _EXTS for f in glob.glob(os.path.join(_DIR, ext)))

pytestmark = pytest.mark.skipif(
    not _FILES, reason=f"no free-kit tags in {_DIR} (set FREELENS_FREEKIT)"
)


def _load(path):
    if path.lower().endswith(".pdf"):
        fitz = pytest.importorskip("fitz")
        pix = fitz.open(path)[0].get_pixmap(matrix=fitz.Matrix(3, 3))
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return Image.open(path).convert("RGB")


@pytest.mark.parametrize("path", _FILES, ids=[os.path.basename(p) for p in _FILES])
def test_freekit_tag_passes_affine(path):
    tags = freelens.detect_tags(_load(path), n=5, validate_crc="affine")
    assert tags, f"no tag detected in {os.path.basename(path)}"
    assert any(t.valid for t in tags), (
        f"no affine-valid tag in {os.path.basename(path)}")
