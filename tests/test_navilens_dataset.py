import csv
import os
from pathlib import Path

import pytest

from scripts.verify_navilens_dataset import (
    EXPECTED_CASE_COUNT,
    DatasetVerificationError,
    is_lfs_pointer,
    verify_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPOSITORY_ROOT / "dataset" / "navilens-provided"
MANIFEST = DATASET_ROOT / "manifest.csv"
REQUIRE_DATASET = os.environ.get("FREELENS_REQUIRE_NAVILENS_DATASET") == "1"
MISSING_DATASET_MESSAGE = "NaviLens dataset is not present; run `git lfs pull`"


def _dataset_paths_from_manifest():
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [
        DATASET_ROOT / row[field]
        for row in rows
        for field in ("source_file", "image_file")
        if row.get(field)
    ]


@pytest.mark.integration
def test_navilens_provided_dataset():
    if not MANIFEST.is_file():
        if REQUIRE_DATASET:
            pytest.fail(MISSING_DATASET_MESSAGE)
        pytest.skip(MISSING_DATASET_MESSAGE)

    lfs_pointers = [
        path for path in _dataset_paths_from_manifest() if is_lfs_pointer(path)
    ]
    if lfs_pointers:
        if REQUIRE_DATASET:
            pytest.fail(MISSING_DATASET_MESSAGE)
        pytest.skip(MISSING_DATASET_MESSAGE)

    try:
        passed = verify_dataset(MANIFEST, expected_count=EXPECTED_CASE_COUNT)
    except DatasetVerificationError as error:
        pytest.fail(str(error))

    assert passed == EXPECTED_CASE_COUNT
