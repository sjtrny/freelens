from pathlib import Path

import pytest
from PIL import Image

from scripts.import_navilens_dataset import EXPECTED_CASE_COUNT, import_dataset
from scripts.verify_navilens_dataset import read_manifest, verify_checksums


def _write_source_image(path, colour):
    Image.new("RGB", (16, 16), colour).save(path)


def test_importer_writes_independent_messages_manifest_and_checksums(tmp_path):
    source = tmp_path / "source-input"
    source.mkdir()
    _write_source_image(source / "Bathroom_AAB00B.png", "red")
    _write_source_image(source / "Exit_B1269C.png", "blue")
    destination = tmp_path / "imported"

    count = import_dataset(
        source,
        destination,
        expected_count=2,
        zoom=3.0,
        force=False,
    )
    rows = read_manifest(destination / "manifest.csv", expected_count=2)

    assert count == 2
    assert [row["case_id"] for row in rows] == ["aab00b", "b1269c"]
    assert [row["expected_message"] for row in rows] == [
        f"{int('AAB00B', 16):024b}",
        f"{int('B1269C', 16):024b}",
    ]
    assert all(row["expected_crc"] == "" for row in rows)
    assert verify_checksums(destination / "SHA256SUMS") == 5


def test_importer_refuses_to_overwrite_without_force(tmp_path):
    source = tmp_path / "source-input"
    source.mkdir()
    _write_source_image(source / "Exit_B1269C.png", "blue")
    destination = tmp_path / "imported"

    import_dataset(
        source,
        destination,
        expected_count=1,
        zoom=3.0,
        force=False,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        import_dataset(
            source,
            destination,
            expected_count=1,
            zoom=3.0,
            force=False,
        )


def test_importer_fails_on_an_unexpected_case_count(tmp_path):
    source = tmp_path / "source-input"
    source.mkdir()
    _write_source_image(source / "Exit_B1269C.png", "blue")

    with pytest.raises(
        ValueError,
        match=f"found 1 source cases; expected {EXPECTED_CASE_COUNT}",
    ):
        import_dataset(
            source,
            tmp_path / "imported",
            expected_count=EXPECTED_CASE_COUNT,
            zoom=3.0,
            force=False,
        )
