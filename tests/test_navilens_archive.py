import os
import zipfile
from pathlib import Path

import pytest

from scripts.verify_navilens_archive import (
    EXPECTED_CASE_COUNT,
    ArchiveVerificationError,
    read_archive_cases,
    verify_archive,
)

ARCHIVE_ENVIRONMENT_VARIABLE = "FREELENS_NAVILENS_ARCHIVE"
MISSING_ARCHIVE_MESSAGE = (
    f"NaviLens archive is not present; set {ARCHIVE_ENVIRONMENT_VARIABLE}"
)


def _write_archive(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name in members:
            archive.writestr(name, b"PDF test placeholder")


def test_archive_reader_derives_messages_from_filenames(tmp_path):
    archive = tmp_path / "codes.zip"
    _write_archive(
        archive,
        ["Bathroom_AAB00B_210mm.pdf", "Exit_B1269C_210mm.pdf"],
    )

    cases = read_archive_cases(archive, expected_count=2)

    assert [case["case_id"] for case in cases] == ["aab00b", "b1269c"]
    assert [case["expected_message"] for case in cases] == [
        f"{int('AAB00B', 16):024b}",
        f"{int('B1269C', 16):024b}",
    ]


def test_archive_reader_rejects_duplicate_codes(tmp_path):
    archive = tmp_path / "codes.zip"
    _write_archive(
        archive,
        ["Bathroom_AAB00B_210mm.pdf", "Other_AAB00B_105mm.pdf"],
    )

    with pytest.raises(ArchiveVerificationError, match="duplicate six-hex"):
        read_archive_cases(archive, expected_count=2)


def test_archive_reader_rejects_unexpected_members(tmp_path):
    archive = tmp_path / "codes.zip"
    _write_archive(archive, ["Bathroom_AAB00B_210mm.pdf", "README.txt"])

    with pytest.raises(ArchiveVerificationError, match="unexpected non-PDF"):
        read_archive_cases(archive, expected_count=2)


def test_archive_reader_rejects_unexpected_case_count(tmp_path):
    archive = tmp_path / "codes.zip"
    _write_archive(archive, ["Bathroom_AAB00B_210mm.pdf"])

    with pytest.raises(ArchiveVerificationError, match="contains 1 cases; expected 2"):
        read_archive_cases(archive, expected_count=2)


def test_archive_verifier_rejects_a_different_archive(tmp_path):
    archive = tmp_path / "codes.zip"
    _write_archive(archive, ["Bathroom_AAB00B_210mm.pdf"])

    with pytest.raises(ArchiveVerificationError, match="SHA-256 mismatch"):
        verify_archive(archive, expected_count=1)


@pytest.mark.integration
def test_navilens_provided_archive():
    archive_value = os.environ.get(ARCHIVE_ENVIRONMENT_VARIABLE)
    if not archive_value:
        pytest.skip(MISSING_ARCHIVE_MESSAGE)

    archive = Path(archive_value)
    if not archive.is_file():
        pytest.fail(f"{MISSING_ARCHIVE_MESSAGE}: {archive}")

    try:
        passed = verify_archive(archive, expected_count=EXPECTED_CASE_COUNT)
    except ArchiveVerificationError as error:
        pytest.fail(str(error))

    assert passed == EXPECTED_CASE_COUNT
