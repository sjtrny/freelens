#!/usr/bin/env python3
"""Verify the manifest-driven NaviLens-provided 5x5 corpus."""

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

from PIL import Image

from freelens import detect_tags

EXPECTED_CASE_COUNT = 142
REQUIRED_COLUMNS = (
    "case_id",
    "source_file",
    "image_file",
    "expected_n",
    "expected_message",
    "expected_crc",
    "expected_valid",
    "notes",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SOURCE_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class DatasetVerificationError(RuntimeError):
    """Raised when any corpus integrity or known-answer check fails."""


def is_lfs_pointer(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return file.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def _resolve_dataset_path(dataset_root, relative_path, field, case_id):
    if not relative_path:
        raise DatasetVerificationError(f"{case_id}: {field} is empty")

    dataset_root = dataset_root.resolve()
    path = (dataset_root / relative_path).resolve()
    if path != dataset_root and dataset_root not in path.parents:
        raise DatasetVerificationError(
            f"{case_id}: {field} escapes the dataset directory"
        )
    return path


def read_manifest(manifest_path, expected_count=EXPECTED_CASE_COUNT):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise DatasetVerificationError(f"manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DatasetVerificationError(f"manifest is missing columns: {missing}")
        rows = list(reader)

    if len(rows) != expected_count:
        raise DatasetVerificationError(
            f"manifest contains {len(rows)} cases; expected {expected_count}"
        )

    seen_case_ids = set()
    for row_number, row in enumerate(rows, start=2):
        case_id = row["case_id"].strip()
        if not case_id:
            raise DatasetVerificationError(
                f"manifest row {row_number} has an empty case_id"
            )
        if case_id in seen_case_ids:
            raise DatasetVerificationError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        row["case_id"] = case_id

        if row["expected_n"].strip() != "5":
            raise DatasetVerificationError(
                f"{case_id}: expected_n must be 5 for CRC-validated cases"
            )

        message = row["expected_message"].strip()
        if len(message) != 24 or set(message) - {"0", "1"}:
            raise DatasetVerificationError(
                f"{case_id}: expected_message must contain exactly 24 binary bits"
            )
        row["expected_message"] = message

        expected_crc = row["expected_crc"].strip()
        if expected_crc and not re.fullmatch(r"[0-9A-F]{4}", expected_crc):
            raise DatasetVerificationError(
                f"{case_id}: expected_crc must be empty or four uppercase hex digits"
            )
        row["expected_crc"] = expected_crc

        expected_valid = row["expected_valid"].strip().lower()
        if expected_valid not in {"true", "false"}:
            raise DatasetVerificationError(
                f"{case_id}: expected_valid must be true or false"
            )
        row["expected_valid"] = expected_valid == "true"

    return rows


def _checksum_target(checksum_file, recorded_path):
    candidate = Path(recorded_path)
    if candidate.is_absolute():
        return candidate

    from_checksum_directory = (checksum_file.parent / candidate).resolve()
    if from_checksum_directory.exists():
        target = from_checksum_directory
    else:
        target = (Path.cwd() / candidate).resolve()

    dataset_root = checksum_file.parent.resolve()
    if target != dataset_root and dataset_root not in target.parents:
        raise DatasetVerificationError(
            f"checksum target escapes the dataset directory: {recorded_path}"
        )
    return target


def verify_checksums(checksum_file, expected_files=None):
    checksum_file = Path(checksum_file)
    if not checksum_file.is_file():
        raise DatasetVerificationError(f"checksum file not found: {checksum_file}")

    checked_paths = set()
    errors = []
    with checksum_file.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match:
                errors.append(f"SHA256SUMS line {line_number} is malformed")
                continue

            expected, recorded_path = match.groups()
            try:
                path = _checksum_target(checksum_file, recorded_path)
            except DatasetVerificationError as error:
                errors.append(str(error))
                continue
            if not path.is_file():
                errors.append(f"checksum target is missing: {recorded_path}")
                continue
            if is_lfs_pointer(path):
                errors.append(f"Git LFS object is not present: {recorded_path}")
                continue

            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"checksum mismatch: {recorded_path}")
                continue
            checked_paths.add(path)

    if expected_files is not None:
        expected_paths = {Path(path).resolve() for path in expected_files}
        for path in sorted(expected_paths - checked_paths):
            errors.append(
                f"file is not represented in SHA256SUMS: "
                f"{path.relative_to(checksum_file.parent.resolve())}"
            )
        for path in sorted(checked_paths - expected_paths):
            errors.append(f"unexpected checksum entry: {path}")

    if errors:
        raise DatasetVerificationError("\n".join(errors))
    if not checked_paths:
        raise DatasetVerificationError("SHA256SUMS contains no file entries")
    return len(checked_paths)


def verify_dataset(manifest_path, expected_count=EXPECTED_CASE_COUNT):
    manifest_path = Path(manifest_path)
    dataset_root = manifest_path.parent
    rows = read_manifest(manifest_path, expected_count=expected_count)
    errors = []
    listed_images = set()
    listed_sources = set()

    try:
        checksum_file = dataset_root / "SHA256SUMS"
        expected_checksum_files = {
            path.resolve()
            for path in dataset_root.rglob("*")
            if path.is_file() and path != checksum_file
        }
        verify_checksums(checksum_file, expected_files=expected_checksum_files)
    except DatasetVerificationError as error:
        errors.extend(str(error).splitlines())

    for row in rows:
        case_id = row["case_id"]
        try:
            source_path = _resolve_dataset_path(
                dataset_root, row["source_file"].strip(), "source_file", case_id
            )
            image_path = _resolve_dataset_path(
                dataset_root, row["image_file"].strip(), "image_file", case_id
            )
        except DatasetVerificationError as error:
            errors.append(str(error))
            continue

        if image_path in listed_images:
            errors.append(f"{case_id}: duplicate image_file in manifest")
        listed_images.add(image_path)
        listed_sources.add(source_path)
        missing = False
        for kind, path in (("source", source_path), ("image", image_path)):
            if not path.is_file():
                errors.append(f"{case_id}: {kind} file is missing: {path}")
                missing = True
            elif is_lfs_pointer(path):
                errors.append(f"{case_id}: {kind} Git LFS object is not present")
                missing = True
        if missing:
            continue

        try:
            with Image.open(image_path) as image:
                tags = detect_tags(
                    image.convert("RGB"),
                    n=5,
                    validate_crc=True,
                    require_valid_crc=False,
                )
        except Exception as error:  # report all corpus failures together
            errors.append(f"{case_id}: decode failed: {type(error).__name__}: {error}")
            continue

        matching = [tag for tag in tags if tag.message == row["expected_message"]]
        if not matching:
            errors.append(
                f"{case_id}: no decoded tag matched expected message "
                f"{row['expected_message']}"
            )
            continue

        if row["expected_crc"]:
            matching = [
                tag
                for tag in matching
                if f"{int(tag.crc, 2):04X}" == row["expected_crc"]
            ]
            if not matching:
                errors.append(
                    f"{case_id}: no matching tag carried CRC {row['expected_crc']}"
                )
                continue

        if not any(tag.crc_valid is row["expected_valid"] for tag in matching):
            errors.append(
                f"{case_id}: CRC validity did not equal "
                f"{str(row['expected_valid']).lower()}"
            )

    images_directory = dataset_root / "images"
    actual_images = (
        {
            path.resolve()
            for path in images_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if images_directory.is_dir()
        else set()
    )
    for extra in sorted(actual_images - listed_images):
        errors.append(
            f"image is not represented in manifest: {extra.relative_to(dataset_root)}"
        )

    source_directory = dataset_root / "source"
    actual_sources = (
        {
            path.resolve()
            for path in source_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
        }
        if source_directory.is_dir()
        else set()
    )
    for extra in sorted(actual_sources - listed_sources):
        errors.append(
            f"source is not represented in manifest: {extra.relative_to(dataset_root)}"
        )

    if errors:
        raise DatasetVerificationError("\n".join(errors))
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/navilens-provided/manifest.csv"),
    )
    parser.add_argument("--expected-count", type=int, default=EXPECTED_CASE_COUNT)
    args = parser.parse_args(argv)

    try:
        count = verify_dataset(args.manifest, expected_count=args.expected_count)
    except DatasetVerificationError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"{count}/{args.expected_count} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
