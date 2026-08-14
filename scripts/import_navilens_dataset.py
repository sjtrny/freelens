#!/usr/bin/env python3
"""Deterministically import an authorized NaviLens free-kit corpus."""

import argparse
import csv
import hashlib
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

EXPECTED_CASE_COUNT = 142
SOURCE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".pdf", ".png", ".tif", ".tiff"}
CODE_RE = re.compile(r"[0-9A-Fa-f]{6}")
MANIFEST_FIELDS = (
    "case_id",
    "source_file",
    "image_file",
    "expected_n",
    "expected_message",
    "expected_crc",
    "expected_valid",
    "notes",
)


def _safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(
                    f"archive member escapes destination: {member.filename}"
                )
        zip_file.extractall(destination)


def _source_tree(source, temporary_directory):
    source = source.resolve()
    if source.is_dir():
        return source
    if source.is_file() and source.suffix.lower() == ".zip":
        extracted = Path(temporary_directory) / "extracted"
        extracted.mkdir()
        _safe_extract(source, extracted)
        return extracted
    raise ValueError("source must be a directory or .zip archive")


def _code_from_name(path):
    matches = CODE_RE.findall(path.stem)
    if not matches:
        raise ValueError(f"no six-hex expected message in filename: {path.name}")
    return matches[-1].upper()


def _load_source_image(path, zoom):
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as error:
            raise RuntimeError(
                "PDF import requires PyMuPDF; install `freelens[dataset]`"
            ) from error

        with fitz.open(path) as document:
            if len(document) != 1:
                raise ValueError(f"expected one PDF page: {path}")
            pixmap = document[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    with Image.open(path) as image:
        return image.convert("RGB")


def _write_png(image, destination):
    image.convert("RGB").save(
        destination,
        format="PNG",
        optimize=False,
        compress_level=9,
    )


def _checksum_paths(destination):
    return sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )


def _write_checksums(destination):
    checksum_file = destination / "SHA256SUMS"
    working_directory = Path.cwd().resolve()
    lines = []
    for path in _checksum_paths(destination):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        resolved = path.resolve()
        try:
            recorded_path = resolved.relative_to(working_directory)
        except ValueError:
            recorded_path = resolved.relative_to(destination.resolve())
        lines.append(f"{digest}  {recorded_path.as_posix()}\n")
    checksum_file.write_text("".join(lines), encoding="utf-8")


def import_dataset(source, destination, *, expected_count, zoom, force):
    destination = destination.resolve()
    source_output = destination / "source"
    image_output = destination / "images"
    managed_targets = (source_output, image_output, destination / "manifest.csv")
    occupied = [target for target in managed_targets if target.exists()]
    if occupied and not force:
        names = ", ".join(str(path) for path in occupied)
        raise FileExistsError(f"refusing to overwrite existing dataset paths: {names}")

    destination.mkdir(parents=True, exist_ok=True)
    if force:
        for directory in (source_output, image_output):
            if directory.is_dir():
                shutil.rmtree(directory)
        for file_path in (destination / "manifest.csv", destination / "SHA256SUMS"):
            if file_path.is_file():
                file_path.unlink()
    source_output.mkdir()
    image_output.mkdir()

    with tempfile.TemporaryDirectory(prefix="freelens-navilens-import-") as temporary:
        source_tree = _source_tree(Path(source), temporary)
        source_files = sorted(
            path
            for path in source_tree.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
        )
        if len(source_files) != expected_count:
            raise ValueError(
                f"found {len(source_files)} source cases; expected {expected_count}"
            )

        rows = []
        seen_codes = set()
        for source_path in source_files:
            relative_source = source_path.relative_to(source_tree)
            code = _code_from_name(source_path)
            if code in seen_codes:
                raise ValueError(f"duplicate six-hex case code: {code}")
            seen_codes.add(code)

            copied_source = source_output / relative_source
            copied_source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, copied_source)

            normalized_image = image_output / f"{code}.png"
            image = _load_source_image(source_path, zoom)
            _write_png(image, normalized_image)

            rows.append(
                {
                    "case_id": code.lower(),
                    "source_file": copied_source.relative_to(destination).as_posix(),
                    "image_file": normalized_image.relative_to(destination).as_posix(),
                    "expected_n": "5",
                    "expected_message": f"{int(code, 16):024b}",
                    "expected_crc": "",
                    "expected_valid": "true",
                    "notes": (
                        "Expected message comes independently from the six-hex "
                        "code in the supplied filename."
                    ),
                }
            )

    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    _write_checksums(destination)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_CASE_COUNT)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        count = import_dataset(
            args.source,
            args.destination,
            expected_count=args.expected_count,
            zoom=args.zoom,
            force=args.force,
        )
    except Exception as error:
        print(f"import failed: {error}", file=sys.stderr)
        return 1

    print(f"imported {count}; skipped 0; failed 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
