#!/usr/bin/env python3
"""Verify an uncommitted NaviLens PDF archive without extracting it."""

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

from freelens import detect_tags

EXPECTED_CASE_COUNT = 142
EXPECTED_ARCHIVE_SHA256 = (
    "a93706ebe73b17e37e55af015151c531c396e0d5ccb3bfddaba0156edb33b07a"
)
CODE_RE = re.compile(r"(?<![0-9A-Fa-f])([0-9A-Fa-f]{6})(?![0-9A-Fa-f])")


class ArchiveVerificationError(RuntimeError):
    """Raised when archive integrity or a known-answer check fails."""


def _code_from_member(member_name):
    matches = CODE_RE.findall(Path(member_name).stem)
    if len(matches) != 1:
        raise ArchiveVerificationError(
            f"expected exactly one six-hex code in archive member: {member_name}"
        )
    return matches[0].upper()


def read_archive_cases(archive_path, expected_count=EXPECTED_CASE_COUNT):
    """Read and validate case metadata without extracting archive members."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ArchiveVerificationError(f"archive not found: {archive_path}")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise TypeError("expected_count must be an int")
    if expected_count < 1:
        raise ValueError("expected_count must be positive")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            encrypted_members = [
                member.filename for member in members if member.flag_bits & 0x1
            ]
            if encrypted_members:
                names = ", ".join(sorted(encrypted_members))
                raise ArchiveVerificationError(
                    f"encrypted archive members are not supported: {names}"
                )
            bad_member = archive.testzip()
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ArchiveVerificationError(
            f"could not read ZIP archive {archive_path}: {error}"
        ) from error

    if bad_member is not None:
        raise ArchiveVerificationError(f"ZIP integrity check failed: {bad_member}")

    non_pdf_members = [
        member.filename
        for member in members
        if Path(member.filename).suffix.lower() != ".pdf"
    ]
    if non_pdf_members:
        names = ", ".join(sorted(non_pdf_members))
        raise ArchiveVerificationError(f"unexpected non-PDF archive members: {names}")

    if len(members) != expected_count:
        raise ArchiveVerificationError(
            f"archive contains {len(members)} cases; expected {expected_count}"
        )

    cases = []
    seen_members = set()
    seen_codes = set()
    for member in sorted(members, key=lambda item: item.filename):
        if member.filename in seen_members:
            raise ArchiveVerificationError(
                f"duplicate archive member name: {member.filename}"
            )
        seen_members.add(member.filename)

        code = _code_from_member(member.filename)
        if code in seen_codes:
            raise ArchiveVerificationError(f"duplicate six-hex case code: {code}")
        seen_codes.add(code)

        cases.append(
            {
                "case_id": code.lower(),
                "member_name": member.filename,
                "expected_message": f"{int(code, 16):024b}",
            }
        )

    return cases


def _render_pdf(pdf_bytes, member_name, zoom):
    try:
        import pymupdf
    except ImportError as error:
        raise ArchiveVerificationError(
            "PDF verification requires PyMuPDF; install `freelens[dataset]`"
        ) from error

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise ArchiveVerificationError(
            f"could not open PDF {member_name}: {type(error).__name__}: {error}"
        ) from error

    with document:
        if document.page_count != 1:
            raise ArchiveVerificationError(
                f"expected one PDF page in {member_name}; found {document.page_count}"
            )
        pixmap = document[0].get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def verify_archive(
    archive_path,
    expected_count=EXPECTED_CASE_COUNT,
    expected_sha256=EXPECTED_ARCHIVE_SHA256,
    zoom=3.0,
    progress=None,
):
    """Decode every PDF and compare it with its filename-derived message."""
    if isinstance(zoom, bool) or not isinstance(zoom, (int, float)):
        raise TypeError("zoom must be a number")
    if zoom <= 0:
        raise ValueError("zoom must be positive")

    archive_path = Path(archive_path)
    cases = read_archive_cases(archive_path, expected_count=expected_count)
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected_sha256 must be 64 lowercase hexadecimal digits")
        actual_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ArchiveVerificationError(
                f"archive SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )

    errors = []

    with zipfile.ZipFile(archive_path) as archive:
        for index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            try:
                pdf_bytes = archive.read(case["member_name"])
                with _render_pdf(pdf_bytes, case["member_name"], zoom) as image:
                    tags = detect_tags(
                        image,
                        n=5,
                        validate_crc=True,
                        require_valid_crc=False,
                    )
            except Exception as error:  # report every corpus failure together
                errors.append(
                    f"{case_id}: decode failed: {type(error).__name__}: {error}"
                )
            else:
                matching = [
                    tag for tag in tags if tag.message == case["expected_message"]
                ]
                if not matching:
                    decoded = ", ".join(sorted({tag.message for tag in tags}))
                    detail = decoded if decoded else "no tags detected"
                    errors.append(
                        f"{case_id}: no tag matched expected message "
                        f"{case['expected_message']} ({detail})"
                    )
                elif not any(tag.crc_valid is True for tag in matching):
                    crc_values = ", ".join(
                        sorted({f"{int(tag.crc, 2):04X}" for tag in matching})
                    )
                    errors.append(
                        f"{case_id}: matching message failed deployed CRC "
                        f"validation (decoded CRC: {crc_values})"
                    )

            if progress is not None:
                progress(index, len(cases), case_id)

    if errors:
        raise ArchiveVerificationError("\n".join(errors))
    return len(cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_CASE_COUNT)
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_ARCHIVE_SHA256,
        help="expected lowercase archive digest",
    )
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    def report_progress(index, total, case_id):
        if args.verbose or index % 10 == 0 or index == total:
            print(f"checked {index}/{total}: {case_id}", flush=True)

    try:
        count = verify_archive(
            args.archive,
            expected_count=args.expected_count,
            expected_sha256=args.expected_sha256,
            zoom=args.zoom,
            progress=report_progress,
        )
    except ArchiveVerificationError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"{count}/{args.expected_count} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
