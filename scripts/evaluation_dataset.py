"""Load and validate the field-photo evaluation dataset."""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from dataclasses import replace as replace_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from PIL import Image, UnidentifiedImageError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "dataset" / "evaluation.json"
MESSAGE_PATTERN = re.compile(r"[0-9A-F]{6}")
LOCATION_CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class EvaluationDataset:
    """A validated manifest and its exact image allowlist."""

    manifest_path: Path
    cases: tuple[dict, ...]
    image_paths: Mapping[str, Path]
    image_sizes: Mapping[str, tuple[int, int]]

    def case(self, index):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self.cases)
        ):
            raise KeyError(f"unknown evaluation case: {index!r}")
        return self.cases[index]

    def image_path(self, image_id):
        try:
            return self.image_paths[image_id]
        except (KeyError, TypeError) as error:
            raise KeyError(f"unknown dataset image: {image_id!r}") from error

    def image_size(self, image_id):
        try:
            return self.image_sizes[image_id]
        except (KeyError, TypeError) as error:
            raise KeyError(f"unknown dataset image: {image_id!r}") from error


def _validate_conditions(conditions, owner):
    if not isinstance(conditions, list) or not all(
        isinstance(condition, str) and condition for condition in conditions
    ):
        raise ValueError(f"invalid conditions for {owner}")
    return list(conditions)


def _validate_location(location, owner):
    if not isinstance(location, dict) or set(location) != set(LOCATION_CORNERS):
        raise ValueError(f"invalid location for {owner}")

    normalized = {}
    for corner in LOCATION_CORNERS:
        point = location[corner]
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(type(coordinate) is not int for coordinate in point)
            or any(coordinate < 0 for coordinate in point)
        ):
            raise ValueError(f"invalid {corner} for {owner}")
        normalized[corner] = list(point)
    return normalized


def _normalize_tag(tag, image_id, index):
    owner = f"tag {index} in {image_id}"
    if not isinstance(tag, dict):
        raise ValueError(f"{owner} must be an object")

    if "message" not in tag:
        raise ValueError(f"invalid message for {owner}")
    message = tag["message"]
    if message is not None and (
        not isinstance(message, str) or not MESSAGE_PATTERN.fullmatch(message)
    ):
        raise ValueError(f"invalid message for {owner}")

    normalized = {
        "message": message,
        "conditions": _validate_conditions(tag.get("conditions", []), owner),
    }
    if "scorable" in tag:
        if type(tag["scorable"]) is not bool:
            raise ValueError(f"invalid scorable value for {owner}")
        if not tag["scorable"]:
            normalized["scorable"] = False
    if "description" in tag:
        description = tag["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"invalid description for {owner}")
        normalized["description"] = description.strip()
    if "location" in tag:
        normalized["location"] = _validate_location(tag["location"], owner)
    return normalized


def _normalize_case(case, index):
    if not isinstance(case, dict):
        raise ValueError(f"evaluation case {index} must be an object")

    image_id = case.get("image")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError(f"evaluation case {index} has an invalid image")
    if "conditions" in case:
        raise ValueError(f"conditions for {image_id} must be attached to tags")
    if "tags" not in case:
        raise ValueError(f"missing tags for {image_id}")

    tags = case["tags"]
    if tags is not None and not isinstance(tags, list):
        raise ValueError(f"invalid tags for {image_id}")

    return {
        "image": image_id,
        "tags": (
            None
            if tags is None
            else [
                _normalize_tag(tag, image_id, tag_index)
                for tag_index, tag in enumerate(tags)
            ]
        ),
    }


def _find_dataset_images(dataset_root):
    resolved_root = dataset_root.resolve()
    images = {}

    for path in (dataset_root / "images").iterdir():
        if not path.is_file():
            continue

        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"dataset image resolves outside dataset: {path}")

        image_id = path.relative_to(dataset_root).as_posix()
        images[image_id] = resolved

    return images


def _read_image_sizes(image_paths):
    sizes = {}
    for image_id, path in image_paths.items():
        try:
            with Image.open(path) as image:
                sizes[image_id] = image.size
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"invalid dataset image: {image_id}") from error
    return sizes


def _validate_location_bounds(cases, image_sizes):
    for case in cases:
        width, height = image_sizes[case["image"]]
        for tag_index, tag in enumerate(case["tags"] or []):
            for corner, (x, y) in tag.get("location", {}).items():
                if x >= width or y >= height:
                    raise ValueError(
                        f"{corner} for tag {tag_index} in {case['image']} "
                        f"is outside the image"
                    )


def load_dataset(manifest_path=DEFAULT_MANIFEST):
    """Return a validated manifest and an allowlist of its image paths."""

    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("evaluation manifest must contain a list")

    normalized_cases = tuple(
        _normalize_case(case, index) for index, case in enumerate(cases)
    )
    listed_images = [case["image"] for case in normalized_cases]
    if len(listed_images) != len(set(listed_images)):
        raise ValueError("evaluation manifest contains duplicate images")

    image_paths = _find_dataset_images(manifest_path.parent)
    if set(listed_images) != set(image_paths):
        missing = sorted(set(image_paths) - set(listed_images))
        extra = sorted(set(listed_images) - set(image_paths))
        raise ValueError(f"manifest coverage differs: missing={missing}, extra={extra}")

    image_sizes = _read_image_sizes(image_paths)
    _validate_location_bounds(normalized_cases, image_sizes)

    return EvaluationDataset(
        manifest_path=manifest_path,
        cases=normalized_cases,
        image_paths=MappingProxyType(image_paths),
        image_sizes=MappingProxyType(image_sizes),
    )


def load_cases(manifest_path=DEFAULT_MANIFEST):
    """Return evaluation cases for callers that do not need image lookup."""

    return list(load_dataset(manifest_path).cases)


def _write_cases(dataset, cases):
    manifest_path = dataset.manifest_path
    descriptor, temporary_name = tempfile.mkstemp(
        dir=manifest_path.parent,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(cases, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        updated = load_dataset(temporary_path)
        os.replace(temporary_path, manifest_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return replace_dataclass(updated, manifest_path=manifest_path)


def add_tag(dataset, case_index, tag):
    """Validate and atomically append one tag, returning the updated dataset."""

    manifest_path = dataset.manifest_path
    with manifest_path.open(encoding="utf-8") as file:
        cases = json.load(file)

    if (
        type(case_index) is not int
        or not isinstance(cases, list)
        or not 0 <= case_index < len(cases)
    ):
        raise KeyError(f"unknown evaluation case: {case_index!r}")

    case = cases[case_index]
    if not isinstance(case, dict):
        raise KeyError(f"unknown evaluation case: {case_index!r}")
    if case.get("tags") is None:
        case["tags"] = []
    if not isinstance(case["tags"], list):
        raise KeyError(f"unknown evaluation case: {case_index!r}")

    tag_index = len(case["tags"])
    case["tags"].append(_normalize_tag(tag, case.get("image"), tag_index))
    return _write_cases(dataset, cases)


def update_tag(dataset, case_index, tag_index, tag):
    """Validate and atomically persist one tag, returning the updated dataset."""

    manifest_path = dataset.manifest_path
    with manifest_path.open(encoding="utf-8") as file:
        cases = json.load(file)

    if (
        type(case_index) is not int
        or not isinstance(cases, list)
        or not 0 <= case_index < len(cases)
    ):
        raise KeyError(f"unknown evaluation case: {case_index!r}")

    case = cases[case_index]
    tags = case.get("tags") if isinstance(case, dict) else None
    if (
        type(tag_index) is not int
        or not isinstance(tags, list)
        or not 0 <= tag_index < len(tags)
    ):
        raise KeyError(f"unknown tag: {tag_index!r}")

    tags[tag_index] = _normalize_tag(tag, case.get("image"), tag_index)
    return _write_cases(dataset, cases)
