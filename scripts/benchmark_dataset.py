#!/usr/bin/env python3
"""Benchmark detection and decoding against the field photograph corpus."""

import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image

from freelens import detect_tags

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "dataset" / "evaluation.json"
MESSAGE_PATTERN = re.compile(r"[0-9A-F]{6}")


def load_cases(manifest_path):
    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("evaluation manifest must contain a list")

    listed_images = [case["image"] for case in cases]
    if len(listed_images) != len(set(listed_images)):
        raise ValueError("evaluation manifest contains duplicate images")

    dataset_root = manifest_path.parent
    dataset_images = {
        path.relative_to(dataset_root).as_posix()
        for directory in ("positives", "negatives")
        for path in (dataset_root / directory).iterdir()
        if path.is_file()
    }
    if set(listed_images) != dataset_images:
        missing = sorted(dataset_images - set(listed_images))
        extra = sorted(set(listed_images) - dataset_images)
        raise ValueError(f"manifest coverage differs: missing={missing}, extra={extra}")

    for case in cases:
        messages = case["messages"]
        if messages is not None and (
            not isinstance(messages, list)
            or not all(
                isinstance(message, str) and MESSAGE_PATTERN.fullmatch(message)
                for message in messages
            )
        ):
            raise ValueError(f"invalid messages for {case['image']}")

        conditions = case.get("conditions", [])
        if not isinstance(conditions, list) or not all(
            isinstance(condition, str) and condition for condition in conditions
        ):
            raise ValueError(f"invalid conditions for {case['image']}")

    return cases


def benchmark_dataset(manifest_path=DEFAULT_MANIFEST, detector=detect_tags):
    manifest_path = Path(manifest_path)
    dataset_root = manifest_path.parent
    results = []

    for case in load_cases(manifest_path):
        expected = case["messages"]
        result = {
            "image": case["image"],
            "expected": expected,
            "conditions": case.get("conditions", []),
        }

        started = time.perf_counter()
        try:
            with Image.open(dataset_root / case["image"]) as image:
                tags = detector(
                    image.convert("RGB"),
                    n=5,
                    validate_crc=True,
                    require_valid_crc=True,
                )
            actual = sorted(f"{int(tag.message, 2):06X}" for tag in tags)
        except Exception as error:
            result.update(
                status="error",
                actual=[],
                error=f"{type(error).__name__}: {error}",
                seconds=round(time.perf_counter() - started, 6),
            )
            if expected is not None:
                result.update(matched=0, missed=expected, unexpected=[])
            results.append(result)
            continue

        if expected is None:
            result.update(
                status="manual-review",
                actual=actual,
                seconds=round(time.perf_counter() - started, 6),
            )
            results.append(result)
            continue

        unmatched = actual.copy()
        matched = 0
        missed = []
        for message in expected:
            if message in unmatched:
                unmatched.remove(message)
                matched += 1
            else:
                missed.append(message)

        result.update(
            status="pass" if actual == sorted(expected) else "fail",
            actual=actual,
            seconds=round(time.perf_counter() - started, 6),
            matched=matched,
            missed=missed,
            unexpected=unmatched,
        )
        results.append(result)

    return results


def summarize(results):
    scored = [result for result in results if result["expected"] is not None]
    positives = [
        result for result in results if result["image"].startswith("positives/")
    ]
    negatives = [
        result for result in results if result["image"].startswith("negatives/")
    ]
    conditioned_positives = [result for result in positives if result["conditions"]]
    exact = sum(result["status"] == "pass" for result in scored)
    expected = sum(len(result["expected"]) for result in scored)
    matched = sum(result["matched"] for result in scored)
    positive_detections = sum(bool(result["actual"]) for result in positives)
    conditioned_detections = sum(
        bool(result["actual"]) for result in conditioned_positives
    )
    clean_negatives = sum(
        result["status"] != "error" and not result["actual"] for result in negatives
    )

    return {
        "positive_images": len(positives),
        "positive_images_detected": positive_detections,
        "positive_detection_percent": (
            round(100 * positive_detections / len(positives), 2) if positives else 0
        ),
        "conditioned_positive_images": len(conditioned_positives),
        "conditioned_positive_images_detected": conditioned_detections,
        "conditioned_positive_detection_percent": (
            round(100 * conditioned_detections / len(conditioned_positives), 2)
            if conditioned_positives
            else 0
        ),
        "negative_images": len(negatives),
        "clean_negative_images": clean_negatives,
        "scored_images": len(scored),
        "exact_images": exact,
        "exact_percent": round(100 * exact / len(scored), 2) if scored else 0,
        "expected_tags": expected,
        "matched_tags": matched,
        "matched_percent": round(100 * matched / expected, 2) if expected else 100,
        "unexpected_tags": sum(len(result["unexpected"]) for result in scored),
        "errors": sum(result["status"] == "error" for result in results),
        "manual_review_images": sum(result["expected"] is None for result in results),
        "seconds": round(sum(result.get("seconds", 0) for result in results), 3),
    }


def print_report(results, summary):
    for result in results:
        if result["status"] not in {"fail", "error"}:
            continue
        detail = result.get("error") or (
            f"expected={result['expected']} actual={result['actual']}"
        )
        print(f"{result['status'].upper()} {result['image']}: {detail}")

    print(
        "Positive images detected: "
        f"{summary['positive_images_detected']}/{summary['positive_images']} "
        f"({summary['positive_detection_percent']:.2f}%)"
    )
    if summary["conditioned_positive_images"]:
        print(
            "Conditioned positive images detected: "
            f"{summary['conditioned_positive_images_detected']}/"
            f"{summary['conditioned_positive_images']} "
            f"({summary['conditioned_positive_detection_percent']:.2f}%)"
        )
    print(
        f"Negative images clean: {summary['clean_negative_images']}/"
        f"{summary['negative_images']}"
    )
    print(
        f"Exact labelled images: {summary['exact_images']}/{summary['scored_images']} "
        f"({summary['exact_percent']:.2f}%)"
    )
    print(
        f"Expected tags found: {summary['matched_tags']}/{summary['expected_tags']} "
        f"({summary['matched_percent']:.2f}%)"
    )
    print(f"Unexpected CRC-valid tags: {summary['unexpected_tags']}")
    print(f"Errors: {summary['errors']}")
    print(f"Manual review images: {summary['manual_review_images']}")
    print(f"Elapsed: {summary['seconds']:.3f}s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, help="write detailed results as JSON")
    args = parser.parse_args(argv)

    results = benchmark_dataset(args.manifest)
    summary = summarize(results)
    print_report(results, summary)

    if args.output:
        report = {"summary": summary, "cases": results}
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
