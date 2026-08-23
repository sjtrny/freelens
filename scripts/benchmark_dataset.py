#!/usr/bin/env python3
"""Benchmark detection and decoding against the field photograph corpus."""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from freelens import detect_tags
from scripts.evaluation_dataset import DEFAULT_MANIFEST, load_cases


def benchmark_dataset(manifest_path=DEFAULT_MANIFEST, detector=detect_tags):
    manifest_path = Path(manifest_path)
    dataset_root = manifest_path.parent
    results = []

    for case in load_cases(manifest_path):
        if case["tags"] is None:
            continue
        scorable_tags = [tag for tag in case["tags"] if tag.get("scorable", True)]
        if any(tag["message"] is None for tag in scorable_tags):
            continue
        expected = [tag["message"] for tag in scorable_tags]
        ignored = [
            tag["message"]
            for tag in case["tags"]
            if not tag.get("scorable", True) and tag["message"] is not None
        ]
        result = {
            "image": case["image"],
            "expected": expected,
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
            result.update(matched=0, missed=expected, unexpected=[])
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
        for message in ignored:
            if message in unmatched:
                unmatched.remove(message)

        result.update(
            status="pass" if not missed and not unmatched else "fail",
            actual=actual,
            seconds=round(time.perf_counter() - started, 6),
            matched=matched,
            missed=missed,
            unexpected=unmatched,
        )
        results.append(result)

    return results


def summarize(results):
    positives = [result for result in results if result["expected"]]
    negatives = [result for result in results if not result["expected"]]
    exact = sum(result["status"] == "pass" for result in results)
    expected = sum(len(result["expected"]) for result in results)
    matched = sum(result["matched"] for result in results)
    positive_detections = sum(bool(result["actual"]) for result in positives)
    clean_negatives = sum(
        result["status"] != "error" and not result["actual"] for result in negatives
    )

    return {
        "positive_images": len(positives),
        "positive_images_detected": positive_detections,
        "positive_detection_percent": (
            round(100 * positive_detections / len(positives), 2) if positives else 0
        ),
        "negative_images": len(negatives),
        "clean_negative_images": clean_negatives,
        "scorable_images": len(results),
        "exact_images": exact,
        "exact_percent": round(100 * exact / len(results), 2) if results else 0,
        "expected_tags": expected,
        "matched_tags": matched,
        "matched_percent": round(100 * matched / expected, 2) if expected else 100,
        "unexpected_tags": sum(len(result["unexpected"]) for result in results),
        "errors": sum(result["status"] == "error" for result in results),
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
        "Scorable positive images detected: "
        f"{summary['positive_images_detected']}/{summary['positive_images']} "
        f"({summary['positive_detection_percent']:.2f}%)"
    )
    print(
        f"Negative images clean: {summary['clean_negative_images']}/"
        f"{summary['negative_images']}"
    )
    print(
        f"Exact scorable images: {summary['exact_images']}/"
        f"{summary['scorable_images']} "
        f"({summary['exact_percent']:.2f}%)"
    )
    print(
        f"Expected tags found: {summary['matched_tags']}/{summary['expected_tags']} "
        f"({summary['matched_percent']:.2f}%)"
    )
    print(f"Unexpected CRC-valid tags: {summary['unexpected_tags']}")
    print(f"Errors: {summary['errors']}")
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
