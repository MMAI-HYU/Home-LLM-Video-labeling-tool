from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .schema import VISUAL_FIELDS


ALL_VERIFY_FIELDS = ("activity",) + VISUAL_FIELDS


def review_priority(verification: dict[str, Any]) -> tuple[int, list[str]]:
    reasons = []
    score = 0
    passes = (verification.get("image_only", {}), verification.get("activity_conditioned", {}))
    failed = sum(not item.get("ok", False) for item in passes)
    if failed:
        score += failed * 100
        reasons.append(f"vlm_parse_failure:{failed}")

    disagreements = verification.get("disagreement_fields", [])
    if disagreements:
        score += 12 * len(disagreements)
        reasons.append("pass_disagreement:" + ",".join(disagreements))
    if "activity" in disagreements:
        score += 20

    for pass_name, item in zip(("image_only", "conditioned"), passes):
        parsed = item.get("parsed") or {}
        unknown_fields = [
            field
            for field in ALL_VERIFY_FIELDS
            if parsed.get(field) == "unknown" or parsed.get(field) == ["unknown"]
        ]
        if unknown_fields:
            score += 3 * len(unknown_fields)
            reasons.append(f"{pass_name}_unknown:" + ",".join(unknown_fields))
    return score, reasons or ["low_priority_spot_check"]


def create_review_queue(
    base_rows: list[dict[str, Any]], verification_rows: list[dict[str, Any]], output: Path
) -> list[dict[str, Any]]:
    verification_by_id = {row["sample_id"]: row for row in verification_rows}
    queue = []
    for base in base_rows:
        verification = verification_by_id.get(base["sample_id"])
        if verification is None:
            score, reasons = 200, ["missing_vlm_verification"]
        else:
            score, reasons = review_priority(verification)
        queue.append(
            {
                "priority": score,
                "sample_id": base["sample_id"],
                "split": base["split"],
                "source_activity": base["source_activity"],
                "subject_id": base.get("subject_id", "unknown"),
                "image": base["image"],
                "source_video": base["source_video"],
                "review_reason": ";".join(reasons),
            }
        )
    queue.sort(key=lambda row: (-row["priority"], row["sample_id"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(queue[0]) if queue else [
        "priority", "sample_id", "split", "source_activity", "subject_id", "image", "source_video", "review_reason"
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)
    return queue

