from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .build import status_row, vqa_rows
from .io_utils import write_jsonl, write_text
from .schema import SPLITS, VISUAL_FIELDS, normalize_answer


def apply_manual_label(base: dict[str, Any], manual: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    manual_answer = manual.get("answer", {})
    missing = set(VISUAL_FIELDS) - set(manual_answer)
    if missing:
        raise ValueError(f"Manual label {base['sample_id']} is missing visual fields: {sorted(missing)}")
    for field in VISUAL_FIELDS:
        result["answer"][field] = manual_answer[field]
        result["label_source"][field] = "human_verified"
    result["answer"] = normalize_answer(result["answer"])
    result["label_source"]["gold_verified"] = bool(manual.get("gold_verified", False))
    result["review_flag"] = {"needs_human_review": False, "reason": []}
    result["human_review"] = {
        "reviewed": True,
        "reviewer": manual.get("reviewer", "unspecified"),
        "gold_verified": bool(manual.get("gold_verified", False)),
        "notes": manual.get("notes", ""),
    }
    return result


def untouched_label(base: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    result["answer"] = normalize_answer(result["answer"])
    result["human_review"] = {
        "reviewed": False,
        "reviewer": None,
        "gold_verified": False,
        "notes": "",
    }
    return result


def data_report(rows: list[dict[str, Any]], question_count: int) -> str:
    split_counts = Counter(row["split"] for row in rows)
    class_counts = Counter(row["source_activity"] for row in rows)
    reviewed = sum(bool(row["human_review"]["reviewed"]) for row in rows)
    lines = [
        "# Home LLM Labeling Data Report",
        "",
        "## Summary",
        "",
        f"- Clips: {len(rows)}",
        f"- VQA pairs: {len(rows) * question_count}",
        f"- Structured JSON samples: {len(rows)}",
        f"- Human-reviewed clips: {reviewed}",
        f"- Pending-review clips: {len(rows) - reviewed}",
        "",
        "## Splits",
        "",
        "| Split | Clips | VQA | JSON |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        count = split_counts[split]
        lines.append(f"| {split} | {count} | {count * question_count} | {count} |")
    lines.extend(["", "## Source classes", "", "| Class | Clips |", "|---|---:|"])
    for source_class in sorted(class_counts):
        lines.append(f"| {source_class} | {class_counts[source_class]} |")
    lines.extend(
        [
            "",
            "## Provenance policy",
            "",
            "- Activity is derived from the source class or filename.",
            "- Visual fields are marked `human_verified` only when a matching manual record exists.",
            "- Unreviewed visual fields remain rule-prefilled and explicitly flagged for review.",
            "- Service and risk fields remain activity-rule derived.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_dataset(
    base_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_source: dict[str, dict[str, Any]] = {}
    for row in manual_rows:
        sample_id = row.get("sample_id")
        source_video = row.get("source_video")
        if not sample_id and not source_video:
            raise ValueError("Every manual label needs sample_id or source_video")
        if sample_id:
            if sample_id in by_id:
                raise ValueError(f"Duplicate manual sample_id: {sample_id}")
            by_id[sample_id] = row
        if source_video:
            if source_video in by_source:
                raise ValueError(f"Duplicate manual source_video: {source_video}")
            by_source[source_video] = row

    base_ids = {row["sample_id"] for row in base_rows}
    base_sources = {row["source_video"] for row in base_rows}
    orphan_ids = set(by_id) - base_ids
    orphan_sources = set(by_source) - base_sources
    if orphan_ids:
        raise ValueError(f"Manual labels contain unknown sample IDs: {sorted(orphan_ids)[:5]}")
    if orphan_sources:
        raise ValueError(f"Manual labels contain unknown source videos: {sorted(orphan_sources)[:5]}")
    final_rows = []
    for base in base_rows:
        manual = by_id.get(base["sample_id"])
        if manual and manual.get("source_video") not in (None, base["source_video"]):
            raise ValueError(f"Source mismatch for manual label {base['sample_id']}")
        manual = manual or by_source.get(base["source_video"])
        final_rows.append(apply_manual_label(base, manual) if manual else untouched_label(base))

    write_jsonl(output_dir / "clips.final.jsonl", final_rows)
    for split in SPLITS:
        rows = [row for row in final_rows if row["split"] == split]
        write_jsonl(
            output_dir / f"vqa_{split}.final.jsonl",
            (item for row in rows for item in vqa_rows(row, questions)),
        )
        write_jsonl(output_dir / f"status_{split}.final.jsonl", (status_row(row) for row in rows))
    write_text(output_dir / "DATA_REPORT.md", data_report(final_rows, len(questions)))
    return final_rows

