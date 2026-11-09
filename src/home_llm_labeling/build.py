from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .frames import make_temporal_grid
from .io_utils import write_jsonl, write_text
from .schema import SPLITS


SUBJECT_PATTERN = re.compile(r"_p(?P<subject>\d+)_", re.IGNORECASE)


def subject_id(path: Path) -> str:
    match = SUBJECT_PATTERN.search(path.name)
    return match.group("subject") if match else "unknown"


def choose_diverse(files: list[Path], count: int, rng: random.Random) -> list[Path]:
    by_subject: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_subject[subject_id(path)].append(path)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    for paths in by_subject.values():
        rng.shuffle(paths)

    selected: list[Path] = []
    active = subjects
    while active and len(selected) < count:
        remaining = []
        for subject in active:
            if by_subject[subject] and len(selected) < count:
                selected.append(by_subject[subject].pop())
            if by_subject[subject]:
                remaining.append(subject)
        active = remaining
    return selected


def allocate_splits(count: int, ratios: dict[str, float]) -> dict[str, int]:
    # Match the original dataset convention: floor train/validation and place
    # the residual in test so every selected clip is assigned exactly once.
    train = int(count * float(ratios["train"]))
    validation = int(count * float(ratios["val"]))
    return {"train": train, "val": validation, "test": count - train - validation}


def sample_identifier(source_class: str, global_index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", source_class).strip("_").upper()
    return f"TSH_{slug}_{global_index:04d}"


def vqa_rows(row: dict[str, Any], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, question in enumerate(questions, start=1):
        output.append(
            {
                "sample_id": f"{row['sample_id']}_Q{index}",
                "clip_sample_id": row["sample_id"],
                "image": row["image"],
                "source_video": row["source_video"],
                "question": question["text"],
                "answer_type": question["answer_type"],
                "answer": row["answer"][question["field"]],
                "split": row["split"],
                "source_activity": row["source_activity"],
                "label_source": row["label_source"], 
            }
        )
    return output


def status_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": f"{row['sample_id']}_STATUS",
        "clip_sample_id": row["sample_id"],
        "image": row["image"],
        "source_video": row["source_video"],
        "question": "Return the current user status as JSON.",
        "answer_type": "structured_json",
        "answer": row["answer"],
        "split": row["split"],
        "source_activity": row["source_activity"],
        "label_source": row["label_source"],
    }


def build_dataset(
    config: dict[str, Any],
    video_dir: Path,
    output_dir: Path,
    skip_existing_images: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(int(config.get("seed", 0)))
    clips_per_class = int(config["clips_per_class"])
    ratios = {name: float(config["split_ratios"][name]) for name in SPLITS}
    records = []
    global_index = 1

    for source_class, rule_answer in config["classes"].items():
        files = sorted(video_dir.glob(f"{source_class}_*.mp4"))
        if len(files) < clips_per_class:
            raise RuntimeError(
                f"Class {source_class} needs {clips_per_class} clips, but only {len(files)} were found"
            )
        selected = choose_diverse(files, clips_per_class, rng)
        split_counts = allocate_splits(len(selected), ratios)
        split_sequence = [split for split in SPLITS for _ in range(split_counts[split])]
        for video, split in zip(selected, split_sequence):
            sample_id = sample_identifier(source_class, global_index)
            image_path = output_dir / "images" / f"{sample_id}_grid.jpg"
            if not (skip_existing_images and image_path.exists()):
                make_temporal_grid(
                    video,
                    image_path,
                    [float(value) for value in config.get("frame_ratios", [0.2, 0.4, 0.6, 0.8])],
                    int(config.get("grid_cell_size", 448)),
                    int(config.get("jpeg_quality", 92)),
                )
            records.append(
                {
                    "sample_id": sample_id,
                    "source_video": str(video.resolve()),
                    "image": str(image_path.resolve()),
                    "split": split,
                    "source_activity": source_class,
                    "subject_id": subject_id(video),
                    "answer": deepcopy(rule_answer),
                    "label_source": {
                        "activity": "source_filename",
                        "posture": "rule_prefill_needs_human_check",
                        "mobility": "rule_prefill_needs_human_check",
                        "object_interaction": "rule_prefill_needs_human_check",
                        "scene_context": "rule_prefill_needs_human_check",
                        "attention_state": "activity_rule",
                        "interruptibility": "activity_rule",
                        "service_relevance": "activity_rule",
                        "risk_level": "activity_rule",
                        "gold_verified": False,
                    },
                    "review_flag": {
                        "needs_human_review": True,
                        "reason": ["visual_fields_unverified"],
                    },
                }
            )
            global_index += 1

    manifest_dir = output_dir / "manifests"
    write_jsonl(manifest_dir / "clips.jsonl", records)
    for split in SPLITS:
        split_rows = [row for row in records if row["split"] == split]
        write_jsonl(
            manifest_dir / f"vqa_{split}.jsonl",
            (item for row in split_rows for item in vqa_rows(row, config["questions"])),
        )
        write_jsonl(manifest_dir / f"status_{split}.jsonl", (status_row(row) for row in split_rows))

    metadata = {
        "pipeline_version": "1.0.0",
        "seed": int(config.get("seed", 0)),
        "video_dir": str(video_dir.resolve()),
        "clip_count": len(records),
        "config": config,
    }
    write_text(output_dir / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    write_text(output_dir / "config.resolved.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return records

