from __future__ import annotations

import json
import random
from pathlib import Path

from home_llm_labeling.build import allocate_splits, choose_diverse
from home_llm_labeling.finalize import finalize_dataset
from home_llm_labeling.review import create_review_queue, review_priority
from home_llm_labeling.schema import normalize_answer
from home_llm_labeling.validate import validate_rows
from home_llm_labeling.verify import compare_passes, parse_json_object


def answer():
    return {
        "activity": "read_book",
        "posture": "sitting",
        "mobility": "stationary",
        "object_interaction": ["book_or_document"],
        "scene_context": "table_area",
        "attention_state": "focused",
        "interruptibility": "low",
        "service_relevance": ["lighting"],
        "risk_level": "normal",
    }


def base_row(sample_id="TSH_READBOOK_0001", split="train"):
    return {
        "sample_id": sample_id,
        "source_video": f"/videos/{sample_id}.mp4",
        "image": f"/images/{sample_id}.jpg",
        "split": split,
        "source_activity": "Readbook",
        "subject_id": "01",
        "answer": answer(),
        "label_source": {field: "rule" for field in answer()} | {"gold_verified": False},
        "review_flag": {"needs_human_review": True, "reason": ["visual_fields_unverified"]},
    }


def test_split_allocation_is_exact():
    result = allocate_splits(56, {"train": 0.7, "val": 0.1, "test": 0.2})
    assert result == {"train": 39, "val": 5, "test": 12}
    assert sum(result.values()) == 56


def test_subject_round_robin_sampling(tmp_path: Path):
    files = []
    for subject in range(3):
        for clip in range(4):
            path = tmp_path / f"Readbook_p{subject:02d}_r00_v{clip:02d}_c01.mp4"
            path.touch()
            files.append(path)
    selected = choose_diverse(files, 6, random.Random(7))
    assert len(selected) == 6
    assert len({path.name.split("_p")[1][:2] for path in selected[:3]}) == 3


def test_vlm_json_parse_and_disagreement():
    left = parse_json_object(
        '```json\n{"activity":"read_book","posture":"sitting","mobility":"stationary",'
        '"object_interaction":["book_or_document"],"scene_context":"table_area"}\n```'
    )
    right = dict(left, posture="standing")
    result = compare_passes({"ok": True, "parsed": left}, {"ok": True, "parsed": right})
    assert result == ["posture"]


def test_review_priority_and_csv(tmp_path: Path):
    verification = {
        "sample_id": "TSH_READBOOK_0001",
        "image_only": {"ok": False, "parsed": None},
        "activity_conditioned": {"ok": True, "parsed": {"activity": "read_book"}},
        "disagreement_fields": ["activity", "posture"],
    }
    score, reasons = review_priority(verification)
    assert score >= 100
    assert any("parse_failure" in reason for reason in reasons)
    output = tmp_path / "queue.csv"
    queue = create_review_queue([base_row()], [verification], output)
    assert queue[0]["sample_id"] == "TSH_READBOOK_0001"
    assert output.exists()


def test_finalize_and_validate(tmp_path: Path):
    row = base_row()
    manual = {
        "sample_id": row["sample_id"],
        "source_video": row["source_video"],
        "answer": {
            "posture": "sitting",
            "mobility": "stationary",
            "object_interaction": ["book_or_document"],
            "scene_context": "sofa_area",
        },
        "gold_verified": True,
        "reviewer": "tester",
    }
    questions = [{"field": "activity", "answer_type": "single_choice", "text": "Activity?"}]
    final = finalize_dataset([row], [manual], questions, tmp_path / "final")
    assert final[0]["human_review"]["reviewed"] is True
    assert final[0]["answer"]["scene_context"] == "sofa_area"
    assert validate_rows(final) == []
    assert (tmp_path / "final" / "DATA_REPORT.md").exists()


def test_duplicate_source_is_rejected():
    first = base_row("A", "train")
    second = base_row("B", "test")
    second["source_video"] = first["source_video"]
    errors = validate_rows([first, second])
    assert any("duplicate source_video" in error for error in errors)
    assert any("leakage" in error for error in errors)

