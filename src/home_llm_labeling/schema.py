from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ANSWER_FIELDS = (
    "activity",
    "posture",
    "mobility",
    "object_interaction",
    "scene_context",
    "attention_state",
    "interruptibility",
    "service_relevance",
    "risk_level",
)
VISUAL_FIELDS = ("posture", "mobility", "object_interaction", "scene_context")
LIST_FIELDS = ("object_interaction", "service_relevance")
SPLITS = ("train", "val", "test")


class ConfigurationError(ValueError):
    """Raised when a pipeline configuration is invalid."""


def normalize_list(value: Any, default: str | None = None) -> list[str]:
    if value is None:
        values: list[Any] = []
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    normalized = sorted({str(item).strip() for item in values if str(item).strip()})
    if not normalized and default is not None:
        return [default]
    if "unknown" in normalized and len(normalized) > 1:
        normalized.remove("unknown")
    return normalized


def normalize_answer(answer: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(answer)
    for field in ANSWER_FIELDS:
        if field not in result:
            raise ConfigurationError(f"Missing answer field: {field}")
    for field in LIST_FIELDS:
        result[field] = normalize_list(result[field], "unknown" if field == "object_interaction" else None)
    for field in set(ANSWER_FIELDS) - set(LIST_FIELDS):
        result[field] = str(result[field]).strip()
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("The YAML root must be an object")
    if not isinstance(config.get("classes"), dict) or not config["classes"]:
        raise ConfigurationError("At least one class rule is required")
    if not isinstance(config.get("questions"), list) or not config["questions"]:
        raise ConfigurationError("At least one VQA question is required")

    ratios = config.get("split_ratios", {})
    if set(ratios) != set(SPLITS):
        raise ConfigurationError(f"split_ratios must contain exactly {SPLITS}")
    if any(float(ratios[name]) < 0 for name in SPLITS):
        raise ConfigurationError("Split ratios cannot be negative")
    if abs(sum(float(ratios[name]) for name in SPLITS) - 1.0) > 1e-8:
        raise ConfigurationError("Split ratios must sum to 1.0")

    for source_class, answer in config["classes"].items():
        try:
            config["classes"][source_class] = normalize_answer(answer)
        except ConfigurationError as error:
            raise ConfigurationError(f"Class {source_class}: {error}") from error
    for index, question in enumerate(config["questions"], start=1):
        required = {"field", "answer_type", "text"}
        missing = required - set(question)
        if missing:
            raise ConfigurationError(f"Question {index} missing: {sorted(missing)}")
        if question["field"] not in ANSWER_FIELDS:
            raise ConfigurationError(f"Question {index} has unknown field {question['field']}")
    return config

