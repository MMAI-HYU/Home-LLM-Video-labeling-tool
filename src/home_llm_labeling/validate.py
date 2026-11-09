from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schema import ANSWER_FIELDS, SPLITS, normalize_answer


def validate_rows(rows: list[dict[str, Any]], check_images: bool = False) -> list[str]:
    errors: list[str] = []
    ids = [row.get("sample_id") for row in rows]
    sources = [row.get("source_video") for row in rows]
    images = [row.get("image") for row in rows]
    for label, values in (("sample_id", ids), ("source_video", sources), ("image", images)):
        duplicates = [value for value, count in Counter(values).items() if value is not None and count > 1]
        if duplicates:
            errors.append(f"duplicate {label}: {duplicates[:5]}")

    sources_by_split: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows, start=1):
        prefix = row.get("sample_id", f"row:{index}")
        missing = {"sample_id", "source_video", "image", "split", "source_activity", "answer"} - set(row)
        if missing:
            errors.append(f"{prefix}: missing keys {sorted(missing)}")
            continue
        if row["split"] not in SPLITS:
            errors.append(f"{prefix}: invalid split {row['split']!r}")
        else:
            sources_by_split[row["split"]].add(row["source_video"])
        try:
            normalize_answer(row["answer"])
        except Exception as error:
            errors.append(f"{prefix}: invalid answer schema: {error}")
        if check_images:
            image = Path(row["image"])
            if not image.is_file():
                errors.append(f"{prefix}: missing image {image}")
            else:
                try:
                    from PIL import Image

                    with Image.open(image) as handle:
                        handle.verify()
                except Exception as error:
                    errors.append(f"{prefix}: invalid image: {error}")

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = sources_by_split[left] & sources_by_split[right]
            if overlap:
                errors.append(f"source-video leakage {left}/{right}: {sorted(overlap)[:5]}")
    return errors

