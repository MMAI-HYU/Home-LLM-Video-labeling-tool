from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .io_utils import write_jsonl
from .schema import VISUAL_FIELDS, normalize_list


VERIFY_FIELDS = ("activity",) + VISUAL_FIELDS
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    match = JSON_OBJECT.search(candidate)
    if not match:
        raise ValueError("No JSON object found")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("VLM output is not a JSON object")
    missing = set(VERIFY_FIELDS) - set(value)
    if missing:
        raise ValueError(f"Missing verification fields: {sorted(missing)}")
    result = {field: str(value[field]).strip() for field in VERIFY_FIELDS if field != "object_interaction"}
    result["object_interaction"] = normalize_list(value["object_interaction"], "unknown")
    return result


def visual_prompt(source_activity: str | None) -> str:
    context = (
        "Do not assume an activity label; infer it only from the image."
        if source_activity is None
        else f"The source filename activity is {source_activity!r}. Use it only as context and correct it if visually inconsistent."
    )
    return (
        "Inspect the four chronological frames in this 2x2 image. "
        f"{context} Return only one JSON object with exactly these keys: "
        "activity, posture, mobility, object_interaction, scene_context. "
        "object_interaction must be a JSON array. Use 'unknown' when the image is insufficient."
    )


class QwenVLBackend:
    """Optional Qwen2.5-VL inference backend loaded only by the verify command."""

    def __init__(self, model_id: str, device: str = "auto", max_new_tokens: int = 192):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Install the optional VLM dependencies with: pip install -e '.[vlm]'") from error

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        load_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, image_path: Path, prompt: str) -> str:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def run_pass(backend: QwenVLBackend, image: Path, source_activity: str | None) -> dict[str, Any]:
    prompt = visual_prompt(source_activity)
    try:
        raw = backend.generate(image, prompt)
        parsed = parse_json_object(raw)
        return {"ok": True, "prompt_mode": "image_only" if source_activity is None else "conditioned", "raw": raw, "parsed": parsed}
    except Exception as error:
        return {
            "ok": False,
            "prompt_mode": "image_only" if source_activity is None else "conditioned",
            "raw": locals().get("raw", ""),
            "parsed": None,
            "error": f"{type(error).__name__}: {error}",
        }


def compare_passes(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    if not first["ok"] or not second["ok"]:
        return list(VERIFY_FIELDS)
    disagreements = []
    for field in VERIFY_FIELDS:
        left, right = first["parsed"][field], second["parsed"][field]
        if field == "object_interaction":
            if set(left) != set(right):
                disagreements.append(field)
        elif left != right:
            disagreements.append(field)
    return disagreements


def verify_manifest(rows: list[dict[str, Any]], backend: QwenVLBackend, output: Path) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows, start=1):
        image = Path(row["image"])
        image_only = run_pass(backend, image, None)
        conditioned = run_pass(backend, image, row["source_activity"])
        results.append(
            {
                "sample_id": row["sample_id"],
                "source_video": row["source_video"],
                "image": row["image"],
                "source_activity": row["source_activity"],
                "image_only": image_only,
                "activity_conditioned": conditioned,
                "disagreement_fields": compare_passes(image_only, conditioned),
            }
        )
        if index % 25 == 0:
            print(f"verified {index}/{len(rows)}", flush=True)
    write_jsonl(output, results)
    return results

