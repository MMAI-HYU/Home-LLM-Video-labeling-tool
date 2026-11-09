from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .build import build_dataset
from .finalize import finalize_dataset
from .io_utils import read_jsonl
from .review import create_review_queue
from .schema import load_config
from .validate import validate_rows
from .verify import QwenVLBackend, verify_manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="home-llm-label", description="Home LLM video labeling pipeline")
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="sample videos and create initial labels")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--video-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--clips-per-class", type=int, help="override the YAML sampling count")
    build.add_argument("--skip-existing-images", action="store_true")

    verify = commands.add_parser("verify", help="run two-pass Qwen-VL visual verification")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    verify.add_argument("--device", default="auto")
    verify.add_argument("--max-new-tokens", type=int, default=192)

    review = commands.add_parser("review-queue", help="prioritize samples for human review")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--verification", type=Path, help="optional two-pass VLM output")
    review.add_argument("--output", type=Path, required=True)

    finalize = commands.add_parser("finalize", help="merge human labels and create final manifests")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--manual-labels", type=Path, help="optional human-review JSONL")
    finalize.add_argument("--output-dir", type=Path, required=True)

    validate = commands.add_parser("validate", help="check schema, duplicates, leakage, and images")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--check-images", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        config = load_config(args.config)
        if args.clips_per_class is not None:
            if args.clips_per_class <= 0:
                raise SystemExit("--clips-per-class must be positive")
            config["clips_per_class"] = args.clips_per_class
        rows = build_dataset(config, args.video_dir, args.output_dir, args.skip_existing_images)
        print(f"built clips={len(rows)} output={args.output_dir}")
        return
    if args.command == "verify":
        rows = read_jsonl(args.manifest)
        backend = QwenVLBackend(args.model_id, args.device, args.max_new_tokens)
        results = verify_manifest(rows, backend, args.output)
        failures = sum(not item["image_only"]["ok"] or not item["activity_conditioned"]["ok"] for item in results)
        print(f"verified clips={len(results)} samples_with_failure={failures} output={args.output}")
        return
    if args.command == "review-queue":
        verification = read_jsonl(args.verification) if args.verification else []
        queue = create_review_queue(read_jsonl(args.manifest), verification, args.output)
        print(f"review_queue clips={len(queue)} output={args.output}")
        return
    if args.command == "finalize":
        config = load_config(args.config)
        rows = finalize_dataset(
            read_jsonl(args.manifest),
            read_jsonl(args.manual_labels, required=False) if args.manual_labels else [],
            config["questions"],
            args.output_dir,
        )
        reviewed = sum(row["human_review"]["reviewed"] for row in rows)
        print(f"finalized clips={len(rows)} human_reviewed={reviewed} output={args.output_dir}")
        return
    if args.command == "validate":
        rows = read_jsonl(args.manifest)
        errors = validate_rows(rows, args.check_images)
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        print(
            json.dumps(
                {"ok": True, "clips": len(rows), "splits": Counter(row["split"] for row in rows)},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

