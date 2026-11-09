#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <config.yaml> <video-dir> <work-dir> [device]" >&2
  exit 2
fi

CONFIG="$1"
VIDEO_DIR="$2"
WORK_DIR="$3"
DEVICE="${4:-auto}"

home-llm-label build \
  --config "${CONFIG}" \
  --video-dir "${VIDEO_DIR}" \
  --output-dir "${WORK_DIR}" \
  --skip-existing-images

home-llm-label verify \
  --manifest "${WORK_DIR}/manifests/clips.jsonl" \
  --output "${WORK_DIR}/label_review/vlm_verification.jsonl" \
  --device "${DEVICE}"

home-llm-label review-queue \
  --manifest "${WORK_DIR}/manifests/clips.jsonl" \
  --verification "${WORK_DIR}/label_review/vlm_verification.jsonl" \
  --output "${WORK_DIR}/label_review/review_queue.csv"

echo "Review ${WORK_DIR}/label_review/review_queue.csv and create manual_labels.jsonl."
echo "Then run:"
echo "home-llm-label finalize --config ${CONFIG} --manifest ${WORK_DIR}/manifests/clips.jsonl --manual-labels ${WORK_DIR}/label_review/manual_labels.jsonl --output-dir ${WORK_DIR}/final"

