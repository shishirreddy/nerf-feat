#!/usr/bin/env bash
# run_pipeline.sh — Full NeRF-Feat training pipeline for a list of objects.
#
# Usage:
#   ./run_pipeline.sh [--config configs/tless.yaml] [object_id ...]
#
# Examples:
#   ./run_pipeline.sh                     # all 15 LM objects, lm.yaml
#   ./run_pipeline.sh 1 5 10              # specific objects
#   ./run_pipeline.sh --config configs/tless.yaml 1 2 3
set -euo pipefail

CONFIG="configs/lm.yaml"
if [[ "${1:-}" == "--config" ]]; then
    CONFIG="$2"
    shift 2
fi

if [[ $# -gt 0 ]]; then
    OBJECT_IDS=("$@")
else
    OBJECT_IDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
fi

# ── Stage 1: Train radiance field ─────────────────────────────────────────────
echo "=== Stage 1: train_nerf ==="
for id in "${OBJECT_IDS[@]}"; do
    echo "--- object $id ---"
    python train_nerf.py --config "$CONFIG" --object-id "$id"
done

# ── Stage 2: Generate surface correspondences ─────────────────────────────────
echo "=== Stage 2: generate_correspondences ==="
for id in "${OBJECT_IDS[@]}"; do
    echo "--- object $id ---"
    python generate_correspondences.py --config "$CONFIG" --object-id "$id"
done

# ── Stage 3: Train pose encoder ───────────────────────────────────────────────
echo "=== Stage 3: train_pose ==="
for id in "${OBJECT_IDS[@]}"; do
    echo "--- object $id ---"
    python train_pose.py --config "$CONFIG" --object-id "$id"
done

# ── Stage 4: Export surface features (used at inference time) ─────────────────
echo "=== Stage 4: export_features ==="
for id in "${OBJECT_IDS[@]}"; do
    echo "--- object $id ---"
    python export_features.py --config "$CONFIG" --object-id "$id"
done

echo "=== Pipeline complete ==="
