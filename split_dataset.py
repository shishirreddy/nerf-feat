"""Generate a reproducible train/eval image-ID split for one BOP object.

Scans the scene_camera.json of the requested object, randomly draws
``--train-count`` frames for training and assigns the rest to evaluation.
Both sets are written as NumPy arrays so every downstream script can load
them with a single ``np.load()`` call.

Output files
------------
  splits/<object-id>/train_ids.npy   (committed to the repo)
  splits/<object-id>/eval_ids.npy

Usage examples
--------------
  # 200-image training split using lm.yaml defaults
  python split_dataset.py --config configs/lm.yaml --object-id 1 --train-count 200

  # 150-image split, explicit paths, different seed
  python split_dataset.py --object-id 5 --train-count 150 --seed 0 \\
      --dataset-root bop/lm
"""
import argparse
import json
from pathlib import Path

import numpy as np

from nerffeat import cli
from nerffeat.config import (
    DEFAULT_SPLIT_ROOT,
    add_common_dataset_args,
    configure_logger,
)

EXPERIMENT_CONFIG = cli.load_experiment_config("split")

arg_parser = argparse.ArgumentParser(
    description="Create a random train/eval frame split for one BOP object",
    parents=[EXPERIMENT_CONFIG.bootstrap_parser],
)
arg_parser.add_argument(
    "--object-id",
    dest="object_id",
    required=True,
    help="BOP object ID (e.g. 1 for Linemod ape).",
)
arg_parser.add_argument(
    "--train-count",
    dest="train_count",
    type=int,
    default=200,
    help="Number of frames reserved for training (default: 200). Ignored when --all is set.",
)
arg_parser.add_argument(
    "--all",
    dest="all_images",
    action="store_true",
    help="Use all available frames as training IDs (e.g. T-LESS real training split).",
)
arg_parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42).",
)
arg_parser.add_argument(
    "--split-name",
    dest="split_name",
    default=EXPERIMENT_CONFIG.section.get("split_name", "train"),
    help="Subdirectory under --dataset-root that holds the object frames (default: train).",
)
arg_parser.add_argument(
    "--split-root",
    dest="split_root",
    default=None,
    metavar="PATH",
    help=f"Directory to write splits into (default: {DEFAULT_SPLIT_ROOT}).",
)
add_common_dataset_args(arg_parser)
args = arg_parser.parse_args()

LOGGER = configure_logger(__name__)

dataset_root = Path(cli.dataset_root(args, EXPERIMENT_CONFIG.paths))
object_id = str(args.object_id)
object_id_padded = object_id.zfill(6)

object_dir = dataset_root / args.split_name / object_id_padded
camera_json = object_dir / "scene_camera.json"
if not camera_json.exists():
    raise FileNotFoundError(
        f"scene_camera.json not found at {camera_json}. "
        "Check --dataset-root and --split-name."
    )

with open(camera_json, encoding="utf-8") as f:
    camera_data = json.load(f)

all_ids = np.array(sorted(int(k) for k in camera_data))
total = len(all_ids)
LOGGER.info("object_id=%s  total_frames=%d", object_id, total)

if args.all_images:
    train_ids = all_ids
    eval_ids = np.array([], dtype=np.int64)
else:
    if args.train_count >= total:
        raise ValueError(
            f"--train-count {args.train_count} must be less than total frames {total}"
        )
    rng = np.random.default_rng(args.seed)
    train_ids = np.sort(rng.choice(all_ids, size=args.train_count, replace=False))
    eval_ids = np.array(sorted(set(all_ids.tolist()) - set(train_ids.tolist())))

split_root = Path(args.split_root) if args.split_root is not None else Path(
    EXPERIMENT_CONFIG.paths.get("split_root", DEFAULT_SPLIT_ROOT)
)
split_dir = split_root / object_id
split_dir.mkdir(parents=True, exist_ok=True)
np.save(split_dir / "train_ids.npy", train_ids)
np.save(split_dir / "eval_ids.npy", eval_ids)

LOGGER.info(
    "split saved  train=%d  eval=%d  seed=%d  path=%s",
    len(train_ids),
    len(eval_ids),
    args.seed,
    split_dir,
)
