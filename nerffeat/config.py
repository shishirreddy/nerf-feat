import argparse
import logging
import os
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ROOT = Path(os.environ.get("NERFFEAT_DATASET_ROOT", "bop/lm"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("NERFFEAT_OUTPUT_ROOT", "lm_out"))
DEFAULT_COCO_ROOT = Path(os.environ.get("NERFFEAT_COCO_ROOT", "data/coco"))
DEFAULT_MASK_ROOT = Path(os.environ.get("NERFFEAT_MASK_ROOT", "segmentation_mask"))
DEFAULT_SPLIT_ROOT = Path(os.environ.get("NERFFEAT_SPLIT_ROOT", "splits"))
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def configure_logger(name: str, level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML experiment configuration file.",
    )


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    import yaml

    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def section(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Read a nested config key, returning {} if absent."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, dict) else {}


def str2bool(value: object) -> bool:
    """Accept true/false/1/0/yes/no from CLI flags."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def resolve_path(path: str | os.PathLike[str] | None, default: Path) -> str:
    raw = Path(path) if path is not None else default
    return str(raw.expanduser().resolve())


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", default=None, help="BOP dataset root.")
    parser.add_argument("--output-root", default=None, help="Output root for checkpoints and previews.")
    parser.add_argument(
        "--mask-root",
        dest="mask_root",
        default=None,
        metavar="PATH",
        help="Predicted mask root. Resolved as {mask_root}/{dataset_name}/scene{object_id}/{image_id:06d}.png.",
    )


def add_split_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-file",
        dest="split_file",
        default=None,
        metavar="PATH",
        help=(
            "Path to train_ids.npy. "
            "Defaults to splits/<object-id>/train_ids.npy (committed to the repo). "
            "Training: load only these IDs. Inference: skip these IDs."
        ),
    )


def add_object_args(parser: argparse.ArgumentParser, default_object_id: str = "15") -> None:
    parser.add_argument("--object-id", default=default_object_id, help="BOP object id.")
