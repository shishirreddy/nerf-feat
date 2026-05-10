
import argparse
from dataclasses import dataclass
from pathlib import Path

from nerffeat.config import (
    DEFAULT_COCO_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MASK_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPLIT_ROOT,
    add_config_arg,
    load_config,
    resolve_path,
    section,
)


@dataclass(frozen=True)
class ExperimentConfig:
    bootstrap_parser: argparse.ArgumentParser
    config: dict
    paths: dict
    section: dict


def load_experiment_config(section_name: str) -> ExperimentConfig:
    """Load config early so argparse defaults can come from YAML."""
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    add_config_arg(bootstrap_parser)
    config_args, _ = bootstrap_parser.parse_known_args()
    config = load_config(config_args.config)
    return ExperimentConfig(
        bootstrap_parser=bootstrap_parser,
        config=config,
        paths=section(config, "paths"),
        section=section(config, section_name),
    )


def dataset_root(args, paths: dict) -> str:
    return resolve_path(args.dataset_root, Path(paths.get("dataset_root", DEFAULT_DATASET_ROOT)))


def output_root(args, paths: dict) -> str:
    return resolve_path(args.output_root, Path(paths.get("output_root", DEFAULT_OUTPUT_ROOT)))


def coco_root(args, paths: dict) -> str:
    return resolve_path(args.coco_root, Path(paths.get("coco_root", DEFAULT_COCO_ROOT)))


def mask_root(args, paths: dict) -> Path:
    return Path(resolve_path(args.mask_root, Path(paths.get("mask_root", DEFAULT_MASK_ROOT))))


def split_file(args, object_id: str, paths: dict) -> Path:
    """Return the split file path: CLI arg if given, else the committed default."""
    if args.split_file is not None:
        return Path(args.split_file)
    root = Path(paths.get("split_root", DEFAULT_SPLIT_ROOT))
    return root / str(object_id) / "train_ids.npy"
