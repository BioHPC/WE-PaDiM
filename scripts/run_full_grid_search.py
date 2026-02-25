"""Runner script for launching a full WE-PaDiM wavelet grid search.

This utility accepts a backbone model name, dataset name, and optional
class subset, builds a Config object with the standard wavelet grid, and
invokes run_experiment to execute the sweep.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config import (
    Config,
    AVAILABLE_MODELS,
    MVTEC_CLASSES,
    VISA_CLASSES,
)
from main import run_experiment

DATASET_REGISTRY = {
    "mvtec": {
        "data_path": "./data/MVTec",
        "dataset_type": "mvtec",
        "default_classes": MVTEC_CLASSES,
    },
    "visa": {
        "data_path": "./data/VisA",
        "dataset_type": "visa",
        "default_classes": VISA_CLASSES,
    },
}

DEFAULT_WAVELET_GRID = {
    "wavelet_type": ["haar", "sym4"],
    "wavelet_level": [1, 2],
    "wavelet_kept_subbands": [["LL", "LH", "HL"], ["LL", "LH", "HL", "HH"]],
    "sigma": [2.0, 4.0],
    "cov_reg": [0.01, 0.1],
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standard WE-PaDiM wavelet grid search for a dataset/model pair",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=AVAILABLE_MODELS.keys(),
        help="Backbone model to evaluate (e.g., efficientnet-b0)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_REGISTRY.keys(),
        help="Dataset to evaluate (mvtec or visa)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional subset of class names; defaults to all classes in the dataset",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Override dataset root path (defaults to ./data/<Dataset>)",
    )
    parser.add_argument(
        "--save_path",
        default="./results/gridsearch_runs",
        help="Base directory for saving results",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id to use (-1 for CPU)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=32,
        help="Test batch size",
    )
    try:  # prefer booleanoptionalaction when available (python 3.9+)
        parser.add_argument(
            "--save_visualizations",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Toggle saving per-image anomaly visualizations",
        )
        parser.add_argument(
            "--enable_resource_monitoring",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Toggle resource monitoring during the run",
        )
    except AttributeError:
        parser.add_argument(
            "--save_visualizations",
            action="store_true",
            default=False,
            help="Enable saving per-image anomaly visualizations",
        )
        parser.add_argument(
            "--no_save_visualizations",
            dest="save_visualizations",
            action="store_false",
        )
        parser.add_argument(
            "--enable_resource_monitoring",
            action="store_true",
            default=True,
            help="Enable resource monitoring",
        )
        parser.add_argument(
            "--no_enable_resource_monitoring",
            dest="enable_resource_monitoring",
            action="store_false",
        )
    return parser.parse_args()

def resolve_classes(dataset_key: str, requested: Optional[List[str]]) -> Optional[List[str]]:
    if requested:
        return requested
    return None  # allow config to auto-discover all classes

def main() -> None:
    args = parse_args()
    dataset_key = args.dataset.lower()
    dataset_info = DATASET_REGISTRY[dataset_key]

    data_path = args.data_path or dataset_info["data_path"]
    dataset_type = dataset_info["dataset_type"]
    classes = resolve_classes(dataset_key, args.classes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{dataset_key}_{args.model}_gridsearch_{timestamp}"
    full_save_path = os.path.join(args.save_path, run_dir)

    os.makedirs(full_save_path, exist_ok=True)

    config_args = argparse.Namespace(
        data_path=data_path,
        save_path=full_save_path,
        models=[args.model],
        classes=classes,
        experiment_type="grid_search",
        experiment_group="main_comparison",
        wavelet_type=DEFAULT_WAVELET_GRID["wavelet_type"],
        wavelet_level=DEFAULT_WAVELET_GRID["wavelet_level"],
        wavelet_kept_subbands=DEFAULT_WAVELET_GRID["wavelet_kept_subbands"],
        sigma=DEFAULT_WAVELET_GRID["sigma"],
        cov_reg=DEFAULT_WAVELET_GRID["cov_reg"],
        gpu_id=args.gpu_id,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        dataset_type=dataset_type,
        save_visualizations=args.save_visualizations,
        enable_resource_monitoring=args.enable_resource_monitoring,
        memory_efficient=True,
    )

    config = Config(config_args)
    config.experiment_type = "grid_search"

    print("=== Launching WE-PaDiM grid search ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_key} ({dataset_type})")
    print(f"Data path: {data_path}")
    if classes:
        print(f"Classes: {classes}")
    else:
        print("Classes: all (auto-detected)")
    print(f"Results directory: {full_save_path}")

    run_experiment(config)

if __name__ == "__main__":
    main()
