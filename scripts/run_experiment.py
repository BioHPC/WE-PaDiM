"""General runner for WE-PaDiM experiments.

Supports single-model runs, wavelet grid searches, and comprehensive grid searches.
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

from config import AVAILABLE_MODELS, Config
from main import run_experiment

def _parse_subband_sets(raw: Optional[List[str]]) -> Optional[List[List[str]]]:
    if not raw:
        return None
    parsed: List[List[str]] = []
    for item in raw:
        parts = [tok.strip().upper() for tok in item.split("|") if tok.strip()]
        if not parts:
            raise argparse.ArgumentTypeError(f"Invalid subband set: '{item}'")
        parsed.append(parts)
    return parsed

def _collapse_if_single(values, expect_list: bool):
    if expect_list:
        return values
    if isinstance(values, list) and len(values) == 1:
        return values[0]
    return values

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WE-PaDiM experiments")

    parser.add_argument("--data_path", default="./data/MVTec", help="Dataset root path")
    parser.add_argument("--dataset_type", choices=["mvtec", "visa", "auto"], default="auto")
    parser.add_argument("--save_path", default="./results", help="Base directory for results")
    parser.add_argument("--run_name", default=None, help="Optional run name (overrides timestamped default)")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18"],
        choices=list(AVAILABLE_MODELS.keys()),
        help="Backbone models to run",
    )
    parser.add_argument("--classes", nargs="+", default=None, help="Class subset (default: all)")

    parser.add_argument(
        "--experiment_type",
        choices=["single", "grid_search", "paper"],
        default="single",
        help="Experiment type",
    )
    parser.add_argument(
        "--experiment_group",
        default="main_comparison",
        help="Paper experiment group (paper mode only)",
    )
    parser.add_argument(
        "--comprehensive_grid_search",
        action="store_true",
        help="Run the comprehensive grid search sweep",
    )

    parser.add_argument("--wavelet_type", nargs="+", default=["haar"], help="Wavelet type(s)")
    parser.add_argument("--wavelet_level", nargs="+", type=int, default=[1], help="Wavelet level(s)")
    parser.add_argument(
        "--wavelet_kept_subbands",
        nargs="+",
        default=["LL", "LH", "HL"],
        help="Subbands to keep (single run)",
    )
    parser.add_argument(
        "--wavelet_subband_sets",
        nargs="+",
        default=None,
        help="Grid search subband sets (pipe-delimited, e.g. LL|LH|HL)",
    )
    parser.add_argument("--sigma", nargs="+", type=float, default=[4.0], help="Gaussian sigma")
    parser.add_argument("--cov_reg", nargs="+", type=float, default=[0.01], help="Covariance regularization")

    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id (-1 for CPU)")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)

    try:
        parser.add_argument(
            "--save_visualizations",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Save anomaly visualizations",
        )
        parser.add_argument(
            "--enable_resource_monitoring",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable resource monitoring",
        )
    except AttributeError:
        parser.add_argument("--save_visualizations", action="store_true", default=False)
        parser.add_argument("--no_save_visualizations", dest="save_visualizations", action="store_false")
        parser.add_argument("--enable_resource_monitoring", action="store_true", default=True)
        parser.add_argument("--no_enable_resource_monitoring", dest="enable_resource_monitoring", action="store_false")

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "_".join(args.models)
    run_name = args.run_name or f"{args.dataset_type}_{model_tag}_{args.experiment_type}_{timestamp}"
    full_save_path = os.path.join(args.save_path, run_name)
    os.makedirs(full_save_path, exist_ok=True)

    subband_sets = _parse_subband_sets(args.wavelet_subband_sets)
    wavelet_kept_subbands = subband_sets if subband_sets else args.wavelet_kept_subbands

    config_args = argparse.Namespace(
        data_path=args.data_path,
        save_path=full_save_path,
        models=args.models,
        classes=args.classes,
        experiment_type=args.experiment_type,
        experiment_group=args.experiment_group,
        wavelet_type=_collapse_if_single(args.wavelet_type, args.experiment_type == "grid_search"),
        wavelet_level=_collapse_if_single(args.wavelet_level, args.experiment_type == "grid_search"),
        wavelet_kept_subbands=wavelet_kept_subbands,
        sigma=_collapse_if_single(args.sigma, args.experiment_type == "grid_search"),
        cov_reg=_collapse_if_single(args.cov_reg, args.experiment_type == "grid_search"),
        gpu_id=args.gpu_id,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        dataset_type=args.dataset_type,
        save_visualizations=args.save_visualizations,
        enable_resource_monitoring=args.enable_resource_monitoring,
        memory_efficient=True,
    )

    config = Config(config_args)

    if args.experiment_type == "single":
        config.experiment_type = "single_model"
        config.gaussian_sigma = args.sigma[0] if isinstance(args.sigma, list) else args.sigma
        config.cov_reg_epsilon = args.cov_reg[0] if isinstance(args.cov_reg, list) else args.cov_reg

    if args.comprehensive_grid_search:
        config.comprehensive_grid_search = True
        config.grid_search_models = args.models
        config.grid_search_classes = args.classes

    print("=== Launching WE-PaDiM experiment ===")
    print(f"Dataset: {args.dataset_type} ({args.data_path})")
    print(f"Models: {args.models}")
    print(f"Experiment type: {config.experiment_type}")
    print(f"Save path: {full_save_path}")

    run_experiment(config)

if __name__ == "__main__":
    main()
