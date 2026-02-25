#!/usr/bin/env python3
"""Lightweight staged search for WE-PaDiM wavelet hyperparameters.

This runner avoids the full Cartesian grid by:
  1. Sampling a small random subset across the full parameter space (Stage 1).
  2. Running one-parameter-at-a-time sweeps around the best Stage 1 combos
     for both image-level and pixel-level AUC (Stage 2).

It saves incremental JSON artifacts so you can inspect partial progress and
produces a compact summary (best configs + per-parameter averages).
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import torch

from config import AVAILABLE_MODELS, MVTEC_CLASSES, VISA_CLASSES
from dataset import get_all_class_names
from evaluator import PaDiMEvaluator
from grid_search import run_single_wavelet_experiment
from models import FeatureExtractor
from utils import ResourceMonitor, save_results, set_gpu_environment

DEFAULT_WAVELET_TYPES = ["haar", "db2", "db4", "sym3", "sym4", "coif2"]
DEFAULT_WAVELET_LEVELS = [1, 2]
DEFAULT_SIGMA_VALUES = [1.5, 2.0, 3.0, 4.0, 6.0]
DEFAULT_COV_REG_VALUES = [0.001, 0.01, 0.05, 0.1]
DEFAULT_SUBBAND_OPTIONS = [
    "LL",
    "LL|LH",
    "LL|HL",
    "LL|LH|HL",
    "LL|LH|HL|HH",
    "LL|HL|HH",
]

def parse_subband_options(raw: Optional[Sequence[str]]) -> List[List[str]]:
    values = raw or DEFAULT_SUBBAND_OPTIONS
    parsed: List[List[str]] = []
    for item in values:
        parts = [tok.strip().upper() for tok in item.split("|") if tok.strip()]
        if not parts:
            raise ValueError(f"Invalid subband specification: '{item}'")
        parsed.append(parts)
    return parsed

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Staged wavelet parameter probe for WE-PaDiM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--dataset_type", choices=["mvtec", "visa"], required=True, help="Dataset flavor")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet-b0",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Backbone to evaluate",
    )
    parser.add_argument("--classes", nargs="*", default=None, help="Optional class subset")
    parser.add_argument("--save_path", type=str, default="./results/staged_wavelet_probe", help="Output directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id (-1 for CPU)")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--stage1_samples", type=int, default=10, help="Number of random combos in Stage 1")
    parser.add_argument("--stage1_seed", type=int, default=1024, help="Random seed for Stage 1 sampling")
    parser.add_argument("--stage2_topk", type=int, default=2, help="How many Stage 1 configs become Stage 2 anchors")
    parser.add_argument("--skip_stage2", action="store_true", help="Disable Stage 2 sweeps")
    parser.add_argument("--wavelet_types", nargs="*", default=None, help="Candidate wavelet types")
    parser.add_argument("--wavelet_levels", nargs="*", type=int, default=None, help="Candidate wavelet levels")
    parser.add_argument("--sigma_values", nargs="*", type=float, default=None, help="Candidate Gaussian sigmas")
    parser.add_argument("--cov_reg_values", nargs="*", type=float, default=None, help="Candidate covariance regularizers")
    parser.add_argument(
        "--subband_options",
        nargs="*",
        default=None,
        help="Candidate subband sets (pipe-delimited, e.g. 'LL|LH|HL')",
    )
    parser.add_argument("--save_visualizations", action="store_true", help="Store anomaly maps for each eval")
    parser.add_argument("--enable_resource_monitoring", action="store_true", help="Log per-eval resource usage")
    return parser

def resolve_class_list(args: argparse.Namespace) -> List[str]:
    if args.classes:
        return list(args.classes)
    default_fallback = MVTEC_CLASSES if args.dataset_type == "mvtec" else VISA_CLASSES
    try:
        return get_all_class_names(args.data_path, dataset_type=args.dataset_type)
    except Exception as exc:  pragma: no cover - defensive path
        print(f"Warning: falling back to canonical class list ({exc})")
        return default_fallback

def combo_key(params: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        params["wavelet_type"],
        params["wavelet_level"],
        tuple(params["wavelet_kept_subbands"]),
        float(params["sigma"]),
        float(params["cov_reg"]),
    )

def evaluate_params(
    evaluator: PaDiMEvaluator,
    feature_extractor: FeatureExtractor,
    classes: List[str],
    params: Dict[str, Any],
    args: argparse.Namespace,
    stage_dir: str,
    stage_label: str,
    resource_monitor: Optional[ResourceMonitor],
) -> Optional[Dict[str, Any]]:
    start = time.time()
    try:
        result = run_single_wavelet_experiment(
            evaluator=evaluator,
            feature_extractor=feature_extractor,
            class_names=classes,
            wavelet_type=params["wavelet_type"],
            wavelet_level=params["wavelet_level"],
            wavelet_kept_subbands=params["wavelet_kept_subbands"],
            sigma=params["sigma"],
            cov_reg=params["cov_reg"],
            save_path=stage_dir,
            save_visualizations=args.save_visualizations,
            train_batch_size=args.train_batch_size,
            resource_monitor=resource_monitor,
        )
    except Exception as exc:
        print(f"[WARN] Evaluation failure for {params}: {exc}")
        return None

    elapsed = time.time() - start
    enriched = {
        "stage": stage_label,
        "elapsed_sec": elapsed,
        "params": deepcopy(params),
        "avg_img_auc": result.get("avg_img_auc", 0.0),
        "avg_pixel_auc": result.get("avg_pixel_auc", 0.0),
        "model": feature_extractor.model_name,
        "class_count": len(classes),
    }
    return enriched

def aggregate_by_param(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    accum: Dict[str, Dict[Any, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"count": 0, "img_auc": 0.0, "pixel_auc": 0.0}))
    for entry in results:
        params = entry["params"]
        for key in ("wavelet_type", "wavelet_level", "wavelet_kept_subbands", "sigma", "cov_reg"):
            val = params[key]
            canonical = tuple(val) if isinstance(val, list) else val
            bucket = accum[key][canonical]
            bucket["count"] += 1
            bucket["img_auc"] += entry["avg_img_auc"]
            bucket["pixel_auc"] += entry["avg_pixel_auc"]
    summary: Dict[str, Dict[str, Any]] = {}
    for param_name, buckets in accum.items():
        items: List[Tuple[str, Dict[str, float]]] = []
        for raw_val, stats in buckets.items():
            count = stats["count"]
            avg_img = stats["img_auc"] / max(count, 1)
            avg_pixel = stats["pixel_auc"] / max(count, 1)
            label = ",".join(raw_val) if isinstance(raw_val, tuple) else str(raw_val)
            items.append((label, {"avg_img_auc": avg_img, "avg_pixel_auc": avg_pixel, "count": count}))
        summary[param_name] = {label: metrics for label, metrics in sorted(items, key=lambda item: item[1]["avg_img_auc"], reverse=True)}
    return summary

def save_progress(results: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: str) -> None:
    save_results(results, os.path.join(out_dir, "staged_probe_results.json"))
    summary_path = os.path.join(out_dir, "staged_probe_summary.json")
    with open(summary_path, "w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2)

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_path, f"{args.dataset_type}_{args.model}_probe_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    set_gpu_environment(args.gpu_id)
    device = torch.device("cuda" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")

    classes = resolve_class_list(args)
    print(f"Using {len(classes)} classes: {classes}")

    wavelet_types = args.wavelet_types or DEFAULT_WAVELET_TYPES
    wavelet_levels = args.wavelet_levels or DEFAULT_WAVELET_LEVELS
    sigma_values = args.sigma_values or DEFAULT_SIGMA_VALUES
    cov_reg_values = args.cov_reg_values or DEFAULT_COV_REG_VALUES
    subband_options = parse_subband_options(args.subband_options)

    print("Configured candidate sets:")
    print(f"  wavelet_type: {wavelet_types}")
    print(f"  wavelet_level: {wavelet_levels}")
    print(f"  sigma: {sigma_values}")
    print(f"  cov_reg: {cov_reg_values}")
    print(f"  subbands: {subband_options}")

    feature_extractor = FeatureExtractor(
        model_name=args.model,
        device=device,
        resource_monitor=None,
    )

    resource_monitor = None
    if args.enable_resource_monitoring:
        log_path = os.path.join(run_dir, "resource_log.csv")
        resource_monitor = ResourceMonitor(log_file=log_path)

    evaluator = PaDiMEvaluator(
        feature_extractor=feature_extractor,
        data_path=args.data_path,
        save_path=run_dir,
        test_batch_size=args.test_batch_size,
        resource_monitor=resource_monitor,
        device=device,
        dataset_type=args.dataset_type,
    )

    all_combos = list(itertools.product(wavelet_types, wavelet_levels, subband_options, sigma_values, cov_reg_values))
    sample_count = min(args.stage1_samples, len(all_combos))
    random.seed(args.stage1_seed)
    random.shuffle(all_combos)
    stage1_combos = all_combos[:sample_count]

    evaluated: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    collected: List[Dict[str, Any]] = []

    print(f"\n=== Stage 1: evaluating {sample_count} random combinations ===")
    for idx, combo in enumerate(stage1_combos, start=1):
        params = {
            "wavelet_type": combo[0],
            "wavelet_level": combo[1],
            "wavelet_kept_subbands": combo[2],
            "sigma": combo[3],
            "cov_reg": combo[4],
        }
        key = combo_key(params)
        print(f"Stage 1 [{idx}/{sample_count}] params={params}")
        result = evaluate_params(
            evaluator,
            feature_extractor,
            classes,
            params,
            args,
            run_dir,
            stage_label="stage1",
            resource_monitor=resource_monitor,
        )
        if not result:
            continue
        evaluated[key] = result
        collected.append(result)

    if not collected:
        print("No successful evaluations recorded. Exiting.")
        return

    best_by_img = sorted(collected, key=lambda entry: entry["avg_img_auc"], reverse=True)
    best_by_pixel = sorted(collected, key=lambda entry: entry["avg_pixel_auc"], reverse=True)

    anchors: List[Dict[str, Any]] = []
    if not args.skip_stage2:
        print("\n=== Stage 2: one-parameter sweeps around top Stage 1 configs ===")
        anchors.extend(best_by_img[: args.stage2_topk])
        anchors.extend(best_by_pixel[: args.stage2_topk])

        for anchor_idx, anchor in enumerate(anchors, start=1):
            base_params = anchor["params"]
            anchor_label = f"stage2_anchor_{anchor_idx}"
            print(f"\nAnchor {anchor_idx}: base params={base_params}")
            for param_name, candidates in (
                ("wavelet_type", wavelet_types),
                ("wavelet_level", wavelet_levels),
                ("wavelet_kept_subbands", subband_options),
                ("sigma", sigma_values),
                ("cov_reg", cov_reg_values),
            ):
                for candidate in candidates:
                    new_params = deepcopy(base_params)
                    new_params[param_name] = candidate
                    key = combo_key(new_params)
                    if key in evaluated:
                        continue
                    print(f"Stage 2 sweep on {param_name} -> {candidate}")
                    result = evaluate_params(
                        evaluator,
                        feature_extractor,
                        classes,
                        new_params,
                        args,
                        run_dir,
                        stage_label=anchor_label,
                        resource_monitor=resource_monitor,
                    )
                    if not result:
                        continue
                    evaluated[key] = result
                    collected.append(result)

    summary_payload = {
        "best_image_auc": max(collected, key=lambda entry: entry["avg_img_auc"]),
        "best_pixel_auc": max(collected, key=lambda entry: entry["avg_pixel_auc"]),
        "per_parameter_averages": aggregate_by_param(collected),
        "total_evaluations": len(collected),
    }

    save_progress(collected, summary_payload, run_dir)

    best_img_entry = summary_payload["best_image_auc"]
    best_pix_entry = summary_payload["best_pixel_auc"]

    print("\n=== Summary ===")
    print("Best Image AUC combo:")
    print(json.dumps(best_img_entry, indent=2))
    print("\nBest Pixel AUC combo:")
    print(json.dumps(best_pix_entry, indent=2))
    print(f"\nEvaluations stored under: {run_dir}")

if __name__ == "__main__":
    main()
