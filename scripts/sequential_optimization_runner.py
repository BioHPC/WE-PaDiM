# sequential_optimization_runner.py
import os
import sys
import argparse
import itertools
import numpy as np
from datetime import datetime

# ensure src path is available
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from experiment import run_model_experiment
# corrected import: import classes from config, not dataset
from dataset import get_all_class_names, resolve_dataset_type
from utils import manage_gpu_memory

# --- search spaces ---
STAGE_1_WAVELETS = ['haar', 'db4', 'sym4', 'coif1']
STAGE_1_LEVELS = [1, 2]

STAGE_2_SIGMAS = [2.0, 4.0, 8.0]
STAGE_2_COV_REGS = [0.1, 0.01, 0.001]

# generate all subband combinations for stage 3
ALL_SUBBANDS = ['LL', 'LH', 'HL', 'HH']
STAGE_3_SUBBANDS = []
for r in range(1, len(ALL_SUBBANDS) + 1):
    STAGE_3_SUBBANDS.extend([list(c) for c in itertools.combinations(ALL_SUBBANDS, r)])

def get_best_params_from_results(results_list, metric='img_auc'):
    """Parses raw result list to find params yielding max metric."""
    best_score = -1.0
    best_config = None

    key_map = {'img_auc': 'avg_img_auc', 'pixel_auc': 'avg_pixel_auc'}
    target_key = key_map.get(metric, 'avg_img_auc')

    for res in results_list:
        score = res.get(target_key, 0.0)
        if score > best_score:
            best_score = score
            best_config = res['params']

    return best_config, best_score

def run_sequential_search(args):
    print(f"=== Starting Sequential Optimization for {args.model} ===")
    print(f"Target Metric: {args.metric.upper()}")
    resolved_dataset_type = resolve_dataset_type(args.data_path, args.dataset_type)
    if resolved_dataset_type != args.dataset_type:
        print(f"Resolved dataset type '{resolved_dataset_type}' from data path (override requested '{args.dataset_type}')")
    dataset_type = resolved_dataset_type
    print(f"Dataset Type: {dataset_type}")

    save_path_root = os.path.join(args.save_path, f"{args.model}_seq_optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_path_root, exist_ok=True)

    # --- determine target classes ---
    if args.classes:
        target_classes = args.classes
    else:
        try:
            target_classes = get_all_class_names(args.data_path, dataset_type=dataset_type)
        except Exception as exc:
            raise RuntimeError(f"Unable to resolve class list for dataset '{args.data_path}': {exc}")

    if not target_classes:
        raise RuntimeError("Resolved class list is empty; aborting optimization run.")

    print(f"Classes to process: {len(target_classes)}")

    # stage 1: wavelet topology
    print("\n--- STAGE 1: Finding Best Wavelet Type & Level ---")
    stage1_config = {
        'wavelet_type': STAGE_1_WAVELETS,
        'wavelet_level': STAGE_1_LEVELS,
        'wavelet_kept_subbands': [['LL', 'LH', 'HL']],  # default baseline
        'sigma': [4.0],  # default baseline
        'cov_reg': [0.01]  # default baseline
    }

    res_s1 = run_model_experiment(
        model_name=args.model,
        data_path=args.data_path,
        save_path=os.path.join(save_path_root, "stage1_topology"),
        class_names=target_classes,
        experiment_type='grid_search',
        config=stage1_config,
        train_batch_size=args.batch_size,
        save_visualizations=False,
        dataset_type=dataset_type
    )
    stage1_results = res_s1.get('results', []) if isinstance(res_s1, dict) else []
    if not stage1_results:
        raise RuntimeError("Stage 1 grid search produced no valid results.")

    best_p_s1, score_s1 = get_best_params_from_results(stage1_results, args.metric)
    if best_p_s1 is None:
        raise RuntimeError("Unable to determine Stage 1 winner; all combinations failed.")
    best_type = best_p_s1['wavelet_type']
    best_level = best_p_s1['wavelet_level']
    print(f"-> Stage 1 Winner: Type={best_type}, Level={best_level} ({args.metric}={score_s1:.4f})")
    manage_gpu_memory()

    # stage 2: hyperparameters
    print("\n--- STAGE 2: Tuning Sigma & Covariance Regularization ---")
    stage2_config = {
        'wavelet_type': [best_type],  # fixed from s1
        'wavelet_level': [best_level],  # fixed from s1
        'wavelet_kept_subbands': [['LL', 'LH', 'HL']],
        'sigma': STAGE_2_SIGMAS,
        'cov_reg': STAGE_2_COV_REGS
    }

    res_s2 = run_model_experiment(
        model_name=args.model,
        data_path=args.data_path,
        save_path=os.path.join(save_path_root, "stage2_hyperparams"),
        class_names=target_classes,
        experiment_type='grid_search',
        config=stage2_config,
        train_batch_size=args.batch_size,
        save_visualizations=False,
        dataset_type=dataset_type
    )
    stage2_results = res_s2.get('results', []) if isinstance(res_s2, dict) else []
    if not stage2_results:
        raise RuntimeError("Stage 2 grid search produced no valid results.")

    best_p_s2, score_s2 = get_best_params_from_results(stage2_results, args.metric)
    if best_p_s2 is None:
        raise RuntimeError("Unable to determine Stage 2 winner; all combinations failed.")
    best_sigma = best_p_s2['sigma']
    best_cov = best_p_s2['cov_reg']
    print(f"-> Stage 2 Winner: Sigma={best_sigma}, CovReg={best_cov} ({args.metric}={score_s2:.4f})")
    manage_gpu_memory()

    # stage 3: subband ablation
    print("\n--- STAGE 3: Subband Selection (Ablation) ---")
    stage3_config = {
        'wavelet_type': [best_type],  # fixed from s1
        'wavelet_level': [best_level],  # fixed from s1
        'sigma': STAGE_2_SIGMAS,  # re-tune smoothing per subband set
        'cov_reg': STAGE_2_COV_REGS,  # re-tune covariance reg per subband set
        'wavelet_kept_subbands': STAGE_3_SUBBANDS
    }

    res_s3 = run_model_experiment(
        model_name=args.model,
        data_path=args.data_path,
        save_path=os.path.join(save_path_root, "stage3_subbands"),
        class_names=target_classes,
        experiment_type='grid_search',
        config=stage3_config,
        train_batch_size=args.batch_size,
        save_visualizations=False,
        dataset_type=dataset_type
    )
    stage3_results = res_s3.get('results', []) if isinstance(res_s3, dict) else []
    if not stage3_results:
        raise RuntimeError("Stage 3 grid search produced no valid results.")

    best_p_final, score_final = get_best_params_from_results(stage3_results, args.metric)
    if best_p_final is None:
        raise RuntimeError("Unable to determine Stage 3 winner; all combinations failed.")
    best_subbands = best_p_final['wavelet_kept_subbands']
    best_sigma = best_p_final['sigma']
    best_cov = best_p_final['cov_reg']

    # final report
    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE for {args.model}")
    print(f"Dataset: {dataset_type}")
    print(f"Optimized for: {args.metric.upper()}")
    print("-" * 50)
    print(f"Best Configuration:")
    print(f"  Wavelet Type:  {best_type}")
    print(f"  Wavelet Level: {best_level}")
    print(f"  Kept Subbands: {best_subbands}")
    print(f"  Sigma:         {best_sigma}")
    print(f"  Cov Reg:       {best_cov}")
    print("-" * 50)
    print(f"Final Score ({args.metric}): {score_final:.4f}")
    print(f"Results saved to: {save_path_root}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Backbone model name (e.g., resnet18)")
    parser.add_argument("--data_path", type=str, default="./data/MVTec")
    parser.add_argument("--save_path", type=str, default="./results/sequential_optim")
    parser.add_argument("--metric", type=str, default="img_auc", choices=["img_auc", "pixel_auc"], help="Metric to optimize for")
    parser.add_argument("--classes", nargs="+", default=None, help="Specific classes to run on (default: all)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_type", type=str, default='auto', choices=['mvtec', 'visa', 'auto'])

    args = parser.parse_args()
    run_sequential_search(args)

