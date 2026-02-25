# phase2_subband_ablation_runner.py
"""
Runner script for WE-PaDiM Phase 2 Subband Ablation Study with CLI args.

Iterates through all non-empty subband combinations for a specified base model,
using previously determined 'best' hyperparameters (chosen via CLI) as the base.
"""

import os
import sys
import traceback
from datetime import datetime
import time
import gc
import torch
import argparse  # import argparse
from itertools import combinations, chain, product
from typing import List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# assuming these modules are in the same directory or accessible via pythonpath
try:
    from config import Config
    from main import run_experiment
    from config import MVTEC_CLASSES
    from utils import manage_gpu_memory
except ImportError as e:
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}", file=sys.stderr)
    sys.exit(1)

# optimized for image auc
BEST_PARAMS_IMG_AUC = {
    "efficientnet-b0": {"wavelet_type": "sym4", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b1": {"wavelet_type": "sym4", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b2": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 4.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b3": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 4.0, "cov_reg_epsilon": 0.01},
    "efficientnet-b4": {"wavelet_type": "sym4", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.01},
    "efficientnet-b5": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 4.0, "cov_reg_epsilon": 0.001},
    "efficientnet-b6": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 4.0, "cov_reg_epsilon": 0.01},
    "resnet18":        {"wavelet_type": "sym4", "wavelet_level": 2, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.01},
}

# optimized for pixel auc
BEST_PARAMS_PIX_AUC = {
    "efficientnet-b0": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b1": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b2": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b3": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b4": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b5": {"wavelet_type": "haar", "wavelet_level": 2, "gaussian_sigma": 4.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b6": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "resnet18":        {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.001},
}

PARAM_SET_MAP = {
    "image": BEST_PARAMS_IMG_AUC,
    "pixel": BEST_PARAMS_PIX_AUC,
}

ALL_SUBBANDS = ['LL', 'LH', 'HL', 'HH']
SUBBAND_COMBINATIONS = list(chain.from_iterable(
    combinations(ALL_SUBBANDS, r) for r in range(1, len(ALL_SUBBANDS) + 1)
))
SUBBAND_COMBINATIONS = [sorted(list(combo)) for combo in SUBBAND_COMBINATIONS]

# dictionary to map cli choice to the actual parameter dictionary
def run_subband_ablation(
    model_name: str,
    base_param_choice: str,
    save_dir_base: str,
    data_path: str,
    dataset_type: str = 'auto',
    wavelet_types: Optional[List[str]] = None,
    wavelet_levels: Optional[List[int]] = None,
    gaussian_sigmas: Optional[List[float]] = None,
    cov_reg_epsilons: Optional[List[float]] = None
):
    """Configure and run the subband ablation study across a wavelet parameter grid."""

    overall_success = True
    run_counter = 0

    if base_param_choice not in PARAM_SET_MAP:
        print(f"ERROR: Invalid base parameter choice '{base_param_choice}'. Choose 'image' or 'pixel'.")
        sys.exit(1)

    base_param_dict = PARAM_SET_MAP[base_param_choice]
    base_param_key_str = f"base_{base_param_choice}_auc"

    if model_name not in base_param_dict:
        print(f"ERROR: Model '{model_name}' not found in the parameter set for '{base_param_choice}' AUC.")
        sys.exit(1)

    base_params = base_param_dict[model_name]

    wavelet_type_candidates = wavelet_types or [base_params["wavelet_type"]]
    wavelet_level_candidates = wavelet_levels or [base_params["wavelet_level"]]
    sigma_candidates = gaussian_sigmas or [base_params["gaussian_sigma"]]
    cov_reg_candidates = cov_reg_epsilons or [base_params["cov_reg_epsilon"]]

    wavelet_param_grid = list(product(
        wavelet_type_candidates,
        wavelet_level_candidates,
        sigma_candidates,
        cov_reg_candidates
    ))

    if not wavelet_param_grid:
        print("ERROR: Wavelet parameter grid is empty. Provide valid wavelet settings.")
        return False

    def _format_float_label(value: float) -> str:
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return str(value).replace('.', 'p')
        return str(value)

    print(f"\n*** Starting Subband Ablation for Model: {model_name} ***")
    print(f"*** Using Base Parameters Optimized For: {base_param_choice.upper()} AUC ***")
    print(f"Base Params: {base_params}")
    if any([wavelet_types, wavelet_levels, gaussian_sigmas, cov_reg_epsilons]):
        print("Overriding wavelet parameter grid with:")
        print(f"  Wavelet types: {wavelet_type_candidates}")
        print(f"  Wavelet levels: {wavelet_level_candidates}")
        print(f"  Gaussian sigmas: {sigma_candidates}")
        print(f"  Covariance regularization epsilons: {cov_reg_candidates}")

    total_runs = len(SUBBAND_COMBINATIONS) * len(wavelet_param_grid)
    print(f"Total runs to execute: {total_runs}")

    for wavelet_type, wavelet_level, sigma, cov_epsilon in wavelet_param_grid:
        wavelet_suffix = "_".join([
            f"wt-{wavelet_type}",
            f"lvl-{wavelet_level}",
            f"sigma-{_format_float_label(sigma)}",
            f"cov-{_format_float_label(cov_epsilon)}"
        ])

        for subband_combo in SUBBAND_COMBINATIONS:
            run_counter += 1
            subband_str = "_".join(subband_combo)
            run_label = f"{model_name}_{base_param_key_str}_{wavelet_suffix}_subbands_{subband_str}"

            print(f"\n{'='*50}")
            print(f"--- Starting Run {run_counter}/{total_runs}: {run_label} ---")
            print(f"--- Wavelet Params: type={wavelet_type}, level={wavelet_level}, sigma={sigma}, cov_epsilon={cov_epsilon} ---")
            print(f"--- Subbands: {subband_combo} ---")
            print(f"{'='*50}")

            config = Config(None)
            config.models = [model_name]
            config.classes = MVTEC_CLASSES
            config.experiment_type = 'single_model'

            config.wavelet_type = wavelet_type
            config.wavelet_level = wavelet_level
            config.gaussian_sigma = sigma
            config.cov_reg_epsilon = cov_epsilon
            config.wavelet_kept_subbands = subband_combo

            config.data_path = data_path
            config.dataset_type = dataset_type
            if dataset_type == 'visa':
                from config import VISA_CLASSES
                config.classes = VISA_CLASSES
            else:
                config.classes = MVTEC_CLASSES

            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.run_name = f"SubbandAblation_{run_label}_{run_timestamp}"
            config.save_path = os.path.join(save_dir_base, config.run_name)

            config.save_visualizations = False
            config.enable_resource_monitoring = False
            config.memory_efficient = True

            print(f"Running with configuration for {run_label}:")
            print(config)
            os.makedirs(config.save_path, exist_ok=True)
            config.save(os.path.join(config.save_path, "config.json"))

            start_time = time.time()
            try:
                run_experiment(config)
                end_time = time.time()
                print(f"\n--- Run '{run_label}' Completed (Took {(end_time - start_time)/60:.2f} minutes) ---")
                print(f"Results saved within: {config.save_path}")
            except Exception as e:
                end_time = time.time()
                print(f"\n--- ERROR During Run '{run_label}' (After {(end_time - start_time)/60:.2f} minutes) ---", file=sys.stderr)
                print(f"Error: {e}", file=sys.stderr)
                traceback.print_exc()
                print(f"Results (if any) might be incomplete in: {config.save_path}")
                overall_success = False

            print(f"--- Cleaning up after run '{run_label}' ---")
            manage_gpu_memory(f"End of run {run_label}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nFinished all subband combinations for model '{model_name}' based on '{base_param_choice}' AUC parameters.")
    return overall_success

# --- command line argument parsing ---
def _parse_list_argument(raw_value: str, cast_fn):
    if raw_value is None or raw_value.strip() == "":
        return None
    items = [item.strip() for item in raw_value.split(',') if item.strip()]
    if not items:
        return None
    try:
        return [cast_fn(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse list values '{raw_value}': {exc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run WE-PaDiM Subband Ablation Study for a specific model and base parameter set.')

    # get available model choices from the parameter dictionaries
    available_models = list(set(BEST_PARAMS_IMG_AUC.keys()) | set(BEST_PARAMS_PIX_AUC.keys()))

    parser.add_argument('--model', type=str, required=True, choices=available_models,
                        help='The CNN backbone model to run the ablation on.')
    parser.add_argument('--base_params', type=str, required=True, choices=['image', 'pixel'],
                        help="Which set of base parameters to use: 'image' (optimized for Image AUC) or 'pixel' (optimized for Pixel AUC).")
    parser.add_argument('--save_dir_base', type=str, default='./results/phase2_subband_ablation_cli',
                        help='Base directory where run-specific result folders will be created.')
    parser.add_argument('--data_path', type=str, default='./data/MVTec',
                        help='Path to the MVTec AD dataset.')
    parser.add_argument('--dataset_type', type=str, default='auto',
                       choices=['mvtec', 'visa', 'auto'],
                       help='Dataset type: mvtec, visa, or auto (default: auto)')
    parser.add_argument('--wavelet_types', type=str, default='',
                        help='Comma-separated list of wavelet types to evaluate (default: use base parameter).')
    parser.add_argument('--wavelet_levels', type=str, default='',
                        help='Comma-separated list of wavelet decomposition levels to evaluate (integers).')
    parser.add_argument('--gaussian_sigmas', type=str, default='',
                        help='Comma-separated list of Gaussian smoothing sigma values to evaluate.')
    parser.add_argument('--cov_reg_epsilons', type=str, default='',
                        help='Comma-separated list of covariance regularization epsilons to evaluate.')
    # add other arguments if needed, e.g., gpu id
    # parser.add_argument('--gpu_id', type=int, default=none, help='gpu id to use.')

    args = parser.parse_args()

    print("--- Starting Subband Ablation Study Runner (CLI Version) ---")
    print(f"Target Model: {args.model}")
    print(f"Base Parameters: Best {args.base_params.upper()} AUC")
    wavelet_types_arg = _parse_list_argument(args.wavelet_types, str)
    wavelet_levels_arg = _parse_list_argument(args.wavelet_levels, int)
    gaussian_sigmas_arg = _parse_list_argument(args.gaussian_sigmas, float)
    cov_reg_eps_arg = _parse_list_argument(args.cov_reg_epsilons, float)

    if any([wavelet_types_arg, wavelet_levels_arg, gaussian_sigmas_arg, cov_reg_eps_arg]):
        print("Wavelet parameter overrides supplied; grid will expand accordingly.")
    else:
        print(f"Will test {len(SUBBAND_COMBINATIONS)} subband combinations using base wavelet parameters.")
    print(f"Results base directory: {args.save_dir_base}")

    overall_start_time = time.time()
    # call the main function with parsed arguments
    success = run_subband_ablation(
        model_name=args.model,
        base_param_choice=args.base_params,
        save_dir_base=args.save_dir_base,
        data_path=args.data_path,
        dataset_type=args.dataset_type,
        wavelet_types=wavelet_types_arg,
        wavelet_levels=wavelet_levels_arg,
        gaussian_sigmas=gaussian_sigmas_arg,
        cov_reg_epsilons=cov_reg_eps_arg
    )
    overall_end_time = time.time()
    print(f"\n--- Subband Ablation Study Finished (Total time: {(overall_end_time - overall_start_time)/60:.2f} minutes) ---")

    if success:
        print("All configured runs completed successfully.")
    else:
        print("One or more configured runs encountered errors during execution.", file=sys.stderr)
        sys.exit(1)
