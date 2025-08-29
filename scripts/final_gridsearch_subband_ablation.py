# phase2_runner_subband_ablation_cli.py
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
import argparse # Import argparse
from itertools import combinations, chain

# Assuming these modules are in the same directory or accessible via PYTHONPATH
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

# Optimized for IMAGE AUC
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

# Optimized for PIXEL AUC
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

# Dictionary to map CLI choice to the actual parameter dictionary
PARAM_SET_MAP = {
    "image": BEST_PARAMS_IMG_AUC,
    "pixel": BEST_PARAMS_PIX_AUC,
}

# --- Define all possible non-empty subband combinations ---
ALL_SUBBANDS = ['LL', 'LH', 'HL', 'HH']
SUBBAND_COMBINATIONS = list(chain.from_iterable(
    combinations(ALL_SUBBANDS, r) for r in range(1, len(ALL_SUBBANDS)+1)
))
# Convert tuples to lists and sort for consistency
SUBBAND_COMBINATIONS = [sorted(list(combo)) for combo in SUBBAND_COMBINATIONS]

# --- Main Ablation Function ---
def run_subband_ablation(model_name: str, base_param_choice: str, save_dir_base: str, data_path: str):
    """
    Configures and runs the subband ablation study for the specified model and base parameters.

    Args:
        model_name (str): The CNN backbone model to use.
        base_param_choice (str): 'image' or 'pixel', indicating which base parameters to use.
        save_dir_base (str): The top-level directory to save results for this run type.
        data_path (str): Path to the MVTec dataset.
    """
    overall_success = True
    run_counter = 0

    # --- Select Base Parameters based on CLI choice ---
    if base_param_choice not in PARAM_SET_MAP:
        print(f"ERROR: Invalid base parameter choice '{base_param_choice}'. Choose 'image' or 'pixel'.")
        sys.exit(1)
    base_param_dict = PARAM_SET_MAP[base_param_choice]
    base_param_key_str = f"base_{base_param_choice}_auc" # For directory naming

    # --- Check if model exists in the chosen parameter set ---
    if model_name not in base_param_dict:
        print(f"ERROR: Model '{model_name}' not found in the parameter set for '{base_param_choice}' AUC.")
        sys.exit(1)

    base_params = base_param_dict[model_name]
    print(f"\n*** Starting Subband Ablation for Model: {model_name} ***")
    print(f"*** Using Base Parameters Optimized For: {base_param_choice.upper()} AUC ***")
    print(f"Base Params: {base_params}")

    # Iterate through each subband combination
    for subband_combo in SUBBAND_COMBINATIONS:
        run_counter += 1
        subband_str = "_".join(subband_combo)
        run_label = f"{model_name}_{base_param_key_str}_subbands_{subband_str}"

        print(f"\n{'='*50}")
        print(f"--- Starting Run {run_counter}/{len(SUBBAND_COMBINATIONS)}: {run_label} ---")
        print(f"--- Subbands: {subband_combo} ---")
        print(f"{'='*50}")

        # --- Configuration per Run ---
        cli_args_for_config = None # Config class expects args object, pass None to use defaults initially
        config = Config(cli_args_for_config)

        # Set parameters for this specific run
        config.models = [model_name]
        config.classes = MVTEC_CLASSES
        config.experiment_type = 'single_model' # Run as single config

        # Apply base hyperparameters (wavelet type, level, sigma, cov_reg)
        config.wavelet_type = base_params["wavelet_type"]
        config.wavelet_level = base_params["wavelet_level"]
        config.gaussian_sigma = base_params["gaussian_sigma"]
        config.cov_reg_epsilon = base_params["cov_reg_epsilon"]

        # Apply the specific subband combination for THIS run
        config.wavelet_kept_subbands = subband_combo

        # Other parameters
        config.data_path = data_path # Use data_path from args
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.run_name = f"SubbandAblation_{run_label}_{run_timestamp}"
        # Organize results under the base save directory
        config.save_path = os.path.join(save_dir_base, config.run_name)

        # Keep other defaults or override as necessary
        config.save_visualizations = False
        config.calculate_pro_score = True
        config.enable_resource_monitoring = False # Often disabled for many small runs
        config.memory_efficient = True
        # Add --gpu_id handling if needed (can be passed through args or set here)
        # config.gpu_id = args.gpu_id # If gpu_id is added to argparse

        # --- Execution for Current Run ---
        print(f"Running with configuration for {run_label}:")
        print(config)
        os.makedirs(config.save_path, exist_ok=True)
        config.save(os.path.join(config.save_path, "config.json"))

        start_time = time.time()
        stage_success = False
        try:
            run_experiment(config) # Calls the main experiment runner
            stage_success = True
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

        # --- Cleanup before next run ---
        print(f"--- Cleaning up after run '{run_label}' ---")
        manage_gpu_memory(f"End of run {run_label}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFinished all subband combinations for model '{model_name}' based on '{base_param_choice}' AUC parameters.")
    return overall_success

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run WE-PaDiM Subband Ablation Study for a specific model and base parameter set.')

    # Get available model choices from the parameter dictionaries
    available_models = list(set(BEST_PARAMS_IMG_AUC.keys()) | set(BEST_PARAMS_PIX_AUC.keys()))

    parser.add_argument('--model', type=str, required=True, choices=available_models,
                        help='The CNN backbone model to run the ablation on.')
    parser.add_argument('--base_params', type=str, required=True, choices=['image', 'pixel'],
                        help="Which set of base parameters to use: 'image' (optimized for Image AUC) or 'pixel' (optimized for Pixel AUC).")
    parser.add_argument('--save_dir_base', type=str, default='./results/phase2_subband_ablation_cli',
                        help='Base directory where run-specific result folders will be created.')
    parser.add_argument('--data_path', type=str, default='./data/MVTec',
                        help='Path to the MVTec AD dataset.')
    # Add other arguments if needed, e.g., GPU ID
    # parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use.')

    args = parser.parse_args()

    print("--- Starting Subband Ablation Study Runner (CLI Version) ---")
    print(f"Target Model: {args.model}")
    print(f"Base Parameters: Best {args.base_params.upper()} AUC")
    print(f"Will test {len(SUBBAND_COMBINATIONS)} subband combinations.")
    print(f"Results base directory: {args.save_dir_base}")

    overall_start_time = time.time()
    # Call the main function with parsed arguments
    success = run_subband_ablation(
        model_name=args.model,
        base_param_choice=args.base_params,
        save_dir_base=args.save_dir_base,
        data_path=args.data_path
    )
    overall_end_time = time.time()
    print(f"\n--- Subband Ablation Study Finished (Total time: {(overall_end_time - overall_start_time)/60:.2f} minutes) ---")

    if success:
        print("All configured runs completed successfully.")
    else:
        print("One or more configured runs encountered errors during execution.", file=sys.stderr)
        sys.exit(1)
