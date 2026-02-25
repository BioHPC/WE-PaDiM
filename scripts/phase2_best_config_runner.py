# phase2_best_config_runner.py
"""
Runner script for WE-PaDiM Phase 2 Evaluation for multiple specific configurations.
Allows running models with parameters optimized for different metrics (e.g., Image AUC vs Pixel AUC).
"""

import os
import sys
import traceback
from datetime import datetime
import time
import gc
import torch

import argparse

def parse_script_args():
    """Parse command line arguments for this script."""
    parser = argparse.ArgumentParser(description='Phase 2 Best Models Runner')
    parser.add_argument('--dataset_type', type=str, default='mvtec',
                       choices=['mvtec', 'visa', 'auto'],
                       help='Dataset type (default: mvtec)')
    parser.add_argument('--data_path', type=str, default='./data/MVTec',
                       help='Path to dataset')
    return parser.parse_args()

# assuming these modules are in the same directory or accessible via pythonpath
try:
    from config import Config
    from main import run_experiment
    from dataset import MVTEC_CLASSES
    from utils import manage_gpu_memory
except ImportError as e:
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}", file=sys.stderr)
    sys.exit(1)

# --- best hyperparameters optimized for image auc ---
BEST_PARAMS_IMG_AUC = {
    "efficientnet-b0": {"wavelet_type": "sym4", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b1": {"wavelet_type": "sym4", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b2": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 4.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b6": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 4.0, "cov_reg_epsilon": 0.01},
    "resnet18":        {"wavelet_type": "sym4", "wavelet_level": 2, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.01},
}

# --- best hyperparameters optimized for pixel auc ---
BEST_PARAMS_PIX_AUC = {
    "efficientnet-b0": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b1": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b2": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "efficientnet-b6": {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.10},
    "resnet18":        {"wavelet_type": "haar", "wavelet_level": 1, "gaussian_sigma": 2.0, "cov_reg_epsilon": 0.001},
}

# --- choose which parameter set dictionary to use ---
PARAM_SETS = {
    "best_img_auc": BEST_PARAMS_IMG_AUC,
    "best_pix_auc": BEST_PARAMS_PIX_AUC,
}

# each dict specifies:
# 'label': a unique identifier for this run (used in folder names).
# 'model_name': the backbone model.
# 'param_set_key': which dictionary ('best_img_auc' or 'best_pix_auc') to use for parameters.
RUN_CONFIGURATIONS = [
    # example: run all models optimized for image auc
    {'label': 'efficientnet-b0_best_img_auc', 'model_name': 'efficientnet-b0', 'param_set_key': 'best_img_auc'},
    {'label': 'efficientnet-b1_best_img_auc', 'model_name': 'efficientnet-b1', 'param_set_key': 'best_img_auc'},
    {'label': 'efficientnet-b2_best_img_auc', 'model_name': 'efficientnet-b2', 'param_set_key': 'best_img_auc'},
    {'label': 'efficientnet-b6_best_img_auc', 'model_name': 'efficientnet-b6', 'param_set_key': 'best_img_auc'},
    {'label': 'resnet18_best_img_auc',        'model_name': 'resnet18',        'param_set_key': 'best_img_auc'},

    # example: run all models optimized for pixel auc
    {'label': 'efficientnet-b0_best_pix_auc', 'model_name': 'efficientnet-b0', 'param_set_key': 'best_pix_auc'},
    {'label': 'efficientnet-b1_best_pix_auc', 'model_name': 'efficientnet-b1', 'param_set_key': 'best_pix_auc'},
    {'label': 'efficientnet-b2_best_pix_auc', 'model_name': 'efficientnet-b2', 'param_set_key': 'best_pix_auc'},
    {'label': 'efficientnet-b6_best_pix_auc', 'model_name': 'efficientnet-b6', 'param_set_key': 'best_pix_auc'},
    {'label': 'resnet18_best_pix_auc',        'model_name': 'resnet18',        'param_set_key': 'best_pix_auc'},

    # add more specific runs here if needed, e.g.,
    # {'label': 'resnet18_alternative_params', 'model_name': 'resnet18', 'param_set_key': 'some_other_param_set'},
]

WAVELET_KEPT_SUBBANDS = ['LL', 'LH', 'HL']  # fixed from phase 1

def run_phase2_multi_config():
    script_args = parse_script_args()
    """Configures and runs the Phase 2 experiment for each defined configuration."""
    overall_success = True

    for run_config in RUN_CONFIGURATIONS:
        run_label = run_config['label']
        model_name = run_config['model_name']
        param_set_key = run_config['param_set_key']

        print(f"\n{'='*50}")
        print(f"--- Starting Phase 2 Run: {run_label} ---")
        print(f"--- Model: {model_name}, Params: {param_set_key} ---")
        print(f"{'='*50}")

        if param_set_key not in PARAM_SETS:
            print(f"ERROR: Parameter set key '{param_set_key}' not found in PARAM_SETS for run '{run_label}'. Skipping.")
            overall_success = False
            continue
        param_dict = PARAM_SETS[param_set_key]

        if model_name not in param_dict:
            print(f"ERROR: Model '{model_name}' not found in parameter set '{param_set_key}' for run '{run_label}'. Skipping.")
            overall_success = False
            continue
        params = param_dict[model_name]

        args = None
        config = Config(args)

        config.models = [model_name]  # only run the current model
        config.classes = MVTEC_CLASSES
        config.experiment_type = 'single_model'

        config.wavelet_type = params["wavelet_type"]
        config.wavelet_level = params["wavelet_level"]
        config.gaussian_sigma = params["gaussian_sigma"]
        config.cov_reg_epsilon = params["cov_reg_epsilon"]
        config.wavelet_kept_subbands = WAVELET_KEPT_SUBBANDS

        config.data_path = script_args.data_path
        config.dataset_type = script_args.dataset_type
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.run_name = f"Phase2_{run_label}_{run_timestamp}"
        config.save_path = os.path.join("./results/phase2", config.run_name)

        config.save_visualizations = True
        config.enable_resource_monitoring = True
        config.memory_efficient = True

        print(f"Running with configuration for {run_label}:")
        config.print_config()
        os.makedirs(config.save_path, exist_ok=True)
        config.save(os.path.join(config.save_path, "config.json"))

        start_time = time.time()
        stage_success = False
        try:
            run_experiment(config)
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

        # --- cleanup before next run ---
        print(f"--- Cleaning up after run '{run_label}' ---")
        manage_gpu_memory(f"End of run {run_label}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{'='*50}\n")

        if script_args.dataset_type == 'visa':
            from config import VISA_CLASSES
            config.classes = VISA_CLASSES
        else:
            config.classes = MVTEC_CLASSES

    return overall_success

if __name__ == "__main__":
    print("--- Starting Multi-Configuration Phase 2 Runner ---")
    overall_start_time = time.time()
    success = run_phase2_multi_config()
    overall_end_time = time.time()
    print(f"\n--- Multi-Configuration Phase 2 Runner Finished (Total time: {(overall_end_time - overall_start_time)/60:.2f} minutes) ---")

    if success:
        print("All configured runs completed successfully.")
    else:
        print("One or more configured runs encountered errors during execution.", file=sys.stderr)
        sys.exit(1)
