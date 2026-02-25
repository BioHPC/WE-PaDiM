# phase1_reduced_grid_search.py
"""
Runner script for WE-PaDiM Parameter Tuning (Reduced Phase 1).
Focuses on finding optimal wavelet_type, level, sigma, cov_reg
for a specific backbone, using a reduced grid and fixed subbands
across all classes.
"""

import os
import argparse
from datetime import datetime
import sys
import traceback
import copy
import itertools
import time
import json
from typing import Optional, List, Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from config import Config, AVAILABLE_SUBBANDS, MVTEC_CLASSES, AVAILABLE_MODELS
    from main import run_experiment
    from visualization import visualize_grid_search_results
except ImportError as e:
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}", file=sys.stderr)
    sys.exit(1)

def calculate_grid_combinations(param_grid: dict) -> int:
    if not param_grid: return 1
    lists = [v if isinstance(v, list) else [v] for v in param_grid.values()]
    processed_lists = []
    for item in lists:
        if isinstance(item, list) and item and isinstance(item[0], list): processed_lists.append(item)
        elif isinstance(item, list): processed_lists.append(item)
        else: processed_lists.append([item])
    return len(list(itertools.product(*processed_lists)))

def run_parameter_search_stage(
    base_args: argparse.Namespace,
    param_grid: dict,
    stage_subdir_prefix: str = "Phase1_ReducedParamSearch",
    classes: Optional[List[str]] = None
):
    """Runs the reduced parameter search stage."""
    stage_name = f"Phase 1 (Reduced): Parameter Search for {base_args.model}"
    print(f"\n{'='*80}")
    print(f"--- Starting {stage_name} ---")
    print(f"{'='*80}")

    stage_config_args = copy.deepcopy(base_args)
    model_name = base_args.model
    stage_config_args.models = [model_name]
    stage_config_args.experiment_type = 'grid_search'
    stage_config_args.save_visualizations = base_args.save_anomaly_maps

    print("  Grid Parameters for this stage:")
    for key, value in param_grid.items():
        setattr(stage_config_args, key, value)
        print(f"    {key}: {value}")

    if classes:
        stage_config_args.classes = classes
        print(f"  Using specific classes for this stage: {classes}")
    else:
        stage_config_args.classes = None
        print(f"  Using all MVTec classes for this stage (Recommended).")

    try:
        config = Config(stage_config_args)
    except Exception as e:
        print(f"Error creating Config object for {stage_name}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

    actual_classes = config.get_classes()
    num_classes = len(actual_classes)
    num_combinations = calculate_grid_combinations(param_grid)

    print(f"\n  Run Details:")
    print(f"    Model: {config.models[0]}")
    print(f"    Save Anomaly Maps during run: {config.save_visualizations}")
    print(f"    Parameter Combinations (Reduced): {num_combinations}")  # updated count
    print(f"    Classes to process: {num_classes} ({'subset' if classes else 'all'})")
    print(f"    Total evaluations planned: {num_combinations * num_classes}")  # updated count

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = base_args.save_path
    stage_save_path = os.path.join(base_save_path, f"{stage_subdir_prefix}_{model_name}_{timestamp}")
    model_results_path = os.path.join(stage_save_path, model_name)

    try:
        os.makedirs(model_results_path, exist_ok=True)
        config.save_path = stage_save_path
        print(f"    Stage base results dir: {stage_save_path}")
        print(f"    Detailed model results/plots expected in: {model_results_path}")
    except OSError as e:
        print(f"Error creating save directories: {e}", file=sys.stderr)
        return None

    print("\n  Launching run_experiment...")
    start_time = time.time()
    stage_success = False
    try:
        run_experiment(config)
        stage_success = True
        end_time = time.time()
        print(f"\n--- {stage_name} Experiment Execution Completed (Took {(end_time - start_time)/60:.2f} minutes) ---")
        print(f"Results saved within: {stage_save_path}")
    except Exception as e:
        end_time = time.time()
        print(f"\n--- ERROR During {stage_name} Execution (After {(end_time - start_time)/60:.2f} minutes) ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        print(f"Results (if any) might be incomplete in: {stage_save_path}")

    # post-processing
    result_prefix = f"{model_name}_wavelet_grid_search"
    final_json_path = os.path.join(model_results_path, f"{result_prefix}_results_final.json")
    print(f"\n--- Post-Processing for {stage_name} ---")
    print(f"Checking for grid search results file: {final_json_path}")
    if os.path.exists(final_json_path):
        print("Attempting to load results from JSON...")
        try:
            with open(final_json_path, 'r') as f: loaded_results = json.load(f)
            if loaded_results:
                print(f"Loaded {len(loaded_results)} results. Generating visualizations...")
                visualize_grid_search_results(results=loaded_results, save_path=model_results_path, prefix=result_prefix)
                print(f"Grid search visualizations saved in: {model_results_path}")
            else: print("Results JSON file was empty.")
        except Exception as e: print(f"An error occurred during loading/visualization: {e}"); traceback.print_exc()
    else:
        print("Final results JSON file not found. Skipping visualization.")
        if stage_success: print("WARNING: Experiment execution seemed successful, but final results JSON is missing.")
    return stage_save_path

def main():
    parser = argparse.ArgumentParser(description='WE-PaDiM Reduced Phase 1 Parameter Search Runner')
    parser.add_argument('--data_path', type=str, required=True, help='Path to MVTec AD dataset.')
    parser.add_argument('--save_path', type=str, default='./results/WEPaDiM_Phase1_Reduced', help='Base path for Reduced Phase 1 results.')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID.')
    parser.add_argument('--model', type=str, required=True, choices=list(AVAILABLE_MODELS.keys()), help='Backbone model for this search phase (e.g., efficientnet-b0, resnet18).')
    parser.add_argument('--classes', type=str, nargs='+', default=None, help='Subset of classes for this run (default: all).')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Test batch size.')
    parser.add_argument('--dataset_type', type=str, default='auto',
                       choices=['mvtec', 'visa', 'auto'],
                       help='Dataset type: mvtec, visa, or auto (default: auto)')
    try:  # flag for anomaly map saving
        parser.add_argument('--save_anomaly_maps', action=argparse.BooleanOptionalAction, default=False, help='Save anomaly map visualizations during testing.')
    except AttributeError:  # fallback
        parser.add_argument('--save_anomaly_maps', action='store_true', default=False)
        parser.add_argument('--no_save_anomaly_maps', dest='save_anomaly_maps', action='store_false')

    base_args = parser.parse_args()
    os.makedirs(base_args.save_path, exist_ok=True)
    print(f"--- Starting Reduced Phase 1 Search for Model: {base_args.model} ---")

    # define the reduced parameter grid for phase 1
    phase1_reduced_param_grid = {
        'wavelet_type': ['haar', 'db4', 'sym4'],  # reduced from 6
        'wavelet_level': [1, 2],  # kept at 2
        'sigma': [2.0, 4.0, 6.0],  # reduced from 7
        'cov_reg': [0.1, 0.01, 0.001],  # reduced from 6
        'wavelet_kept_subbands': [
            ['LL', 'LH', 'HL'],
            ['LL', 'LH', 'HH'],
            ['LL', 'HL', 'HH'],
            ['LL', 'LH', 'HL', 'HH']
        ]
    }

    # run the parameter search stage
    stage1_results_path = run_parameter_search_stage(
        base_args,
        phase1_reduced_param_grid,
        classes=base_args.classes
    )

    print("\n--- Reduced Phase 1 Script Finished ---")
    if stage1_results_path:
        print(f"Phase 1 results saved in: {stage1_results_path}")
        print("\nNEXT STEP: Analyze these results to find the best combination of")
        print("           (wavelet_type, wavelet_level, sigma, cov_reg).")
        print(f"           Use these best parameters to manually configure and run Phase 2 (Subband Ablation) for model {base_args.model}.")
    else:
        print("Phase 1 did not complete successfully.")

if __name__ == "__main__":
    main()
