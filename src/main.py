"""
Main script for PaDiM with Wavelet implementation.
Entry point for running experiments.
"""

import os
import resource
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Clipping input data to the valid range")
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
print(f"File descriptor limit increased from {soft} to {min(4096, hard)}")

import sys
import json
import torch
import random
import time
from datetime import datetime

# Assuming Config and parse_args are correctly defined in config.py
from config import Config, parse_args, MVTEC_CLASSES
# Assuming get_all_class_names is correctly defined in dataset.py
from dataset import get_all_class_names
from experiment import run_all_models, run_paper_experiments
from grid_search import run_comprehensive_grid_search # Import from grid_search.py instead

from utils import setup_device, set_gpu_environment

# Print initial GPU info if available
if torch.cuda.is_available():
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/(1024*1024):.2f} MB")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.2f} GB")


def run_experiment(config: Config):
    """
    Main execution logic, accepting a pre-configured Config object.

    Args:
        config: A Config object containing all experiment settings.
    """
    print("--- Running Experiment with Provided Configuration ---")
    print(config) # Print the configuration being used

    # Use getattr to safely get comprehensive_grid_search (default to False if not provided)
    comprehensive_grid_search = getattr(config, "comprehensive_grid_search", False)

    # Set CUDA_VISIBLE_DEVICES if gpu_id is provided in the config.
    if hasattr(config, 'gpu_id') and config.gpu_id is not None:
        set_gpu_environment(config.gpu_id)

    # Set up device (GPU or CPU)
    device = setup_device(config.gpu_id)

    # Set random seeds for reproducibility.
    random.seed(1024)
    torch.manual_seed(1024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)

    # Create a timestamp based on the config's save_path if available, otherwise generate one.
    # This assumes the save_path in config is already unique per stage/run.
    save_path = config.save_path
    timestamp_from_path = os.path.basename(save_path) # Extract unique part if needed

    # Ensure save path exists (it should be created by the calling script)
    os.makedirs(save_path, exist_ok=True)

    # Save final configuration to file for reference.
    config_path = os.path.join(save_path, "config.json")
    config.save(config_path)

    # Determine class names.
    if config.classes is None or config.classes == ["all"]:
        # Use default MVTec classes defined in config.py or get dynamically
        class_names = get_all_class_names(config.data_path) if os.path.exists(config.data_path) else MVTEC_CLASSES
    else:
        class_names = config.classes

    print(f"Running experiments with models: {config.models}")
    print(f"Using classes: {class_names}")

    start_time = time.time()

    # If running a comprehensive grid search (controlled by config now).
    if comprehensive_grid_search:
        # Make sure grid search parameters are in the config object
        grid_search_models = getattr(config, "grid_search_models", config.models)
        grid_search_classes = getattr(config, "grid_search_classes", None) # Use None for all
        enable_adaptive_fusion_gs = not getattr(config, "skip_fusion_grid_search", False)

        print("Running comprehensive grid search...")
        print(f"Models: {grid_search_models}")
        print(f"Classes: {grid_search_classes if grid_search_classes else 'All available classes'}")
        print("Using wavelet approach only.")

        results = run_comprehensive_grid_search(
            data_path=config.data_path,
            save_path=save_path, # Use the path from config
            models=grid_search_models,
            class_names=grid_search_classes,
            # Pass experiment_config derived from the main config if needed by grid search
            # experiment_config=config.get_experiment_configs(), # Or pass specific grid search config parts
            save_visualizations=config.save_visualizations,
            calculate_pro=config.calculate_pro_score,
            train_batch_size=config.train_batch_size,
            test_batch_size=config.test_batch_size,
            enable_resource_monitoring=config.enable_resource_monitoring,
            device=device,
            enable_adaptive_fusion=enable_adaptive_fusion_gs
        )

        print(f"\nComprehensive grid search completed. Results saved to {save_path}")

    # Handle 'paper' or 'single'/'grid_search' experiment types
    elif config.experiment_type == "paper":
        experiment_configs_list = config.get_experiment_configs() # Should return list for paper
        # Add experiment_group if needed by run_paper_experiments
        for exp_conf_item in experiment_configs_list:
            exp_conf_item['experiment_group'] = config.experiment_group
        results = run_paper_experiments(
            experiment_configs=experiment_configs_list,
            data_path=config.data_path,
            save_path=save_path,
            class_subset=class_names,
            generate_visuals=config.save_visualizations,
            calculate_pro=config.calculate_pro_score,
            train_batch_size=config.train_batch_size,
            test_batch_size=config.test_batch_size,
            enable_resource_monitoring=config.enable_resource_monitoring,
            device=device
        )
        print("\n=== Paper Experiments Completed ===")
        print(f"Results saved to: {save_path}")
    else: # Handles 'single_model' and 'grid_search' from aux script or CLI
        if config.experiment_type == 'grid_search':
            # Original behavior for grid search type from CLI/config
            experiment_config_param = config.get_experiment_configs()
        elif config.experiment_type == 'single_model':
            # NEW: For 'single_model' (e.g., from ablation script),
            # construct the parameter dict directly from the main config object.
            # Note: Ensure keys match what run_model_experiment expects in its 'config' dict
            experiment_config_param = {
                'wavelet_type': config.wavelet_type,
                'wavelet_level': config.wavelet_level,
                'wavelet_kept_subbands': config.wavelet_kept_subbands,
                'sigma': config.gaussian_sigma, # Use config.gaussian_sigma for the 'sigma' key
                'cov_reg': config.cov_reg_epsilon, # Use config.cov_reg_epsilon for the 'cov_reg' key
                # Add other potentially relevant parameters expected by run_model_experiment
                'use_adaptive_fusion': getattr(config, 'use_adaptive_fusion', False),
                'fusion_levels': getattr(config, 'fusion_levels', [1, 2]),
                'fusion_learn_weights': getattr(config, 'fusion_learn_weights', False)
            }
        else:
            # Handle unexpected types if necessary
            raise ValueError(f"Unsupported experiment_type '{config.experiment_type}' encountered in main.run_experiment")
    results = run_all_models(
        model_names=config.models,
        data_path=config.data_path,
        save_path=save_path,
        class_names=class_names,
        experiment_type=config.experiment_type, # Pass the type
        config=experiment_config_param, # Pass the correctly formatted dict
        train_batch_size=config.train_batch_size,
        test_batch_size=config.test_batch_size,
        save_visualizations=config.save_visualizations,
        calculate_pro=config.calculate_pro_score,
        enable_resource_monitoring=config.enable_resource_monitoring,
        device=device
    )
    print("\n=== Experiments Completed ===")
    total_time = time.time() - start_time
    print(f"\nTotal execution time for this run: {total_time/60:.2f} minutes")
    print(f"--- Finished Experiment for Config in {save_path} ---")


def main():
    """
    Original main entry point for command-line execution.
    Parses args and runs the experiment.
    """
    print("--- Running Main Script via Command Line ---")
    # Parse command-line arguments.
    args = parse_args()
    # Create configuration using the parsed arguments.
    config = Config(args)
    # Add timestamp for uniqueness if run directly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.save_path = os.path.join(config.save_path, f"run_{timestamp}")
    os.makedirs(config.save_path, exist_ok=True)
    # Run the experiment logic with the created config
    run_experiment(config)


if __name__ == "__main__":
    main()
