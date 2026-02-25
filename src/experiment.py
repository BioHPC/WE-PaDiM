"""
Experiment runner module for PaDiM with Wavelet implementation.
Handles different types of experiments.
"""

import os
import gc
import sys
import time
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

# import modules
from models import FeatureExtractor
from evaluator import PaDiMEvaluator  # , run_wavelet_grid_search, run_single_wavelet_experiment
from grid_search import run_wavelet_grid_search, run_single_wavelet_experiment
from dataset import get_all_class_names
from utils import ResourceMonitor, estimate_flops, save_results, manage_gpu_memory, json_safe

# import visualization functions
from visualization import (
    generate_model_comparison_table, generate_paper_tables,
    generate_paper_figures, create_summary_visualizations
)

def run_model_experiment(
    model_name: str,
    data_path: str,
    save_path: str,
    class_names: List[str],
    experiment_type: str,
    config: Dict[str, Any],
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    save_visualizations: bool = False,
    resource_monitor: Optional[ResourceMonitor] = None,
    device: Optional[torch.device] = None,
    dataset_type: str = 'auto'
) -> Dict[str, Any]:
    """
    Run experiments for a single model using the wavelet approach.

    Args:
        model_name: Name of the model.
        data_path: Path to MVTec dataset.
        save_path: Path to save results.
        class_names: List of class names to evaluate.
        experiment_type: Type of experiment ('single' or 'grid_search').
        config: Experiment configuration (wavelet parameters).
        train_batch_size: Batch size for training.
        test_batch_size: Batch size for testing.
        resource_monitor: Optional resource monitor.
        device: Device to use (CPU or CUDA).

    Returns:
        Dictionary with experiment results.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_save_path = os.path.join(save_path, model_name)
    os.makedirs(model_save_path, exist_ok=True)

    if resource_monitor:
        resource_monitor.log(phase=f"start_experiment_{model_name}")

    feature_extractor = FeatureExtractor(
        model_name=model_name,
        device=device,
        resource_monitor=resource_monitor
    )

    evaluator = PaDiMEvaluator(
        feature_extractor=feature_extractor,
        data_path=data_path,
        save_path=model_save_path,
        test_batch_size=test_batch_size,
        resource_monitor=resource_monitor,
        device=device,
        dataset_type=dataset_type
    )

    if experiment_type == 'grid_search':
        # build a wavelet parameter grid from configuration.
        wavelet_param_grid = {
            'wavelet_type': config.get('wavelet_type', ['haar']),
            'wavelet_level': config.get('wavelet_level', [1]),
            'wavelet_kept_subbands': config.get('wavelet_kept_subbands', [['LL', 'LH', 'HL']]),
            'sigma': config.get('sigma', [4.0]),
            'cov_reg': config.get('cov_reg', [0.01])
        }
        results, best_params = run_wavelet_grid_search(
            evaluator=evaluator,
            feature_extractor=feature_extractor,
            class_names=class_names,
            param_grid=wavelet_param_grid,
            save_path=model_save_path,
            save_visualizations=save_visualizations,
            train_batch_size=train_batch_size,
            resource_monitor=resource_monitor
        )
        best_params_img = best_params.get('img_auc') if isinstance(best_params, dict) else best_params
        best_params_pixel = best_params.get('pixel_auc') if isinstance(best_params, dict) else None
        experiment_result = {
            'model_name': model_name,
            'best_params': best_params_img,
            'best_params_by_metric': best_params,
            'best_img_auc': max([r['avg_img_auc'] for r in results]),
            'best_pixel_auc': max([r['avg_pixel_auc'] for r in results]),
            'results': results
        }
    else:  # single experiment
        wavelet_type = config.get('wavelet_type', 'haar')
        wavelet_level = config.get('wavelet_level', 1)
        wavelet_kept_subbands = config.get('wavelet_kept_subbands', ['LL', 'LH', 'HL'])
        sigma = config.get('sigma', 4.0)
        cov_reg = config.get('cov_reg', 0.01)
        result = run_single_wavelet_experiment(
            evaluator=evaluator,
            feature_extractor=feature_extractor,
            class_names=class_names,
            wavelet_type=wavelet_type,
            wavelet_level=wavelet_level,
            wavelet_kept_subbands=wavelet_kept_subbands,
            sigma=sigma,
            cov_reg=cov_reg,
            save_path=model_save_path,
            save_visualizations=save_visualizations,
            train_batch_size=train_batch_size,
            resource_monitor=resource_monitor
        )
        experiment_result = {
            'model_name': model_name,
            'params': (wavelet_type, wavelet_level, wavelet_kept_subbands, sigma, cov_reg),
            'img_auc': result['avg_img_auc'],
            'pixel_auc': result['avg_pixel_auc'],
            'result': result
        }

    if resource_monitor:
        resource_monitor.log(phase=f"end_experiment_{model_name}")

    del feature_extractor
    del evaluator
    manage_gpu_memory(f"After {model_name} experiment")

    return experiment_result

def run_all_models(
    model_names: List[str],
    data_path: str,
    save_path: str,
    class_names: List[str],
    experiment_type: str,
    config: Dict[str, Any],
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    save_visualizations: bool = False,
    enable_resource_monitoring: bool = True,
    device: Optional[torch.device] = None,
    dataset_type: str = 'auto'
) -> Dict[str, Dict[str, Any]]:
    """
    Run experiments for multiple models using wavelet dimensionality reduction.

    Args:
        model_names: List of model names.
        data_path: Path to MVTec dataset.
        save_path: Path to save results.
        class_names: List of class names.
        experiment_type: 'single' or 'grid_search'.
        config: Experiment configuration (wavelet parameters).
        train_batch_size: Training batch size.
        test_batch_size: Test batch size.
        save_visualizations: Whether to save visualization images.
        enable_resource_monitoring: Whether to enable resource monitoring.
        device: Device to use.

    Returns:
        Dictionary mapping model names to their results.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(save_path, exist_ok=True)
    all_results = {}

    for model_name in model_names:
        print(f"\n\n{'='*100}")
        print(f"Running experiments for {model_name}")
        print(f"{'='*100}\n")

        resource_monitor = None
        if enable_resource_monitoring:
            resource_monitor = ResourceMonitor(
                log_file=os.path.join(save_path, f"resource_log_{model_name}.csv")
            )

        try:
            result = run_model_experiment(
                model_name=model_name,
                data_path=data_path,
                save_path=save_path,
                class_names=class_names,
                experiment_type=experiment_type,
                config=config,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                save_visualizations=save_visualizations,
                resource_monitor=resource_monitor,
                device=device,
                dataset_type=dataset_type
            )
            all_results[model_name] = result

            if enable_resource_monitoring and resource_monitor:
                summary = resource_monitor.summary()
                json_safe_summary = json_safe(summary)
                with open(os.path.join(save_path, f"{model_name}_resource_summary.json"), 'w') as f:
                    json.dump(json_safe_summary, f, indent=2)
        except Exception as e:
            print(f"Error running experiments for {model_name}: {e}")
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    if experiment_type == 'grid_search' and len(all_results) > 1:
        generate_model_comparison_table(all_results, save_path)

    return all_results

def run_paper_experiments(
    experiment_configs: List[Dict[str, Any]],
    data_path: str,
    save_path: str,
    class_subset: Optional[List[str]] = None,
    generate_visuals: bool = False,
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    enable_resource_monitoring: bool = True,
    device: Optional[torch.device] = None,
    dataset_type: str = 'auto'
) -> List[Dict[str, Any]]:
    """
    Run experiments for generating paper results with wavelet methodology.

    Args:
        experiment_configs: List of experiment configuration dictionaries.
        data_path: Path to MVTec dataset.
        save_path: Path to save results.
        class_subset: Optional subset of classes to evaluate.
        generate_visuals: Whether to generate visualizations.
        train_batch_size: Training batch size.
        test_batch_size: Test batch size.
        enable_resource_monitoring: Whether to enable resource monitoring.
        device: Device to use.

    Returns:
        List of experiment results.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(save_path, exist_ok=True)

    if class_subset is None:
        from dataset import get_all_class_names
        class_names = get_all_class_names(data_path, dataset_type=dataset_type)
    else:
        class_names = class_subset
    print(f"Running paper experiments on classes: {class_names}")

    all_results = []
    summary_data = []
    start_time = time.time()

    for config in experiment_configs:
        print(f"\n{'='*80}")
        print(f"Running experiment: {config['name']}")
        print(f"{'='*80}")

        experiment_name = config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_plus_')
        model = config['model']
        use_wavelets = config.get('use_wavelets', True)
        experiment_group = config.get('experiment_group', '')

        experiment_save_path = os.path.join(save_path, experiment_name)
        os.makedirs(experiment_save_path, exist_ok=True)

        resource_monitor = None
        if enable_resource_monitoring:
            resource_monitor = ResourceMonitor(
                log_file=os.path.join(experiment_save_path, "resource_log.csv")
            )

        feature_extractor = FeatureExtractor(
            model_name=model,
            device=device,
            resource_monitor=resource_monitor
        )

        evaluator = PaDiMEvaluator(
            feature_extractor=feature_extractor,
            data_path=data_path,
            save_path=experiment_save_path,
            test_batch_size=test_batch_size,
            resource_monitor=resource_monitor,
            device=device,
            dataset_type=dataset_type
        )

        wavelet_type = config.get('wavelet_type', 'haar')
        wavelet_level = config.get('wavelet_level', 1)
        wavelet_kept_subbands = config.get('wavelet_kept_subbands', ['LL', 'LH', 'HL'])
        sigma = config.get('sigma', 4.0)
        cov_reg = config.get('cov_reg', 0.01)

        print(f"Using wavelets: {wavelet_type}, level={wavelet_level}, subbands={wavelet_kept_subbands}")
        experiment_params = {
            'wavelet_type': wavelet_type,
            'wavelet_level': wavelet_level,
            'wavelet_kept_subbands': wavelet_kept_subbands,
            'sigma': sigma,
            'cov_reg': cov_reg
        }

        class_results = []
        start_exp_time = time.time()
        memory_usage = 0
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)

        for cls in class_names:
            print(f"\nProcessing class: {cls}")
            class_result = evaluator.evaluate_single_class(
                cls,  # correctly maps to 'class_name'
                # all subsequent arguments passed by keyword
                sigma=sigma,
                cov_reg=cov_reg,
                save_dir=generate_visuals and os.path.join(experiment_save_path, cls),
                train_batch_size=train_batch_size,
                wavelet_type=wavelet_type,
                wavelet_level=wavelet_level,
                wavelet_kept_subbands=wavelet_kept_subbands
                # embedding_cache=... # optional
            )
            class_results.append(class_result)
            if torch.cuda.is_available():
                memory_usage = max(memory_usage, torch.cuda.memory_allocated() / (1024 * 1024))
            print(f"Class {cls} - Image AUC: {class_result['img_auc']:.4f}, Pixel AUC: {class_result['pixel_auc']:.4f}")
            print(f"Time: {class_result['time']:.2f}s")

        avg_img_auc = np.mean([r['img_auc'] for r in class_results])
        avg_pixel_auc = np.mean([r['pixel_auc'] for r in class_results])
        avg_time = np.mean([r['time'] for r in class_results])
        total_time_exp = time.time() - start_exp_time

        d_equiv = wavelet_level * len(wavelet_kept_subbands) * 32  # approximate measure for flops estimation
        flops = estimate_flops(model, True, d_equiv)

        result = {
            'name': config['name'],
            'model': model,
            'use_wavelets': use_wavelets,
            'experiment_group': experiment_group,
            'avg_img_auc': avg_img_auc,
            'avg_pixel_auc': avg_pixel_auc,
            'avg_inference_time': avg_time,
            'memory_usage_mb': memory_usage,
            'flops_g': flops,
            'total_time': total_time_exp,
            'class_results': class_results
        }
        result.update({
            'wavelet_type': wavelet_type,
            'wavelet_level': wavelet_level,
            'wavelet_kept_subbands': wavelet_kept_subbands,
            'sigma': sigma,
            'cov_reg': cov_reg
        })

        save_results([result], os.path.join(experiment_save_path, "results.json"))
        all_results.append(result)

        if enable_resource_monitoring and resource_monitor:
            summary = resource_monitor.summary()
            json_safe_summary = json_safe(summary)
            with open(os.path.join(experiment_save_path, "resource_summary.json"), 'w') as f:
                json.dump(json_safe_summary, f, indent=2)

        print(f"\nExperiment completed: {config['name']}")
        print(f"Average Image AUC: {avg_img_auc:.4f}, Pixel AUC: {avg_pixel_auc:.4f}")
        print(f"Average inference time: {avg_time:.2f}s, Memory usage: {memory_usage:.2f}MB")

        del feature_extractor
        del evaluator
        manage_gpu_memory()

    if all_results:
        generate_paper_tables(all_results, experiment_group, save_path)
        generate_paper_figures(all_results, experiment_group, save_path)

    print(f"\nTotal grid search execution time: {(time.time()-start_time)/60:.2f} minutes")
    return all_results

if __name__ == "__main__":
    # for debugging or standalone run
    print("This module is meant to be imported and run via main.py")
