# grid_search.py
import os
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from itertools import product
from tqdm import tqdm

# import utility modules
from utils import ResourceMonitor, save_results, manage_gpu_memory, timer

def run_single_wavelet_experiment(
    evaluator,
    feature_extractor,
    class_names: List[str],
    wavelet_type: str = 'haar',
    wavelet_level: int = 1,
    wavelet_kept_subbands: List[str] = ['LL', 'LH', 'HL'],
    sigma: float = 4.0,
    cov_reg: float = 0.01,
    save_path: Optional[str] = None,
    save_visualizations: bool = False,
    train_batch_size: int = 32,
    resource_monitor: Optional[ResourceMonitor] = None
) -> Dict[str, Any]:
    """
    Run a single wavelet experiment.
    """
    print(f"\n{'='*80}")
    print(f"Running single wavelet experiment for {feature_extractor.model_name}")
    print(f"Parameters: wavelet_type={wavelet_type}, wavelet_level={wavelet_level}, "
          f"wavelet_kept_subbands={wavelet_kept_subbands}, sigma={sigma}, cov_reg={cov_reg}")
    print(f"{'='*80}")

    start_time = time.time()

    vis_dir = None
    if save_visualizations and save_path:
        vis_dir = os.path.join(save_path, f"{feature_extractor.model_name}_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    class_results = []
    for cls in class_names:
        print(f"\nProcessing class: {cls}")
        result = evaluator.evaluate_single_class(
            cls,
            sigma=sigma,
            cov_reg=cov_reg,
            save_dir=vis_dir,
            train_batch_size=train_batch_size,
            wavelet_type=wavelet_type,
            wavelet_level=wavelet_level,
            wavelet_kept_subbands=wavelet_kept_subbands
        )
        class_results.append(result)
        print(f"Class {cls} - Image AUC: {result['img_auc']:.4f}, Pixel AUC: {result['pixel_auc']:.4f}")
        print(f"Time: {result['time']:.2f}s")

    if class_results:
        avg_img_auc = np.mean([r['img_auc'] for r in class_results])
        avg_pixel_auc = np.mean([r['pixel_auc'] for r in class_results])
        avg_time = np.mean([r['time'] for r in class_results])
        total_time = time.time() - start_time
        experiment_result = {
            'model': feature_extractor.model_name,
            'wavelet_type': wavelet_type,
            'wavelet_level': wavelet_level,
            'wavelet_kept_subbands': wavelet_kept_subbands,
            'sigma': sigma,
            'cov_reg': cov_reg,
            'avg_img_auc': avg_img_auc,
            'avg_pixel_auc': avg_pixel_auc,
            'avg_inference_time': avg_time,
            'time': total_time,
            'class_results': class_results
        }
        if save_path:
            save_results([experiment_result], os.path.join(save_path, f"{feature_extractor.model_name}_wavelet_experiment.json"))
        print(f"\n{feature_extractor.model_name} wavelet experiment completed")
        print(f"Average Image AUC: {avg_img_auc:.4f}, Pixel AUC: {avg_pixel_auc:.4f}")
        print(f"Time taken: {total_time:.2f} seconds")
        return experiment_result
    else:
        print(f"No valid results for {feature_extractor.model_name}")
        return {'model': feature_extractor.model_name, 'avg_img_auc': 0.0, 'avg_pixel_auc': 0.0, 'time': time.time() - start_time, 'error': 'No valid results'}

def run_wavelet_grid_search(
    evaluator,
    feature_extractor,
    class_names: List[str],
    param_grid: Dict[str, List[Any]],
    save_path: str,
    save_visualizations: bool = False,
    train_batch_size: int = 32,
    resource_monitor: Optional[ResourceMonitor] = None
) -> Tuple[List[Dict[str, Any]], Tuple]:
    """
    Run grid search for wavelet parameters.
    """
    import itertools
    wavelet_types = param_grid.get('wavelet_type', ['haar'])
    wavelet_levels = param_grid.get('wavelet_level', [1])
    wavelet_kept_subbands_list = param_grid.get('wavelet_kept_subbands', [['LL', 'LH', 'HL']])
    sigma_values = param_grid.get('sigma', [4.0])
    cov_reg_values = param_grid.get('cov_reg', [0.01])

    param_combinations = list(itertools.product(
        wavelet_types, wavelet_levels, wavelet_kept_subbands_list, sigma_values, cov_reg_values
    ))

    print(f"Running wavelet grid search with {len(param_combinations)} parameter combinations")
    results = []
    best_avg_img_auc = -float("inf")
    best_avg_pixel_auc = -float("inf")
    best_params_img = None
    best_params_pixel = None
    random_seed = 1024
    import random
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    result_prefix = f"{feature_extractor.model_name}_wavelet_grid_search"
    print(f"Using prefix: {result_prefix}")

    from tqdm import tqdm
    grid_start_time = time.time()
    for idx, params in enumerate(tqdm(param_combinations, desc=f"{feature_extractor.model_name} wavelet grid search")):
        wavelet_type, wavelet_level, wavelet_kept_subbands, sigma, cov_reg = params
        print(f"\n{'='*80}")
        print(f"Evaluating Combo {idx+1}/{len(param_combinations)}: type={wavelet_type}, level={wavelet_level}, "
              f"subbands={wavelet_kept_subbands}, sigma={sigma}, cov_reg={cov_reg}")
        print(f"{'='*80}")

        # --- calculate and log dw ---
        try:
            # feature_extractor has feature_dims (list[int]) and target_layers (list[str])
            # assuming feature_dims corresponds directly to target_layers
            layer_dims = feature_extractor.feature_dims
            num_selected_subbands = len(wavelet_kept_subbands)
            final_feature_dim_dw = sum(num_selected_subbands * c_l for c_l in layer_dims)
            print(f"  Calculated Final Feature Dimension (Dw): {final_feature_dim_dw}")
        except Exception as e:
            print(f"  Warning: Could not calculate Dw - {e}")
            final_feature_dim_dw = -1  # indicate failure
        # --- end calculate dw ---
        param_start_time = time.time()
        class_results = []
        for cls in class_names:
            print(f"\nProcessing class: {cls}")
            result = evaluator.evaluate_single_class(
                cls,  # correctly maps to 'class_name'
                # all subsequent arguments passed by keyword
                sigma=sigma,
                cov_reg=cov_reg,
                save_dir=save_visualizations and os.path.join(save_path, f"{result_prefix}_visualizations"),
                train_batch_size=train_batch_size,
                wavelet_type=wavelet_type,
                wavelet_level=wavelet_level,
                wavelet_kept_subbands=wavelet_kept_subbands
                # embedding_cache=... # optional
            )
            class_results.append(result)
        avg_img_auc = np.mean([r['img_auc'] for r in class_results])
        avg_pixel_auc = np.mean([r['pixel_auc'] for r in class_results])
        if avg_img_auc > best_avg_img_auc:
            best_avg_img_auc = avg_img_auc
            best_params_img = (wavelet_type, wavelet_level, wavelet_kept_subbands, sigma, cov_reg)
        if avg_pixel_auc > best_avg_pixel_auc:
            best_avg_pixel_auc = avg_pixel_auc
            best_params_pixel = (wavelet_type, wavelet_level, wavelet_kept_subbands, sigma, cov_reg)
        param_time = time.time() - param_start_time
        param_result = {
            'params': {  # structure params clearly
                'wavelet_type': wavelet_type,
                'wavelet_level': wavelet_level,
                'wavelet_kept_subbands': wavelet_kept_subbands,
                'sigma': sigma,
                'cov_reg': cov_reg,
            },
            'final_feature_dim_dw': final_feature_dim_dw,  # <-- added
            'avg_img_auc': avg_img_auc,
            'avg_pixel_auc': avg_pixel_auc,
            'time': param_time,
            'class_results': class_results
        }
        results.append(param_result)
        save_results(results, os.path.join(save_path, f"{result_prefix}_results_incremental.json"))
        print(f"Parameters result: Image AUC={avg_img_auc:.4f}, Pixel AUC={avg_pixel_auc:.4f}, Time={param_time:.2f}s")

    save_results(results, os.path.join(save_path, f"{result_prefix}_results_final.json"))
    if best_params_img:
        wavelet_type, wavelet_level, wavelet_kept_subbands, sigma, cov_reg = best_params_img
        print(f"\nBest parameters (Image AUC): type={wavelet_type}, level={wavelet_level}, subbands={wavelet_kept_subbands}, sigma={sigma}, cov_reg={cov_reg}")
        print(f"Best average Image AUC: {best_avg_img_auc:.4f}")
    if best_params_pixel:
        wavelet_type, wavelet_level, wavelet_kept_subbands, sigma, cov_reg = best_params_pixel
        print(f"Best parameters (Pixel AUC): type={wavelet_type}, level={wavelet_level}, subbands={wavelet_kept_subbands}, sigma={sigma}, cov_reg={cov_reg}")
        print(f"Best average Pixel AUC: {best_avg_pixel_auc:.4f}")
    from visualization import visualize_grid_search_results
    visualize_grid_search_results(results, save_path, prefix=result_prefix)
    total_time = time.time() - grid_start_time
    print(f"\nTotal grid search time: {total_time/60:.2f} minutes")
    best_params_dict = {
        'img_auc': best_params_img,
        'pixel_auc': best_params_pixel,
    }
    return results, best_params_dict

def run_comprehensive_grid_search(
    data_path: str,
    save_path: str,
    models: List[str],
    class_names: Optional[List[str]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    save_visualizations: bool = False,
    enable_resource_monitoring: bool = True,
    device: Optional[torch.device] = None,
    # dataset_type=config.dataset_type,
    dataset_type: str = 'auto'
) -> Dict[str, Any]:
    """
    Run a comprehensive grid search across models using wavelet methods.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_path, exist_ok=True)
    if class_names is None:
        from dataset import get_all_class_names
        class_names = get_all_class_names(data_path, dataset_type=dataset_type)
        print(f"Using all available classes: {class_names}")
    if experiment_config is None:
        wavelet_param_grid = {
            'wavelet_type': ['haar', 'db2', 'db4', 'sym4'],
            'wavelet_level': [1, 2],
            'wavelet_kept_subbands': [
                ['LL'],
                ['LL', 'LH'],
                ['LL', 'LH', 'HL'],
                ['LL', 'LH', 'HL', 'HH']
            ],
            'sigma': [2.0, 4.0, 6.0],
            'cov_reg': [0.01, 0.001]
        }
        experiment_config = {
            'wavelet_param_grid': wavelet_param_grid
        }
    all_results = {}
    start_time = time.time()
    summary_data = []
    for model_idx, model_name in enumerate(models):
        print(f"\n{'='*80}")
        print(f"Running grid search for model {model_name}")
        print(f"{'='*80}")
        model_save_path = os.path.join(save_path, model_name)
        os.makedirs(model_save_path, exist_ok=True)
        resource_monitor = None
        if enable_resource_monitoring:
            resource_monitor = ResourceMonitor(log_file=os.path.join(model_save_path, "resource_log.csv"))
        from models import FeatureExtractor
        from evaluator import PaDiMEvaluator
        feature_extractor = FeatureExtractor(model_name=model_name, device=device, resource_monitor=resource_monitor)
        evaluator = PaDiMEvaluator(feature_extractor=feature_extractor, data_path=data_path, save_path=model_save_path, test_batch_size=test_batch_size, resource_monitor=resource_monitor, device=device, dataset_type=dataset_type)
        print("\nRunning wavelet grid search...")
        wavelet_results, best_wavelet_params = run_wavelet_grid_search(
            evaluator=evaluator,
            feature_extractor=feature_extractor,
            class_names=class_names,
            param_grid=experiment_config['wavelet_param_grid'],
            save_path=model_save_path,
            save_visualizations=save_visualizations,
            train_batch_size=train_batch_size,
            resource_monitor=resource_monitor
        )
        best_wavelet_img_params = best_wavelet_params.get('img_auc') if isinstance(best_wavelet_params, dict) else best_wavelet_params
        best_wavelet_pixel_params = best_wavelet_params.get('pixel_auc') if isinstance(best_wavelet_params, dict) else None
        model_result = {
            'model': model_name,
            'wavelet_results': wavelet_results,
            'best_wavelet_params': best_wavelet_img_params,
            'best_wavelet_params_by_metric': best_wavelet_params,
            'best_wavelet_img_auc': max([r['avg_img_auc'] for r in wavelet_results]),
            'best_wavelet_pixel_auc': max([r['avg_pixel_auc'] for r in wavelet_results])
        }
        all_results[model_name] = model_result
        from utils import save_results
        save_results([model_result], os.path.join(model_save_path, "grid_search_summary.json"))
        summary_row = {
            'model': model_name,
            'best_wavelet_img_auc': model_result['best_wavelet_img_auc'],
            'best_wavelet_pixel_auc': model_result['best_wavelet_pixel_auc']
        }
        summary_data.append(summary_row)
        del feature_extractor, evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    try:
        from visualization import create_summary_visualizations
        create_summary_visualizations(summary_data, save_path, with_fusion=False)
    except Exception as e:
        print(f"Error creating summary visualizations: {e}")
    all_results_file = os.path.join(save_path, "all_grid_search_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(json_safe(all_results), f, indent=2)
    total_time = time.time() - start_time
    print(f"\nTotal grid search execution time: {total_time/60:.2f} minutes")
    return all_results
