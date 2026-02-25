"""
Utility functions and classes for PaDiM with Wavelet.
"""

import os
import time
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
from tqdm import tqdm

# optional resource monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

@contextmanager
def timer(name: str) -> None:
    """Context manager for timing code execution."""
    start = time.time()
    yield
    end = time.time()
    print(f"[TIMER] {name} took {end - start:.4f} seconds")

def log_gpu_memory(description: str) -> None:
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"[GPU MEMORY] {description}: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

def json_safe(obj):
    """Convert an object to a JSON-serializable version."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                k = '_'.join(map(str, k))
            result[k] = json_safe(v)
        return result
    elif isinstance(obj, list):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, tuple):
        return '_'.join(map(str, obj))
    elif hasattr(obj, 'to_dict'):
        return json_safe(obj.to_dict())
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    else:
        return obj

def save_results(results: List[Dict[str, Any]], filename: str) -> None:
    """Save results to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if key == 'class_results':
                serializable_result['class_results'] = []
                for class_result in value:
                    serializable_class_result = {
                        k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                        for k, v in class_result.items()
                    }
                    serializable_result['class_results'].append(serializable_class_result)
            elif key == 'wavelet_kept_subbands':
                serializable_result[key] = list(value)
            else:
                if isinstance(value, (np.floating, torch.Tensor)):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                else:
                    serializable_result[key] = value
        serializable_results.append(serializable_result)
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def gaussian_filter(tensor: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
    """
    Apply a 2D Gaussian filter to the input tensor.
    """
    device = tensor.device
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    grid = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    gaussian_1d = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d_x = gaussian_1d.view(1, 1, kernel_size, 1).expand(1, 1, kernel_size, kernel_size)
    gaussian_2d_y = gaussian_1d.view(1, 1, 1, kernel_size).expand(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    blurred = F.conv2d(tensor, gaussian_2d_x, padding=(padding, padding))
    blurred = F.conv2d(blurred, gaussian_2d_y, padding=(padding, padding))
    return blurred.squeeze()

def save_anomaly_visualizations(
    image: torch.Tensor,
    score_map: np.ndarray,
    gt_mask: torch.Tensor,
    img_score: float,
    filename: str,
    output_dir: str,
    threshold: float = 0.5
) -> np.ndarray:
    """Save visualizations of anomaly detection results."""
    os.makedirs(output_dir, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="Clipping input data to the valid range")
    plt.figure(figsize=(20, 4))
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    plt.subplot(1, 5, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 5, 2)
    if isinstance(gt_mask, torch.Tensor):
        mask_np = gt_mask.squeeze().cpu().numpy()
    else:
        mask_np = gt_mask.squeeze() if hasattr(gt_mask, 'squeeze') else gt_mask
    plt.imshow(mask_np, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    normalized_score = np.copy(score_map)
    if normalized_score.min() != normalized_score.max():
        normalized_score = (normalized_score - normalized_score.min()) / (normalized_score.max() - normalized_score.min())
    plt.subplot(1, 5, 3)
    plt.imshow(normalized_score, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f'Anomaly Heatmap (Score: {img_score:.3f})')
    plt.axis('off')
    binary_mask = score_map > threshold
    plt.subplot(1, 5, 4)
    plt.imshow(binary_mask, cmap='gray')
    plt.title(f'Binary Mask (t={threshold})')
    plt.axis('off')
    plt.subplot(1, 5, 5)
    overlay = img_np.copy()
    heatmap_colored = plt.cm.jet(normalized_score)[:, :, :3]
    mask = normalized_score > threshold/2
    overlay[mask] = heatmap_colored[mask] * 0.7 + overlay[mask] * 0.3
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=200)
    plt.close()
    np.save(os.path.join(output_dir, f"{filename}_scoremap.npy"), score_map)
    return binary_mask

def adaptive_gaussian_filter(tensor: torch.Tensor, base_sigma: float = 1.0, min_sigma: float = 0.5) -> torch.Tensor:
    """
    Applies adaptive Gaussian filtering based on anomaly characteristics.
    """
    device = tensor.device
    threshold = torch.quantile(tensor.flatten(), 0.95)
    high_conf_mask = tensor > threshold
    labeled = torch.zeros_like(tensor, dtype=torch.int)
    current_label = 1
    with torch.no_grad():
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                if high_conf_mask[i, j] and labeled[i, j] == 0:
                    stack = [(i, j)]
                    while stack:
                        x, y = stack.pop()
                        if 0 <= x < tensor.shape[0] and 0 <= y < tensor.shape[1] and high_conf_mask[x, y] and labeled[x, y] == 0:
                            labeled[x, y] = current_label
                            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
                    region_size = (labeled == current_label).sum().item()
                    if region_size < 50:
                        region_sigma = min_sigma
                    else:
                        region_sigma = min(base_sigma, min_sigma + (base_sigma - min_sigma) * (region_size / 500))
                    region_mask = (labeled == current_label)
                    tensor_region = tensor * region_mask
                    smoothed_region = gaussian_filter(tensor_region, sigma=region_sigma)
                    tensor = tensor * (~region_mask) + smoothed_region
                    current_label += 1
    return tensor

def estimate_flops(model_name: str, use_dim_reduction: bool, d: int) -> float:
    """
    Estimate FLOPs for a given model configuration.
    """
    base_flops = {
        "resnet18": 1.8,
        "resnet34": 3.6,
        "wide_resnet50_2": 11.4,
        "efficientnet-b0": 0.4,
        "efficientnet-b1": 0.7,
        "efficientnet-b2": 1.0,
        "efficientnet-b3": 1.8,
        "efficientnet-b4": 4.2,
        "efficientnet-b5": 9.9,
        "efficientnet-b6": 19.0,
    }
    model_flops = base_flops.get(model_name, 5.0)
    if use_dim_reduction:
        dim_reduction_overhead = 0.05 * (d / 50)
    else:
        dim_reduction_overhead = 0.01
    mahalanobis_overhead = 0.03 * (d / 50)
    return model_flops * (1 + dim_reduction_overhead + mahalanobis_overhead)

def setup_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Set and return the appropriate device.
    """
    if gpu_id == -1 or not torch.cuda.is_available():
        print("Using CPU")
        return torch.device('cpu')
    num_gpus = torch.cuda.device_count()
    if gpu_id is not None and gpu_id >= num_gpus:
        print(f"GPU {gpu_id} not found. Using GPU 0.")
        gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    gpu_name = torch.cuda.get_device_name(gpu_id)
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
    print(f"Using GPU {gpu_id}: {gpu_name} with {total_memory:.2f} GB memory")
    torch.cuda.set_device(gpu_id)
    return device

def set_gpu_environment(gpu_id: Optional[int] = None) -> None:
    """
    Set the CUDA_VISIBLE_DEVICES environment variable.
    """
    if gpu_id is not None:
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Set CUDA_VISIBLE_DEVICES to {gpu_id}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("Disabled CUDA (using CPU)")

class ResourceMonitor:
    """Monitor system resource utilization during experiments."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or f"resource_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_data = []
        self.start_time = time.time()
        with open(self.log_file, 'w') as f:
            f.write("timestamp,phase,gpu_memory_used_mb,gpu_util_percent,cpu_percent,ram_used_mb,disk_read_mb,disk_write_mb,elapsed_time\n")

    def log(self, phase: str = "general") -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = time.time() - self.start_time
        gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        gpu_util = 0
        if torch.cuda.is_available():
            try:
                if GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_util = gpus[0].load * 100
            except:
                gpu_util = 0
        cpu_percent = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0
        ram_used = psutil.virtual_memory().used / (1024 * 1024) if PSUTIL_AVAILABLE else 0
        disk_io = psutil.disk_io_counters() if PSUTIL_AVAILABLE else None
        disk_read = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
        disk_write = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        log_entry = f"{timestamp},{phase},{gpu_memory_used:.2f},{gpu_util:.2f},{cpu_percent:.2f},{ram_used:.2f},{disk_read:.2f},{disk_write:.2f},{elapsed:.2f}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        self.log_data.append({
            'timestamp': timestamp,
            'phase': phase,
            'gpu_memory_used_mb': gpu_memory_used,
            'gpu_util_percent': gpu_util,
            'cpu_percent': cpu_percent,
            'ram_used_mb': ram_used,
            'disk_read_mb': disk_read,
            'disk_write_mb': disk_write,
            'elapsed_time': elapsed
        })
        return log_entry

    def summary(self) -> Dict[str, Dict[str, Any]]:
        if not self.log_data:
            return {}
        try:
            import pandas as pd
            df = pd.DataFrame(self.log_data)
            summary_df = df.groupby('phase').agg({
                'gpu_memory_used_mb': ['mean', 'max'],
                'gpu_util_percent': ['mean', 'max'],
                'cpu_percent': ['mean', 'max'],
                'ram_used_mb': ['mean', 'max'],
                'elapsed_time': ['count', 'sum']
            })
            json_safe_summary = {}
            for phase in summary_df.index:
                json_safe_summary[phase] = {}
                for col in summary_df.columns:
                    metric, agg_type = col
                    key = f"{metric}_{agg_type}"
                    json_safe_summary[phase][key] = float(summary_df.loc[phase, col])
            return json_safe_summary
        except ImportError:
            summary = {}
            phases = set(item['phase'] for item in self.log_data)
            for phase in phases:
                phase_data = [item for item in self.log_data if item['phase'] == phase]
                summary[phase] = {
                    'gpu_memory_used_mb_mean': np.mean([item['gpu_memory_used_mb'] for item in phase_data]),
                    'gpu_memory_used_mb_max': max([item['gpu_memory_used_mb'] for item in phase_data]),
                    'gpu_util_percent_mean': np.mean([item['gpu_util_percent'] for item in phase_data]),
                    'gpu_util_percent_max': max([item['gpu_util_percent'] for item in phase_data]),
                    'cpu_percent_mean': np.mean([item['cpu_percent'] for item in phase_data]),
                    'cpu_percent_max': max([item['cpu_percent'] for item in phase_data]),
                    'ram_used_mb_mean': np.mean([item['ram_used_mb'] for item in phase_data]),
                    'ram_used_mb_max': max([item['ram_used_mb'] for item in phase_data]),
                    'elapsed_time_count': len(phase_data),
                    'elapsed_time_sum': sum([item['elapsed_time'] for item in phase_data])
                }
            return summary

def manage_gpu_memory(description: str = ""):
    """Actively manage GPU memory by clearing caches."""
    gc.collect()
    if torch.cuda.is_available():
        before_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        torch.cuda.empty_cache()
        after_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        if description:
            print(f"Memory usage {description}: {after_memory:.2f} MB (freed {before_memory - after_memory:.2f} MB)")

class MemoryUsageMonitor:
    """Context manager to monitor memory usage during execution."""
    def __init__(self, description: str = ""):
        self.description = description
        self.start_memory = 0
        self.peak_memory = 0
        self.end_memory = 0

    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"Starting {self.description}: {self.start_memory:.2f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.end_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"Completed {self.description}:")
            print(f"  Start memory: {self.start_memory:.2f} MB")
            print(f"  Peak memory: {self.peak_memory:.2f} MB")
            print(f"  End memory: {self.end_memory:.2f} MB")
            print(f"  Memory increase: {self.end_memory - self.start_memory:.2f} MB")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
