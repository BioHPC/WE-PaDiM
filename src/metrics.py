# metrics.py **
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from typing import Tuple, List, Dict, Any, Optional


def calculate_image_level_roc_auc(
    scores: torch.Tensor, 
    labels: torch.Tensor
) -> float:
    """
    Calculate image-level ROC AUC score.
    
    Args:
        scores: Anomaly scores tensor
        labels: Ground truth labels tensor
        
    Returns:
        Image-level ROC AUC score
    """
    # Convert to numpy if necessary
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    return roc_auc_score(labels, scores)


def calculate_pixel_level_roc_auc(
    scores: torch.Tensor, 
    masks: torch.Tensor
) -> float:
    """
    Calculate pixel-level ROC AUC score.
    
    Args:
        scores: Anomaly score maps tensor
        masks: Ground truth mask tensor
        
    Returns:
        Pixel-level ROC AUC score
    """
    # Flatten tensors
    if isinstance(scores, torch.Tensor):
        scores = scores.view(-1).cpu().numpy()
    else:
        scores = scores.flatten()
        
    if isinstance(masks, torch.Tensor):
        masks = masks.view(-1).cpu().numpy()
    else:
        masks = masks.flatten()
    
    return roc_auc_score(masks, scores)


def calculate_pro_score(scores, masks, max_steps=100, max_fpr=0.3):
    """
    Calculate PRO (Per-Region Overlap) score as defined in the PaDiM paper.
    
    Args:
        scores: Anomaly score maps
        masks: Ground truth masks
        max_steps: Number of thresholds to evaluate
        max_fpr: Maximum false positive rate to consider
    """
    from scipy import ndimage
    
    # Convert to numpy
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    # Ensure binary masks
    binary_masks = masks > 0.5
    
    # Skip if no anomalies
    if np.sum(binary_masks) == 0:
        return 0.0
    
    # Find connected components in ground truth
    labeled_mask, num_regions = ndimage.label(binary_masks)
    
    if num_regions == 0:
        return 0.0
    
    # Store region sizes for normalization
    region_sizes = np.array([(labeled_mask == i).sum() for i in range(1, num_regions+1)])
    
    # Generate thresholds (log space often works better)
    min_score, max_score = scores.min(), scores.max()
    if min_score == max_score:
        return 0.0
    thresholds = np.linspace(min_score, max_score, max_steps)
    
    # Track detection rates and FPRs
    region_detection_rates = []
    fprs = []
    
    for threshold in thresholds:
        # Binary prediction at this threshold
        pred_mask = scores >= threshold
        
        # Calculate per-region detection rates
        detection_rates = []
        for region_idx in range(1, num_regions + 1):
            # Get region mask
            region_mask = (labeled_mask == region_idx)
            
            # Calculate overlap (what percentage of region is detected)
            detection_rate = np.logical_and(pred_mask, region_mask).sum() / region_sizes[region_idx-1]
            detection_rates.append(detection_rate)
        
        # Average detection rate across all regions
        avg_detection_rate = np.mean(detection_rates)
        region_detection_rates.append(avg_detection_rate)
        
        # Calculate false positive rate
        gt_neg = ~binary_masks
        fp = np.logical_and(pred_mask, gt_neg).sum()
        tn = np.logical_and(~pred_mask, gt_neg).sum()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fprs.append(fpr)
    
    # Convert to arrays and sort by FPR
    detection_rates = np.array(region_detection_rates)
    fprs = np.array(fprs)
    
    # Sort by increasing FPR
    sort_idxs = np.argsort(fprs)
    fprs = fprs[sort_idxs]
    detection_rates = detection_rates[sort_idxs]
    
    # Find indices with FPR <= max_fpr
    valid_indices = np.where(fprs <= max_fpr)[0]
    
    if len(valid_indices) == 0:
        return 0.0
    
    # Calculate area under curve up to max_fpr using trapezoidal rule
    valid_fprs = fprs[valid_indices]
    valid_rates = detection_rates[valid_indices]
    
    # Make sure curve extends to max_fpr
    if valid_fprs[-1] < max_fpr:
        valid_fprs = np.append(valid_fprs, max_fpr)
        valid_rates = np.append(valid_rates, valid_rates[-1])
    
    # Calculate integral using trapezoidal rule and normalize by max_fpr
    pro_score = np.trapz(valid_rates, valid_fprs) / max_fpr
    
    return float(pro_score)

def calculate_mahalanobis_distance(
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    cov_inv: torch.Tensor,
    batch_size: int = 50
) -> torch.Tensor:
    """
    Calculate Mahalanobis distance for embeddings with memory-efficient implementation.
    """
    B, C, P = embeddings.shape
    device = embeddings.device
    distances = torch.zeros(P, B, device=device)
    
    # Process in smaller batches to limit memory usage
    for b_start in range(0, B, batch_size):
        b_end = min(b_start + batch_size, B)
        b_size = b_end - b_start
        
        # Process positions in chunks
        chunk_size = min(256, P)  # Adjust based on GPU memory
        for p_start in range(0, P, chunk_size):
            # Clear cache before processing each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            p_end = min(p_start + chunk_size, P)
            
            for pos in range(p_start, p_end):
                # Get vectors for this position
                pos_vectors = embeddings[b_start:b_end, :, pos]  # (batch_size, C)
                pos_mean = mean[:, pos]  # (C,)
                pos_cov_inv = cov_inv[pos]  # (C, C)
                
                # Calculate centered vectors
                centered = pos_vectors - pos_mean.unsqueeze(0)  # (batch_size, C)
                
                # Calculate Mahalanobis distance - fixed tensor dimensions
                # (batch_size, C) -> (batch_size, 1, C)
                # (C, C) -> Matrix multiplication
                # Result: (batch_size, 1, C)
                # Then multiply with centered.unsqueeze(2) -> (batch_size, C, 1)
                # Result: (batch_size, 1, 1)
                # Sum over last dimension -> (batch_size, 1)
                # Flatten to (batch_size)
                dist = torch.bmm(torch.bmm(centered.unsqueeze(1), pos_cov_inv.unsqueeze(0).expand(b_size, C, C)), 
                                  centered.unsqueeze(2)).view(b_size)
                
                # Store result
                distances[pos, b_start:b_end] = dist
    
    return distances


def calculate_precision_recall_f1(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predictions: Binary prediction array
        ground_truth: Binary ground truth array
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # Calculate true positives, false positives, false negatives
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def find_optimal_threshold(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    num_steps: int = 100
) -> Dict[str, Any]:
    """
    Find optimal threshold based on F1 score.
    
    Args:
        scores: Anomaly score array
        ground_truth: Binary ground truth array
        num_steps: Number of threshold steps to evaluate
        
    Returns:
        Dictionary with optimal threshold and metrics
    """
    # Flatten arrays
    scores_flat = scores.flatten()
    gt_flat = ground_truth.flatten()
    
    # Generate thresholds
    min_score = np.min(scores_flat)
    max_score = np.max(scores_flat)
    thresholds = np.linspace(min_score, max_score, num_steps)
    
    # Find threshold with best F1 score
    best_threshold = 0
    best_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
    
    for threshold in thresholds:
        predictions = (scores_flat >= threshold).astype(np.uint8)
        metrics = calculate_precision_recall_f1(predictions, gt_flat)
        
        # Update best threshold if F1 score improves
        if metrics['f1'] > best_metrics['f1']:
            best_threshold = threshold
            best_metrics = metrics
    
    return {
        'threshold': best_threshold,
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1']
    }
