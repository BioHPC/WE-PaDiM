# evaluator.py
"""
Evaluator module for PaDiM with Wavelet implementation.
Provides model evaluation utilities.
(Revised for DWT-before-concatenation)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import gc
from tqdm import tqdm # Added import

# Import necessary components from other files
from utils import save_results
from dataset import load_dataset, create_efficient_dataloaders
from utils import (
    ResourceMonitor, timer, log_gpu_memory,
    manage_gpu_memory, gaussian_filter, save_anomaly_visualizations,
    MemoryUsageMonitor
)
from metrics import (
    calculate_image_level_roc_auc, calculate_pixel_level_roc_auc,
    calculate_pro_score, calculate_mahalanobis_distance
)

# Import FeatureExtractor for type hinting and interaction
# Note: We no longer directly use wavelet_transform classes here for compression
# as it's handled within the FeatureExtractor now.
from models import FeatureExtractor, CachedFeatureExtractor # Import base and cached extractor

# --- EnhancedWaveletFusion class (commented out as likely incompatible without redesign) ---
# If needed, this class would need significant changes to work with the new workflow.
# class EnhancedWaveletFusion(nn.Module):
#     # ... (Original class definition) ...
#     pass

###############################
# Evaluator Class
###############################
class PaDiMEvaluator:
    """Evaluator class for PaDiM anomaly detection using DWT-before-concatenation features."""

    def __init__(
        self,
        feature_extractor: Union[FeatureExtractor, CachedFeatureExtractor], # Expects the revised FeatureExtractor
        data_path: str,
        save_path: str,
        test_batch_size: int,
        resource_monitor: Optional[ResourceMonitor] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator.

        Args:
            feature_extractor: Instance of the revised FeatureExtractor or CachedFeatureExtractor.
            data_path: Path to the dataset.
            save_path: Path to save results.
            test_batch_size: Batch size for processing test data.
            resource_monitor: Optional resource monitor.
            device: Device to use (CPU or GPU).
        """
        self.feature_extractor = feature_extractor
        self.data_path = data_path
        self.save_path = save_path
        self.test_batch_size = test_batch_size
        self.resource_monitor = resource_monitor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PaDiMEvaluator initialized to use device: {self.device}")

    def calculate_covariance(
        self,
        embedding_vectors: torch.Tensor,
        reg_factor: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mean and covariance matrices using a memory-efficient approach.
        Input embedding_vectors are expected to be the final DWT-before-concat features.

        Args:
            embedding_vectors: Tensor of shape (B, C_new, H_new, W_new)
            reg_factor: Regularization factor added to the diagonal of covariance.

        Returns:
            Tuple of (mean, covariance) tensors.
        """
        if self.resource_monitor:
            self.resource_monitor.log(phase="before_covariance_calc")

        B, C, H, W = embedding_vectors.size() # C, H, W are dimensions *after* DWT & concat
        P = H * W # Total number of spatial positions
        print(f"Calculating covariance for shape B={B}, C={C}, H={H}, W={W} (P={P})")
        embedding_vectors = embedding_vectors.view(B, C, P).to(self.device) # Ensure on correct device

        # --- Memory-Efficient Mean Calculation ---
        mean_sum = torch.zeros(C, P, device=self.device)
        batch_size_mean = min(64, B) # Process B samples in chunks for mean
        for i in range(0, B, batch_size_mean):
            batch_end = min(i + batch_size_mean, B)
            mean_sum += embedding_vectors[i:batch_end].sum(dim=0)
            manage_gpu_memory(f"Mean calc batch {i // batch_size_mean}")
        mean = mean_sum / B
        del mean_sum # Free memory

        # --- Memory-Efficient Covariance Calculation ---
        cov_sum = torch.zeros(C, C, P, device=self.device)
        batch_size_cov = min(16, B) # Smaller chunks for covariance (C*C can be large)
        print(f"Using covariance batch size: {batch_size_cov}")

        # Precompute mean outer product E[x]E[x]^T
        # Reshape mean for bmm: (P, C, 1) and (P, 1, C)
        mean_reshaped_T = mean.permute(1, 0).unsqueeze(2) # (P, C, 1)
        mean_reshaped = mean.permute(1, 0).unsqueeze(1)   # (P, 1, C)
        mean_outer = torch.bmm(mean_reshaped_T, mean_reshaped) # (P, C, C)
        mean_outer = mean_outer.permute(1, 2, 0) # Back to (C, C, P)

        for i in range(0, B, batch_size_cov):
            batch_start = i
            batch_end = min(i + batch_size_cov, B)
            batch = embedding_vectors[batch_start:batch_end] # Shape (batch_size_cov, C, P)
            print(f"  Processing covariance batch {i // batch_size_cov + 1}/{ (B + batch_size_cov - 1) // batch_size_cov } (Samples {batch_start}-{batch_end-1})")

            # Calculate E[x*x^T] for the batch
            # Reshape batch for bmm: (P, batch_size_cov, C)
            batch_t = batch.permute(2, 0, 1)
            # Calculate outer product sum for the batch: (P, C, C)
            outer_prod_sum_batch = torch.bmm(batch_t.transpose(1, 2), batch_t) # (P, C, batch_size_cov) x (P, batch_size_cov, C) -> (P, C, C)
            cov_sum += outer_prod_sum_batch.permute(1, 2, 0) # Permute back to (C, C, P) and accumulate

            del batch, batch_t, outer_prod_sum_batch
            manage_gpu_memory(f"Covariance calc batch {i // batch_size_cov}")

        # Final covariance: E[x*x^T] - E[x]E[x]^T
        cov = (cov_sum / B) - mean_outer
        del cov_sum, mean_outer # Free memory

        # Add regularization C = C + lambda * I
        identity = torch.eye(C, device=self.device).unsqueeze(2) # Shape (C, C, 1)
        cov += identity * reg_factor

        manage_gpu_memory("After covariance calculation")
        if self.resource_monitor:
            self.resource_monitor.log(phase="after_covariance_calc")
        print(f"Covariance calculation complete. Mean shape: {mean.shape}, Cov shape: {cov.shape}")
        return mean, cov # Shapes: (C, P), (C, C, P)


    def evaluate_test_set(
        self,
        test_dataloader,
        test_embedding_vectors: torch.Tensor,
        mean: torch.Tensor,
        cov: torch.Tensor,
        sigma: float,
        save_dir: Optional[str] = None,
        calculate_pro: bool = False,
        return_anomaly_maps: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the test set using pre-computed, DWT-processed embeddings.
        (Corrected image score calculation)
        """
        print("Starting test set evaluation...")
        if self.resource_monitor:
            self.resource_monitor.log(phase="test_evaluation_start")

        embedding_vectors = test_embedding_vectors.to(self.device)
        B, C, H, W_dim = embedding_vectors.size()
        P = H * W_dim
        print(f"Test embedding shape: B={B}, C={C}, H={H}, W={W_dim} (P={P})")
        embedding_vectors_reshaped = embedding_vectors.view(B, C, P)

        # --- Compute inverse covariance matrices (precision matrices) ---
        # (Same as previous version)
        print(f"Calculating {P} precision matrices (inverse covariance) for C={C}...")
        precision_matrices = []
        identity_reg = torch.eye(C, device=self.device) * 1e-5
        cov = cov.detach()
        precision_chunk_size = min(P, 512)
        for i in tqdm(range(0, P, precision_chunk_size), desc="Inverting Covariances"):
            chunk_end = min(i + precision_chunk_size, P)
            cov_chunk = cov[:, :, i:chunk_end]
            cov_chunk_inv_ready = (cov_chunk + identity_reg.unsqueeze(2)).permute(2, 0, 1)
            try:
                precision_chunk = torch.linalg.inv(cov_chunk_inv_ready)
                precision_matrices.append(precision_chunk.permute(1, 2, 0))
            except torch.linalg.LinAlgError as e:
                 print(f"\nWarning: Covariance matrix inversion failed for chunk {i}-{chunk_end}. Error: {e}. Trying pseudo-inverse.")
                 try:
                     pseudo_inv_chunk = torch.linalg.pinv(cov_chunk_inv_ready)
                     precision_matrices.append(pseudo_inv_chunk.permute(1, 2, 0))
                 except torch.linalg.LinAlgError as e_pinv:
                     print(f"ERROR: Pseudo-inverse also failed for chunk {i}-{chunk_end}. Error: {e_pinv}. Using identity matrix as fallback.")
                     identity_chunk = torch.eye(C, device=self.device).unsqueeze(2).expand(-1, -1, chunk_end - i)
                     precision_matrices.append(identity_chunk)
            del cov_chunk, cov_chunk_inv_ready, precision_chunk
            manage_gpu_memory(f"Precision matrix chunk {i // precision_chunk_size}")

        precision_tensor = torch.cat(precision_matrices, dim=2)
        print(f"Precision tensor shape: {precision_tensor.shape}")
        del precision_matrices, cov # Free cov memory

        # --- Calculate Mahalanobis distance ---
        print("Calculating Mahalanobis distances...")
        mean = mean.detach().to(self.device)
        precision_tensor = precision_tensor.detach().to(self.device)
        precision_tensor_reshaped = precision_tensor.permute(2, 0, 1)

        dist_list = calculate_mahalanobis_distance(embedding_vectors_reshaped, mean, precision_tensor_reshaped)
        print(f"Mahalanobis distance list shape: {dist_list.shape}") # Should be (P, B)
        dist_tensor = dist_list.t().reshape(B, H, W_dim) # Reshape back to (B, H_new, W_new)
        del precision_tensor, precision_tensor_reshaped, embedding_vectors_reshaped, mean, dist_list

        # --- Collect original test images and ground truth masks ---
        print("Collecting original test images and ground truth masks...")
        test_imgs_orig = []
        gt_labels_list = []
        gt_mask_list = []
        with torch.no_grad():
            for x, y, mask in tqdm(test_dataloader, desc="Loading GT data"):
                test_imgs_orig.extend(x.cpu())
                gt_labels_list.extend(y.cpu().numpy())
                gt_mask_list.extend(mask.cpu())

        # --- Upsample score map and apply Gaussian smoothing ---
        if not test_imgs_orig:
            raise ValueError("No test images loaded.")
        orig_img_size = test_imgs_orig[0].shape[1:]
        print(f"Upsampling score map from {(H, W_dim)} to original image size {orig_img_size}...")
        # Keep the raw score map after upsampling and smoothing
        score_map_raw_upsampled = F.interpolate(dist_tensor.unsqueeze(1), size=orig_img_size, mode='bilinear', align_corners=False).squeeze(1)
        del dist_tensor

        print(f"Applying Gaussian smoothing with sigma={sigma}...")
        # Create a tensor to store smoothed maps
        score_map_smoothed = torch.zeros_like(score_map_raw_upsampled)
        for i in tqdm(range(score_map_raw_upsampled.shape[0]), desc="Smoothing Scores"):
            # Perform smoothing on GPU for speed, then move to CPU
            score_map_smoothed[i] = gaussian_filter(score_map_raw_upsampled[i].to(self.device), sigma=sigma).cpu()
            if i % 10 == 0: manage_gpu_memory(f"Smoothing image {i}")
        del score_map_raw_upsampled

        # --- *** Calculate Image Level Score (BEFORE per-image normalization) *** ---
        print("Calculating image-level scores (using max pixel from smoothed map)...")
        img_scores = torch.amax(score_map_smoothed.view(B, -1), dim=1)

        # --- Normalize scores PER IMAGE (for Pixel AUC, PRO, Visualization) ---
        print("Normalizing scores (per image) for pixel metrics...")
        scores_normalized_per_image = torch.zeros_like(score_map_smoothed)
        for i in tqdm(range(score_map_smoothed.shape[0]), desc="Normalizing Scores"):
            img_map = score_map_smoothed[i]
            min_val, max_val = img_map.min(), img_map.max()
            if max_val > min_val:
                scores_normalized_per_image[i] = (img_map - min_val) / (max_val - min_val)

        # --- Prepare data for metrics ---
        img_scores_np = img_scores.cpu().numpy() # Use the scores derived *before* per-image normalization
        scores_for_pixel_metrics_np = scores_normalized_per_image.cpu().numpy() # Use per-image normalized scores for pixel metrics
        gt_labels_np = np.array(gt_labels_list)
        try:
            gt_mask_tensor = torch.stack(gt_mask_list).squeeze(1).cpu().numpy()
        except Exception as e:
             print(f"Error stacking GT masks: {e}. Check mask dimensions.")
             gt_mask_tensor = np.zeros_like(scores_for_pixel_metrics_np)

        # --- Optional Diagnostics (can be kept or removed) ---
        print("\n--- AUC Diagnostics (using pre-norm scores) ---")
        print(f"Shape img_scores_np: {img_scores_np.shape}, dtype: {img_scores_np.dtype}")
        print(f"Shape gt_labels_np: {gt_labels_np.shape}, dtype: {gt_labels_np.dtype}")
        unique_labels, label_counts = np.unique(gt_labels_np, return_counts=True)
        print(f"Unique GT labels: {unique_labels}, Counts: {label_counts}")
        if len(unique_labels) < 2: print("WARNING: Ground truth labels contain only one class!")
        if len(img_scores_np) > 0:
             score_min, score_max = np.min(img_scores_np), np.max(img_scores_np)
             score_mean, score_std = np.mean(img_scores_np), np.std(img_scores_np)
             print(f"Image Score Stats: Min={score_min:.6f}, Max={score_max:.6f}, Mean={score_mean:.6f}, StdDev={score_std:.6f}")
             if score_std < 1e-9: print("WARNING: Image scores still have near-zero standard deviation!")
             print("First 5 scores vs labels:")
             for i in range(min(5, len(img_scores_np))): print(f"  Score: {img_scores_np[i]:.6f}, Label: {gt_labels_np[i]}")
        else: print("WARNING: No image scores generated.")
        print("--- End AUC Diagnostics ---\n")
        # --- End Diagnostics ---

        # --- Calculate Metrics ---
        print("Calculating evaluation metrics...")
        img_roc_auc, pixel_roc_auc, pro_score = 0.0, 0.0, 0.0
        if len(unique_labels) >= 2 and len(img_scores_np) > 0:
             try:
                 # Use img_scores_np (derived before per-image norm) for Image AUC
                 img_roc_auc = calculate_image_level_roc_auc(img_scores_np, gt_labels_np)
                 # Use scores_for_pixel_metrics_np (per-image norm) for Pixel AUC
                 pixel_roc_auc = calculate_pixel_level_roc_auc(scores_for_pixel_metrics_np, gt_mask_tensor)
             except Exception as e:
                 print(f"ERROR calculating metrics: {e}")
                 img_roc_auc, pixel_roc_auc, pro_score = 0.0, 0.0, 0.0
        else:
            print("Skipping AUC calculation due to insufficient unique labels or scores.")

        # --- Save Visualizations ---
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving visualizations to {save_dir}...")
            vis_threshold = 0.5 # Threshold for binary mask in visualization
            num_saved = 0
            max_vis_save = 50
            # Use the PER-IMAGE NORMALIZED scores for visualization consistency
            scores_for_vis_np = scores_normalized_per_image.cpu().numpy()
            for i in tqdm(range(min(len(test_imgs_orig), max_vis_save)), desc="Saving Visuals"):
                # ... (rest of visualization saving loop using scores_for_vis_np[i] for the heatmap) ...
                 img = test_imgs_orig[i]
                 gt_label = gt_labels_np[i]
                 gt_mask = gt_mask_tensor[i]
                 score_map_vis = scores_for_vis_np[i] # Use normalized score map for consistent visualization
                 img_score_val = img_scores_np[i] # Use the actual image score value for title
                 base_name = f"{i:04d}_{'anomaly' if gt_label == 1 else 'normal'}_score{img_score_val:.4f}"
                 try:
                     save_anomaly_visualizations(img, score_map_vis, gt_mask, img_score_val, base_name, save_dir, threshold=vis_threshold)
                     num_saved += 1
                 except Exception as e:
                     print(f"Error saving visualization for image {i}: {e}")
            print(f"Saved {num_saved} visualizations.")

        manage_gpu_memory("After test evaluation")
        if self.resource_monitor:
            self.resource_monitor.log(phase="test_evaluation_end")
        print("Test set evaluation finished.")

        return {
            'img_roc_auc': float(img_roc_auc),
            'pixel_roc_auc': float(pixel_roc_auc),
            'pro_score': float(pro_score),
            # Return the per-image normalized scores if maps are requested
            'score_maps': scores_for_pixel_metrics_np if return_anomaly_maps else None,
            # Return the actual image scores used for Image AUC
            'img_scores': img_scores_np if return_anomaly_maps else None
        }

    # (evaluate_single_class remains the same as the previous version)
    def evaluate_single_class(
        self,
        class_name: str,
        sigma: float = 4.0,
        cov_reg: float = 0.01,
        save_dir: Optional[str] = None,
        calculate_pro: bool = False,
        train_batch_size: int = 32,
        wavelet_type: str = 'haar',
        wavelet_level: int = 1,
        wavelet_kept_subbands: Optional[List[str]] = None,
        return_anomaly_maps: bool = False,
        embedding_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single class using the DWT-before-concatenation workflow.
        (Handles both base and cached feature extractors)
        """
        print(f"\n--- Evaluating Class: {class_name} ---")
        print(f"Parameters: Sigma={sigma}, CovReg={cov_reg}, Wavelet={wavelet_type}, Level={wavelet_level}, Subbands={wavelet_kept_subbands}")
        class_start_time = time.time()
        if wavelet_kept_subbands is None: wavelet_kept_subbands = ['LL', 'LH', 'HL']
        wavelet_params = {'wavelet_type': wavelet_type, 'wavelet_level': wavelet_level, 'wavelet_kept_subbands': wavelet_kept_subbands}
        if self.resource_monitor: self.resource_monitor.log(phase=f"start_{class_name}_evaluation")

        with MemoryUsageMonitor(f"evaluate_{class_name}"):
            # 1. Load DataLoaders
            print("Loading datasets...")
            try:
                train_loader, test_loader = create_efficient_dataloaders(self.data_path, class_name, train_batch_size, self.test_batch_size)
            except Exception as e:
                 print(f"ERROR loading data for class {class_name}: {e}")
                 return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': time.time() - class_start_time, 'error': str(e)}

            # 2. Get Training Embeddings
            train_embeddings = None
            print("Getting training embeddings (DWT-before-concat)...")
            try:
                 with MemoryUsageMonitor("feature_extraction_train"):
                     if isinstance(self.feature_extractor, CachedFeatureExtractor):
                         train_embeddings = self.feature_extractor.get_embedding_vectors(dataloader=train_loader, class_name=class_name, wavelet_params=wavelet_params)
                     else:
                         train_embeddings = self.feature_extractor.get_embedding_vectors(dataloader=train_loader, wavelet_params=wavelet_params)
            except Exception as e:
                 print(f"ERROR getting training embeddings for {class_name}: {e}")
                 import traceback; traceback.print_exc()
                 current_time = time.time(); return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': f"Train embedding extraction failed: {e}"}
            if train_embeddings is None:
                 print(f"ERROR: Training embedding extraction returned None for {class_name}.")
                 current_time = time.time(); return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': "Train embedding extraction returned None"}
            train_embeddings = train_embeddings.to(self.device); manage_gpu_memory("After getting training embeddings")

            # 3. Calculate Mean and Covariance
            print(f"Calculating covariance matrix for {class_name}...")
            try:
                 with MemoryUsageMonitor("covariance_calculation"):
                     mean, cov = self.calculate_covariance(train_embeddings, reg_factor=cov_reg)
                 del train_embeddings; manage_gpu_memory("After training covariance calculation")
            except Exception as e:
                 print(f"ERROR calculating covariance for {class_name}: {e}")
                 import traceback; traceback.print_exc()
                 current_time = time.time(); return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': f"Covariance calculation failed: {e}"}

            # 4. Get Test Embeddings
            print("Getting test embeddings (DWT-before-concat)...")
            test_embeddings = None
            try:
                 with MemoryUsageMonitor("feature_extraction_test"):
                     if isinstance(self.feature_extractor, CachedFeatureExtractor):
                         test_embeddings = self.feature_extractor.get_embedding_vectors(dataloader=test_loader, class_name=f"{class_name}_test", wavelet_params=wavelet_params)
                     else:
                         test_embeddings = self.feature_extractor.get_embedding_vectors(dataloader=test_loader, wavelet_params=wavelet_params)
            except Exception as e:
                 print(f"ERROR getting test embeddings for {class_name}: {e}")
                 import traceback; traceback.print_exc()
                 del mean, cov; current_time = time.time(); return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': f"Test embedding extraction failed: {e}"}
            if test_embeddings is None:
                 print(f"ERROR: Test embedding extraction returned None for {class_name}.")
                 del mean, cov; current_time = time.time(); return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': "Test embedding extraction returned None"}
            test_embeddings = test_embeddings.to(self.device); manage_gpu_memory("After getting test embeddings")

            # 5. Evaluate Test Set
            print(f"Evaluating test set for {class_name}...")
            try:
                 with MemoryUsageMonitor("test_evaluation"):
                     results = self.evaluate_test_set(test_dataloader=test_loader, test_embedding_vectors=test_embeddings, mean=mean, cov=cov, sigma=sigma, save_dir=save_dir, calculate_pro=calculate_pro, return_anomaly_maps=return_anomaly_maps)
                 del test_embeddings, mean, cov; manage_gpu_memory(f"After evaluating test set for {class_name}")
            except Exception as e:
                 print(f"ERROR evaluating test set for {class_name}: {e}")
                 import traceback; traceback.print_exc()
                 try: del test_embeddings, mean, cov
                 except NameError: pass
                 manage_gpu_memory(f"After error during test evaluation for {class_name}")
                 current_time = time.time(); return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': f"Test set evaluation failed: {e}"}

        total_time = time.time() - class_start_time
        print(f"--- Finished Class: {class_name} in {total_time:.2f} seconds ---")
        img_auc_res = results.get('img_roc_auc', 0.0); pix_auc_res = results.get('pixel_roc_auc', 0.0); pro_res = results.get('pro_score', 0.0)
        print(f"  Image AUC: {img_auc_res:.4f}, Pixel AUC: {pix_auc_res:.4f}, PRO Score: {pro_res:.4f}")
        ret = {'class': class_name, 'img_auc': img_auc_res, 'pixel_auc': pix_auc_res, 'pro_score': pro_res, 'time': total_time}
        if return_anomaly_maps: ret['score_maps'], ret['img_scores'] = results.get('score_maps'), results.get('img_scores')
        if self.resource_monitor: self.resource_monitor.log(phase=f"end_{class_name}_evaluation")
        return ret
    
    def evaluate_single_class(
        self,
        # model parameter removed
        class_name: str,
        sigma: float = 4.0,
        cov_reg: float = 0.01,
        save_dir: Optional[str] = None,
        calculate_pro: bool = False,
        train_batch_size: int = 32,
        # use_wavelets parameter removed
        wavelet_type: str = 'haar',
        wavelet_level: int = 1,
        wavelet_kept_subbands: Optional[List[str]] = None,
        return_anomaly_maps: bool = False,
        embedding_cache: Optional[Dict[str, torch.Tensor]] = None # External cache dict (less used now)
    ) -> Dict[str, Any]:
        """
        Evaluate a single class using the DWT-before-concatenation workflow.
        (Handles both base and cached feature extractors)
        """
        print(f"\n--- Evaluating Class: {class_name} ---")
        print(f"Parameters: Sigma={sigma}, CovReg={cov_reg}, Wavelet={wavelet_type}, Level={wavelet_level}, Subbands={wavelet_kept_subbands}")

        # Define class_start_time early
        class_start_time = time.time()

        if wavelet_kept_subbands is None:
            wavelet_kept_subbands = ['LL', 'LH', 'HL']

        wavelet_params = {
            'wavelet_type': wavelet_type,
            'wavelet_level': wavelet_level,
            'wavelet_kept_subbands': wavelet_kept_subbands
        }

        if self.resource_monitor:
            self.resource_monitor.log(phase=f"start_{class_name}_evaluation")

        # Use a memory usage monitor for the entire evaluation of this class
        with MemoryUsageMonitor(f"evaluate_{class_name}"):
            # 1. Load DataLoaders
            print("Loading datasets...")
            try:
                train_loader, test_loader = create_efficient_dataloaders(self.data_path, class_name, train_batch_size, self.test_batch_size)
            except Exception as e:
                 print(f"ERROR loading data for class {class_name}: {e}")
                 return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': time.time() - class_start_time, 'error': str(e)}


            # 2. Get Training Embeddings
            train_embeddings = None
            print("Getting training embeddings (DWT-before-concat)...")
            try:
                 with MemoryUsageMonitor("feature_extraction_train"):
                     # --- MODIFIED CALL ---
                     # Check if the extractor is the cached version
                     if isinstance(self.feature_extractor, CachedFeatureExtractor):
                         # Cached extractor handles class_name for caching keys
                         train_embeddings = self.feature_extractor.get_embedding_vectors(
                             dataloader=train_loader,
                             class_name=class_name, # Pass class name for caching
                             wavelet_params=wavelet_params
                         )
                     else:
                         # Base extractor does not use class_name
                         train_embeddings = self.feature_extractor.get_embedding_vectors(
                             dataloader=train_loader,
                             wavelet_params=wavelet_params # No class_name here
                         )
                     # --- END MODIFIED CALL ---

            except Exception as e:
                 print(f"ERROR getting training embeddings for {class_name}: {e}")
                 # Added traceback print for debugging
                 import traceback
                 traceback.print_exc()
                 # Ensure time calculation works even if class_start_time was just defined
                 current_time = time.time()
                 return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': f"Train embedding extraction failed: {e}"}

            # Ensure embeddings are on the correct device for covariance calculation
            if train_embeddings is None: # Check if extraction failed silently
                 print(f"ERROR: Training embedding extraction returned None for {class_name}.")
                 current_time = time.time()
                 return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': "Train embedding extraction returned None"}

            train_embeddings = train_embeddings.to(self.device)
            manage_gpu_memory("After getting training embeddings")

            # 3. Calculate Mean and Covariance
            print(f"Calculating covariance matrix for {class_name}...")
            try:
                 with MemoryUsageMonitor("covariance_calculation"):
                     mean, cov = self.calculate_covariance(train_embeddings, reg_factor=cov_reg)
                 del train_embeddings # Free memory immediately after use
                 manage_gpu_memory("After training covariance calculation")
            except Exception as e:
                 print(f"ERROR calculating covariance for {class_name}: {e}")
                 import traceback
                 traceback.print_exc()
                 current_time = time.time()
                 return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': f"Covariance calculation failed: {e}"}


            # 4. Get Test Embeddings
            print("Getting test embeddings (DWT-before-concat)...")
            test_embeddings = None
            try:
                 with MemoryUsageMonitor("feature_extraction_test"):
                    # --- MODIFIED CALL ---
                     if isinstance(self.feature_extractor, CachedFeatureExtractor):
                         test_embeddings = self.feature_extractor.get_embedding_vectors(
                             dataloader=test_loader,
                             class_name=f"{class_name}_test", # Use different key for test cache
                             wavelet_params=wavelet_params
                         )
                     else:
                         test_embeddings = self.feature_extractor.get_embedding_vectors(
                             dataloader=test_loader,
                             wavelet_params=wavelet_params # No class_name here
                         )
                    # --- END MODIFIED CALL ---

            except Exception as e:
                 print(f"ERROR getting test embeddings for {class_name}: {e}")
                 import traceback
                 traceback.print_exc()
                 del mean, cov # Clean up calculated stats
                 current_time = time.time()
                 return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': f"Test embedding extraction failed: {e}"}

            if test_embeddings is None: # Check if extraction failed silently
                 print(f"ERROR: Test embedding extraction returned None for {class_name}.")
                 del mean, cov # Clean up calculated stats
                 current_time = time.time()
                 return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': "Test embedding extraction returned None"}

            test_embeddings = test_embeddings.to(self.device)
            manage_gpu_memory("After getting test embeddings")

            # 5. Evaluate Test Set
            print(f"Evaluating test set for {class_name}...")
            try:
                 with MemoryUsageMonitor("test_evaluation"):
                     results = self.evaluate_test_set(
                         test_dataloader=test_loader, # Pass loader for GT images/masks
                         test_embedding_vectors=test_embeddings, # Pass processed embeddings
                         mean=mean,
                         cov=cov,
                         sigma=sigma,
                         save_dir=save_dir,
                         calculate_pro=calculate_pro,
                         return_anomaly_maps=return_anomaly_maps
                     )
                 del test_embeddings, mean, cov # Free memory after use
                 manage_gpu_memory(f"After evaluating test set for {class_name}")
            except Exception as e:
                 print(f"ERROR evaluating test set for {class_name}: {e}")
                 import traceback
                 traceback.print_exc()
                 # Attempt to clean up memory even on error
                 try: del test_embeddings, mean, cov
                 except NameError: pass
                 manage_gpu_memory(f"After error during test evaluation for {class_name}")
                 current_time = time.time()
                 return {'class': class_name, 'img_auc': 0.0, 'pixel_auc': 0.0, 'pro_score': 0.0, 'time': current_time - class_start_time, 'error': f"Test set evaluation failed: {e}"}

        # End of MemoryUsageMonitor block

        # Ensure total_time calculation uses the defined class_start_time
        total_time = time.time() - class_start_time
        print(f"--- Finished Class: {class_name} in {total_time:.2f} seconds ---")
        # Safely get results
        img_auc_res = results.get('img_roc_auc', 0.0)
        pix_auc_res = results.get('pixel_roc_auc', 0.0)
        pro_res = results.get('pro_score', 0.0)
        print(f"  Image AUC: {img_auc_res:.4f}, Pixel AUC: {pix_auc_res:.4f}, PRO Score: {pro_res:.4f}")

        ret = {
            'class': class_name,
            'img_auc': img_auc_res,
            'pixel_auc': pix_auc_res,
            'pro_score': pro_res,
            'time': total_time
        }
        if return_anomaly_maps:
            ret['score_maps'] = results.get('score_maps')
            ret['img_scores'] = results.get('img_scores')

        if self.resource_monitor:
            self.resource_monitor.log(phase=f"end_{class_name}_evaluation")

        return ret
