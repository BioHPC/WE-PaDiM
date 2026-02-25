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
from tqdm import tqdm

# import necessary components from other files
from utils import save_results
from dataset import load_dataset, create_efficient_dataloaders
from utils import (
    ResourceMonitor, timer, log_gpu_memory,
    manage_gpu_memory, gaussian_filter, save_anomaly_visualizations,
    MemoryUsageMonitor
)
from metrics import (
    calculate_image_level_roc_auc, calculate_pixel_level_roc_auc,
    calculate_mahalanobis_distance
)

# import featureextractor for type hinting and interaction
# note: we no longer directly use wavelet_transform classes here for compression
# as it's handled within the featureextractor now.
from models import FeatureExtractor, CachedFeatureExtractor  # import base and cached extractor

# --- enhancedwaveletfusion class (commented out as likely incompatible without redesign) ---
# if needed, this class would need significant changes to work with the new workflow.
# class enhancedwaveletfusion(nn.module):
# # ... (original class definition) ...
# pass

# evaluator class
class PaDiMEvaluator:
    """Evaluator class for PaDiM anomaly detection using DWT-before-concatenation features."""

    def __init__(
        self,
        feature_extractor: Union[FeatureExtractor, CachedFeatureExtractor],  # expects the revised featureextractor
        data_path: str,
        save_path: str,
        test_batch_size: int,
        resource_monitor: Optional[ResourceMonitor] = None,
        device: Optional[torch.device] = None,
        dataset_type: str = 'auto'
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
        self.dataset_type = dataset_type
        print(f"Dataset type: {self.dataset_type}")

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

        B, C, H, W = embedding_vectors.size()  # c, h, w are dimensions *after* dwt & concat
        P = H * W  # total number of spatial positions
        print(f"Calculating covariance for shape B={B}, C={C}, H={H}, W={W} (P={P})")
        embedding_vectors = embedding_vectors.to(self.device, dtype=torch.float64).view(B, C, P)

        # --- memory-efficient mean calculation ---
        mean_sum = torch.zeros(C, P, device=self.device, dtype=torch.float64)
        batch_size_mean = min(64, B)  # process b samples in chunks for mean
        for i in range(0, B, batch_size_mean):
            batch_end = min(i + batch_size_mean, B)
            mean_sum += embedding_vectors[i:batch_end].sum(dim=0)
            manage_gpu_memory(f"Mean calc batch {i // batch_size_mean}")
        mean = mean_sum / B
        del mean_sum  # free memory

        # --- memory-efficient covariance calculation ---
        cov_sum = torch.zeros(C, C, P, device=self.device, dtype=torch.float64)
        batch_size_cov = min(16, B)  # smaller chunks for covariance (c*c can be large)
        print(f"Using covariance batch size: {batch_size_cov}")

        # precompute mean outer product e[x]e[x]^t
        # reshape mean for bmm: (p, c, 1) and (p, 1, c)
        mean_reshaped_T = mean.permute(1, 0).unsqueeze(2)
        mean_reshaped = mean.permute(1, 0).unsqueeze(1)
        mean_outer = torch.bmm(mean_reshaped_T, mean_reshaped)
        mean_outer = mean_outer.permute(1, 2, 0)  # back to (c, c, p)

        for i in range(0, B, batch_size_cov):
            batch_start = i
            batch_end = min(i + batch_size_cov, B)
            batch = embedding_vectors[batch_start:batch_end]
            print(f"  Processing covariance batch {i // batch_size_cov + 1}/{ (B + batch_size_cov - 1) // batch_size_cov } (Samples {batch_start}-{batch_end-1})")

            # calculate e[x*x^t] for the batch
            # reshape batch for bmm: (p, batch_size_cov, c)
            batch_t = batch.permute(2, 0, 1)
            # calculate outer product sum for the batch: (p, c, c)
            outer_prod_sum_batch = torch.bmm(batch_t.transpose(1, 2), batch_t)
            cov_sum += outer_prod_sum_batch.permute(1, 2, 0)  # permute back to (c, c, p) and accumulate

            del batch, batch_t, outer_prod_sum_batch
            manage_gpu_memory(f"Covariance calc batch {i // batch_size_cov}")

        # final covariance: e[x*x^t] - e[x]e[x]^t
        cov = (cov_sum / B) - mean_outer
        del cov_sum, mean_outer  # free memory

        # add regularization c = c + lambda * i
        identity = torch.eye(C, device=self.device, dtype=torch.float64).unsqueeze(2)
        cov += identity * reg_factor

        manage_gpu_memory("After covariance calculation")
        if self.resource_monitor:
            self.resource_monitor.log(phase="after_covariance_calc")
        print(f"Covariance calculation complete. Mean shape: {mean.shape}, Cov shape: {cov.shape}")
        return mean, cov

    def _invert_covariance_batch(
        self,
        covariance_batch: torch.Tensor,
        identity_matrix: torch.Tensor,
        initial_jitter: float = 1e-6,
        max_attempts: int = 5
    ) -> torch.Tensor:
        """Invert a batch of covariance matrices using adaptive Cholesky with pseudo-inverse fallback."""
        inverses: List[torch.Tensor] = []
        for idx, covariance in enumerate(covariance_batch):
            covariance = 0.5 * (covariance + covariance.transpose(0, 1))
            jitter = initial_jitter
            success = False
            for attempt in range(max_attempts):
                try:
                    adjusted = covariance + jitter * identity_matrix
                    chol = torch.linalg.cholesky(adjusted)
                    inv = torch.cholesky_inverse(chol)
                    inverses.append(inv)
                    success = True
                    break
                except RuntimeError:
                    jitter *= 10.0
            if not success:
                print(f"Warning: Falling back to pseudo-inverse for covariance index {idx} within chunk.")
                inverses.append(torch.linalg.pinv(covariance))
        return torch.stack(inverses, dim=0)

    def evaluate_test_set(
        self,
        test_dataloader,
        test_embedding_vectors: torch.Tensor,
        mean: torch.Tensor,
        cov: torch.Tensor,
        sigma: float,
        save_dir: Optional[str] = None,
        return_anomaly_maps: bool = False,
        max_visuals: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the test set using pre-computed, DWT-processed embeddings.
        (Corrected image score calculation)
        """
        print("Starting test set evaluation...")
        if self.resource_monitor:
            self.resource_monitor.log(phase="test_evaluation_start")

        embedding_vectors = test_embedding_vectors.to(self.device, dtype=torch.float64)
        B, C, H, W_dim = embedding_vectors.size()
        P = H * W_dim
        print(f"Test embedding shape: B={B}, C={C}, H={H}, W={W_dim} (P={P})")
        embedding_vectors_reshaped = embedding_vectors.view(B, C, P)

        # --- compute inverse covariance matrices (precision matrices) ---
        print(f"Calculating {P} precision matrices (inverse covariance) for C={C}...")
        precision_matrices = []
        cov = cov.detach().to(self.device, dtype=torch.float64)
        stability_eye = torch.eye(C, device=self.device, dtype=torch.float64)
        precision_chunk_size = min(P, 512)
        for i in tqdm(range(0, P, precision_chunk_size), desc="Inverting Covariances"):
            chunk_end = min(i + precision_chunk_size, P)
            cov_chunk = cov[:, :, i:chunk_end].permute(2, 0, 1)
            precision_chunk = self._invert_covariance_batch(cov_chunk, stability_eye)
            precision_matrices.append(precision_chunk.permute(1, 2, 0))
            del cov_chunk, precision_chunk
            manage_gpu_memory(f"Precision matrix chunk {i // precision_chunk_size}")

        precision_tensor = torch.cat(precision_matrices, dim=2)
        print(f"Precision tensor shape: {precision_tensor.shape}")
        del precision_matrices, cov  # free cov memory

        # --- calculate mahalanobis distance ---
        print("Calculating Mahalanobis distances...")
        mean = mean.detach().to(self.device, dtype=torch.float64)
        precision_tensor = precision_tensor.detach().to(self.device, dtype=torch.float64)
        precision_tensor_reshaped = precision_tensor.permute(2, 0, 1)

        dist_list = calculate_mahalanobis_distance(embedding_vectors_reshaped, mean, precision_tensor_reshaped)
        print(f"Mahalanobis distance list shape: {dist_list.shape}")  # should be (p, b)
        dist_tensor = dist_list.t().reshape(B, H, W_dim).float()  # reshape back to (b, h_new, w_new)
        del precision_tensor, precision_tensor_reshaped, embedding_vectors_reshaped, mean, dist_list

        # --- collect original test images and ground truth masks ---
        print("Collecting original test images and ground truth masks...")
        test_imgs_orig = []
        gt_labels_list = []
        gt_mask_list = []
        with torch.no_grad():
            for x, y, mask in tqdm(test_dataloader, desc="Loading GT data"):
                test_imgs_orig.extend(x.cpu())
                gt_labels_list.extend(y.cpu().numpy())
                gt_mask_list.extend(mask.cpu())

        # --- upsample score map and apply gaussian smoothing ---
        if not test_imgs_orig:
            raise ValueError("No test images loaded.")
        orig_img_size = test_imgs_orig[0].shape[1:]
        print(f"Upsampling score map from {(H, W_dim)} to original image size {orig_img_size}...")
        # keep the raw score map after upsampling and smoothing
        score_map_raw_upsampled = F.interpolate(dist_tensor.unsqueeze(1), size=orig_img_size, mode='bilinear', align_corners=False).squeeze(1)
        del dist_tensor

        print(f"Applying Gaussian smoothing with sigma={sigma}...")
        # create a tensor to store smoothed maps
        score_map_smoothed = torch.zeros_like(score_map_raw_upsampled)
        for i in tqdm(range(score_map_raw_upsampled.shape[0]), desc="Smoothing Scores"):
            # perform smoothing on gpu for speed, then move to cpu
            score_map_smoothed[i] = gaussian_filter(score_map_raw_upsampled[i].to(self.device), sigma=sigma).cpu()
            if i % 10 == 0: manage_gpu_memory(f"Smoothing image {i}")
        del score_map_raw_upsampled

        # --- *** calculate image level score (before per-image normalization) *** ---
        print("Calculating image-level scores (using max pixel from smoothed map)...")
        img_scores = torch.amax(score_map_smoothed.view(B, -1), dim=1)

        # --- prepare pixel metrics tensor before any per-image normalization ---
        scores_for_pixel_metrics = score_map_smoothed.detach().cpu()

        # --- normalize scores per image (for visualization only) ---
        print("Normalizing scores (per image) for visualization...")
        scores_normalized_per_image = torch.zeros_like(scores_for_pixel_metrics)
        for i in tqdm(range(scores_for_pixel_metrics.shape[0]), desc="Normalizing Scores"):
            img_map = scores_for_pixel_metrics[i]
            min_val, max_val = img_map.min(), img_map.max()
            if max_val > min_val:
                scores_normalized_per_image[i] = (img_map - min_val) / (max_val - min_val)

        # --- prepare data for metrics ---
        img_scores_np = img_scores.cpu().numpy()  # use the scores derived *before* per-image normalization
        scores_for_pixel_metrics_np = scores_for_pixel_metrics.numpy()  # use globally scaled scores for pixel metrics
        gt_labels_np = np.array(gt_labels_list)
        try:
            gt_mask_tensor = torch.stack(gt_mask_list).squeeze(1).cpu().numpy()
        except Exception as e:
             print(f"Error stacking GT masks: {e}. Check mask dimensions.")
             gt_mask_tensor = np.zeros_like(scores_for_pixel_metrics_np)

        # --- optional diagnostics (can be kept or removed) ---
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
        # --- end diagnostics ---

        # --- calculate metrics ---
        print("Calculating evaluation metrics...")
        img_roc_auc, pixel_roc_auc = 0.0, 0.0
        if len(unique_labels) >= 2 and len(img_scores_np) > 0:
             try:
                 # use img_scores_np (derived before per-image norm) for image auc
                 img_roc_auc = calculate_image_level_roc_auc(img_scores_np, gt_labels_np)
                 # use scores_for_pixel_metrics_np (per-image norm) for pixel auc
                 pixel_roc_auc = calculate_pixel_level_roc_auc(scores_for_pixel_metrics_np, gt_mask_tensor)
             except Exception as e:
                 print(f"ERROR calculating metrics: {e}")
                 img_roc_auc, pixel_roc_auc = 0.0, 0.0
        else:
            print("Skipping AUC calculation due to insufficient unique labels or scores.")

        # --- save visualizations ---
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving visualizations to {save_dir}...")
            vis_threshold = 0.5  # threshold for binary mask in visualization
            num_saved = 0
            max_vis_save = max_visuals if (max_visuals is not None and max_visuals > 0) else 50
            # use the per-image normalized scores for visualization consistency
            scores_for_vis_np = scores_normalized_per_image.numpy()
            for i in tqdm(range(min(len(test_imgs_orig), max_vis_save)), desc="Saving Visuals"):
                # ... (rest of visualization saving loop using scores_for_vis_np[i] for the heatmap) ...
                 img = test_imgs_orig[i]
                 gt_label = gt_labels_np[i]
                 gt_mask = gt_mask_tensor[i]
                 score_map_vis = scores_for_vis_np[i]  # use normalized score map for consistent visualization
                 img_score_val = img_scores_np[i]  # use the actual image score value for title
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
            # return the per-image normalized scores if maps are requested
            'score_maps': scores_for_pixel_metrics_np if return_anomaly_maps else None,
            # return the actual image scores used for image auc
            'img_scores': img_scores_np if return_anomaly_maps else None
        }

    def evaluate_single_class(
        self,
        class_name: str,
        sigma: float = 4.0,
        cov_reg: float = 0.01,
        save_dir: Optional[str] = None,
        train_batch_size: int = 32,
        wavelet_type: str = 'haar',
        wavelet_level: int = 1,
        wavelet_kept_subbands: Optional[List[str]] = None,
        return_anomaly_maps: bool = False,
        embedding_cache: Optional[Dict[str, torch.Tensor]] = None,
        max_train_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        max_visuals: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate a single class using the DWT-before-concatenation workflow."""
        print(f"\n--- Evaluating Class: {class_name} ---")
        print(
            f"Parameters: Sigma={sigma}, CovReg={cov_reg}, Wavelet={wavelet_type}, "
            f"Level={wavelet_level}, Subbands={wavelet_kept_subbands}"
        )

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

        with MemoryUsageMonitor(f"evaluate_{class_name}"):
            print("Loading datasets...")
            try:
                train_loader, test_loader = create_efficient_dataloaders(
                    self.data_path,
                    class_name,
                    train_batch_size,
                    self.test_batch_size,
                    dataset_type=self.dataset_type,
                    max_train_samples=max_train_samples,
                    max_test_samples=max_test_samples
                )
            except Exception as e:
                print(f"ERROR loading data for class {class_name}: {e}")
                return {
                    'class': class_name,
                    'img_auc': 0.0,
                    'pixel_auc': 0.0,
                    'time': time.time() - class_start_time,
                    'error': str(e)
                }

            print("Getting training embeddings (DWT-before-concat)...")
            try:
                with MemoryUsageMonitor("feature_extraction_train"):
                    if isinstance(self.feature_extractor, CachedFeatureExtractor):
                        train_embeddings = self.feature_extractor.get_embedding_vectors(
                            dataloader=train_loader,
                            class_name=class_name,
                            wavelet_params=wavelet_params
                        )
                    else:
                        train_embeddings = self.feature_extractor.get_embedding_vectors(
                            dataloader=train_loader,
                            wavelet_params=wavelet_params
                        )
            except Exception as e:
                print(f"ERROR getting training embeddings for {class_name}: {e}")
                import traceback
                traceback.print_exc()
                current_time = time.time()
                return {
                    'class': class_name,
                    'img_auc': 0.0,
                    'pixel_auc': 0.0,
                    'time': current_time - class_start_time,
                    'error': f"Train embedding extraction failed: {e}"
                }

            if train_embeddings is None:
                print(f"ERROR: Training embedding extraction returned None for {class_name}.")
                current_time = time.time()
                return {
                    'class': class_name,
                    'img_auc': 0.0,
                    'pixel_auc': 0.0,
                    'time': current_time - class_start_time,
                    'error': "Train embedding extraction returned None"
                }

            train_embeddings = train_embeddings.to(self.device)
            manage_gpu_memory("After getting training embeddings")

            print(f"Calculating covariance matrix for {class_name}...")
            try:
                with MemoryUsageMonitor("covariance_calculation"):
                    mean, cov = self.calculate_covariance(train_embeddings, reg_factor=cov_reg)
                del train_embeddings
                manage_gpu_memory("After training covariance calculation")
            except Exception as e:
                print(f"ERROR calculating covariance for {class_name}: {e}")
                import traceback
                traceback.print_exc()
                current_time = time.time()
                return {
                    'class': class_name,
                    'img_auc': 0.0,
                    'pixel_auc': 0.0,
                    'time': current_time - class_start_time,
                    'error': f"Covariance calculation failed: {e}"
                }

            print("Getting test embeddings (DWT-before-concat)...")
            try:
                with MemoryUsageMonitor("feature_extraction_test"):
                    if isinstance(self.feature_extractor, CachedFeatureExtractor):
                        test_embeddings = self.feature_extractor.get_embedding_vectors(
                            dataloader=test_loader,
                            class_name=f"{class_name}_test",
                            wavelet_params=wavelet_params
                        )
                    else:
                        test_embeddings = self.feature_extractor.get_embedding_vectors(
                            dataloader=test_loader,
                            wavelet_params=wavelet_params
                        )
            except Exception as e:
                print(f"ERROR getting test embeddings for {class_name}: {e}")
                import traceback
                traceback.print_exc()
                del mean, cov
                current_time = time.time()
                return {
                    'class': class_name,
                    'img_auc': 0.0,
                    'pixel_auc': 0.0,
                    'time': current_time - class_start_time,
                    'error': f"Test embedding extraction failed: {e}"
                }

            if test_embeddings is None:
                print(f"ERROR: Test embedding extraction returned None for {class_name}.")
                del mean, cov
                current_time = time.time()
                return {
                    'class': class_name,
                    'img_auc': 0.0,
                    'pixel_auc': 0.0,
                    'time': current_time - class_start_time,
                    'error': "Test embedding extraction returned None"
                }

            test_embeddings = test_embeddings.to(self.device)
            manage_gpu_memory("After getting test embeddings")

            print(f"Evaluating test set for {class_name}...")
            try:
                with MemoryUsageMonitor("test_evaluation"):
                    results = self.evaluate_test_set(
                        test_dataloader=test_loader,
                        test_embedding_vectors=test_embeddings,
                        mean=mean,
                        cov=cov,
                        sigma=sigma,
                        save_dir=save_dir,
                        return_anomaly_maps=return_anomaly_maps,
                        max_visuals=max_visuals
                    )
                del test_embeddings, mean, cov
                manage_gpu_memory(f"After evaluating test set for {class_name}")
            except Exception as e:
                print(f"ERROR evaluating test set for {class_name}: {e}")
                import traceback
                traceback.print_exc()
                try:
                    del test_embeddings, mean, cov
                except NameError:
                    pass
                manage_gpu_memory(f"After error during test evaluation for {class_name}")
                current_time = time.time()
                return {
                    'class': class_name,
                    'img_auc': 0.0,
                    'pixel_auc': 0.0,
                    'time': current_time - class_start_time,
                    'error': f"Test set evaluation failed: {e}"
                }

        total_time = time.time() - class_start_time
        print(f"--- Finished Class: {class_name} in {total_time:.2f} seconds ---")
        img_auc_res = results.get('img_roc_auc', 0.0)
        pix_auc_res = results.get('pixel_roc_auc', 0.0)
        print(f"  Image AUC: {img_auc_res:.4f}, Pixel AUC: {pix_auc_res:.4f}")

        ret: Dict[str, Any] = {
            'class': class_name,
            'img_auc': img_auc_res,
            'pixel_auc': pix_auc_res,
            'time': total_time
        }
        if return_anomaly_maps:
            ret['score_maps'] = results.get('score_maps')
            ret['img_scores'] = results.get('img_scores')

        if self.resource_monitor:
            self.resource_monitor.log(phase=f"end_{class_name}_evaluation")

        return ret
