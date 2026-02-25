from __future__ import annotations

"""
Wavelet transform utilities for WE-PaDiM.
Provides wavelet-based feature fusion and compression helpers.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# optional dependency: pytorch-wavelets
try:
    from pytorch_wavelets import DWTForward, DWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    DWTForward = None  # type: ignore[assignment]
    DWTInverse = None  # type: ignore[assignment]
    print("Warning: pytorch_wavelets not found. Please install with: pip install pytorch-wavelets")

class AdaptiveWaveletFusion(nn.Module):
    """Simple multi-level wavelet fusion module."""

    def __init__(
        self,
        wave: str = "haar",
        levels: Optional[List[int]] = None,
        kept_subbands: Optional[List[str]] = None,
        learn_weights: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if not PYTORCH_WAVELETS_AVAILABLE:
            raise ImportError(
                "pytorch_wavelets is required for AdaptiveWaveletFusion. Install with: pip install pytorch-wavelets"
            )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.levels = levels or [1, 2]
        self.kept_subbands = kept_subbands or ["LL"]
        self.wave = wave

        self.wavelets = nn.ModuleList(
            [DWTForward(J=level, wave=wave, mode="zero").to(self.device) for level in self.levels]
        )

        self.learn_weights = learn_weights
        if learn_weights:
            self.level_weights = nn.Parameter(torch.ones(len(self.levels), device=self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fusion. Uses no gradients during eval mode."""
        if not self.training:
            with torch.no_grad():
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        level_features: List[torch.Tensor] = []

        for level_idx, level in enumerate(self.levels):
            yl, _ = self.wavelets[level_idx](x)
            if yl.shape[2:] != (height, width):
                yl = F.interpolate(yl, size=(height, width), mode="bilinear", align_corners=False)
            level_features.append(yl)

        if self.learn_weights:
            weights = F.softmax(self.level_weights, dim=0)
            combined = torch.zeros_like(level_features[0])
            for idx, feat in enumerate(level_features):
                combined += feat * weights[idx]
        else:
            combined = sum(level_features) / len(level_features)
        return combined

    def train_weights(
        self,
        train_dataloader,
        feature_extractor,
        num_epochs: int = 5,
        learning_rate: float = 0.01,
    ) -> Optional[Any]:
        if not self.learn_weights:
            print("AdaptiveWaveletFusion: learn_weights disabled; nothing to train.")
            return None

        if not hasattr(self, "level_weights"):
            self.level_weights = nn.Parameter(torch.ones(len(self.levels), device=self.device))

        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            print("AdaptiveWaveletFusion: no trainable parameters; skipping training.")
            return None

        self.train()
        optimizer = torch.optim.Adam([self.level_weights], lr=learning_rate)
        criterion = torch.nn.MSELoss()

        for _ in range(num_epochs):
            for x, _, _ in train_dataloader:
                x = x.to(self.device)
                with torch.no_grad():
                    reference = feature_extractor.get_embedding_vectors(x)
                fused = self(x)
                loss = criterion(fused, reference)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.eval()
        with torch.no_grad():
            return F.softmax(self.level_weights, dim=0).cpu().numpy()

class WaveletFeatureCompression(nn.Module):
    """Wavelet-based feature compression for PaDiM."""

    def __init__(
        self,
        wave: str = "haar",
        level: int = 1,
        kept_subbands: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if not PYTORCH_WAVELETS_AVAILABLE:
            raise ImportError(
                "pytorch_wavelets is required for wavelet transforms. Install with: pip install pytorch-wavelets"
            )

        self.wave = wave
        self.level = level
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if kept_subbands is None:
            self.kept_subbands = ["LL", "LH", "HL", "HH"] if level == 1 else None
        else:
            if level == 1:
                valid = {"LL", "LH", "HL", "HH"}
                invalid = [sb for sb in kept_subbands if sb not in valid]
                if invalid:
                    raise ValueError(f"Invalid subbands: {invalid}. Valid options: {sorted(valid)}")
            self.kept_subbands = kept_subbands

        self.dwt = DWTForward(J=level, wave=wave, mode="zero").to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        yl, yh = self.dwt(x)

        if self.level == 1:
            ll = yl
            lh, hl, hh = yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]
            components: List[torch.Tensor] = []
            subband_map = {"LL": ll, "LH": lh, "HL": hl, "HH": hh}
            for sb in self.kept_subbands:
                components.append(subband_map[sb])
            return components[0] if len(components) == 1 else torch.cat(components, dim=1)

        components = [yl]
        for level_idx in range(self.level):
            high_comps = yh[level_idx]
            if self.kept_subbands:
                if "LH" in self.kept_subbands:
                    components.append(high_comps[:, :, 0])
                if "HL" in self.kept_subbands:
                    components.append(high_comps[:, :, 1])
                if "HH" in self.kept_subbands:
                    components.append(high_comps[:, :, 2])
            else:
                for i in range(3):
                    components.append(high_comps[:, :, i])

        resized = [components[0]]
        target_h, target_w = components[0].shape[2:]
        for comp in components[1:]:
            if comp.shape[2:] != (target_h, target_w):
                comp = F.interpolate(comp, size=(target_h, target_w), mode="bilinear", align_corners=False)
            resized.append(comp)
        return torch.cat(resized, dim=1)

    @staticmethod
    def get_compression_ratio(
        input_shape: Tuple[int, int, int, int],
        level: int = 1,
        kept_subbands: Optional[List[str]] = None,
    ) -> float:
        _, channels, height, width = input_shape
        original_size = channels * height * width
        if kept_subbands is None:
            return 1.0

        if level == 1:
            subband_size = channels * (height // 2) * (width // 2)
            compressed = len(kept_subbands) * subband_size
            return original_size / compressed

        compressed = channels * (height // (2 ** level)) * (width // (2 ** level))
        if kept_subbands:
            for l in range(level):
                level_h = height // (2 ** (l + 1))
                level_w = width // (2 ** (l + 1))
                num = sum(1 for sb in kept_subbands if sb != "LL" and f"L{l+1}" in sb)
                compressed += num * channels * level_h * level_w
        return original_size / compressed

class OptimizedWaveletCompression:
    """Memory-aware wavelet compression with configurable CUDA fallback."""

    _global_force_cpu: bool = False

    def __init__(
        self,
        wave: str = "haar",
        level: int = 1,
        kept_subbands: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        allow_cpu_fallback: bool = False,
    ) -> None:
        self.wave = wave
        self.level = level
        self.kept_subbands = kept_subbands or ["LL"]
        self.allow_cpu_fallback = allow_cpu_fallback
        self.preferred_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not PYTORCH_WAVELETS_AVAILABLE:
            raise ImportError(
                "pytorch_wavelets is required for OptimizedWaveletCompression. Install with: pip install pytorch-wavelets"
            )

        if OptimizedWaveletCompression._global_force_cpu or self.preferred_device.type != "cuda":
            self.device = torch.device("cpu")
            self.force_cpu = True
        else:
            self.device = self.preferred_device
            self.force_cpu = False

        self.dwt = DWTForward(J=level, wave=wave, mode="zero").to(self.device)
        self.cpu_dwt: Optional[DWTForward] = self.dwt if self.device.type == "cpu" else None
        self.current_device = self.device

        self.process_LL = "LL" in self.kept_subbands
        subband_map = {"LH": 0, "HL": 1, "HH": 2}
        self.subband_indices: List[int] = [subband_map[sb] for sb in self.kept_subbands if sb in subband_map]

    def _ensure_cpu_dwt(self) -> DWTForward:
        if self.cpu_dwt is None:
            self.cpu_dwt = DWTForward(J=self.level, wave=self.wave, mode="zero").to(torch.device("cpu"))
        return self.cpu_dwt

    @staticmethod
    def _safe_empty_cache() -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _apply_dwt_gpu_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.current_device.type != "cuda":
            raise RuntimeError("GPU DWT requested while running on CPU.")
        return self.dwt(batch.contiguous())

    def _apply_dwt_gpu_sequential(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if batch.size(0) == 1:
            return self._apply_dwt_gpu_batch(batch)

        yl_parts: List[torch.Tensor] = []
        yh_parts: List[List[torch.Tensor]] = [[] for _ in range(self.level)]
        for single in batch.split(1, dim=0):
            single_yl, single_yh = self._apply_dwt_gpu_batch(single)
            yl_parts.append(single_yl)
            for level_idx in range(self.level):
                yh_parts[level_idx].append(single_yh[level_idx])
            self._safe_empty_cache()

        yl = torch.cat(yl_parts, dim=0)
        yh = [torch.cat(parts, dim=0) for parts in yh_parts] if self.level > 0 else []
        return yl, yh

    def _apply_dwt_cpu(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        cpu_dwt = self._ensure_cpu_dwt()
        batch_cpu = batch.detach().cpu().contiguous()
        return cpu_dwt(batch_cpu)

    def __call__(self, x: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        total_batch_size = x.shape[0]
        target_device = x.device
        use_gpu = self.current_device.type == "cuda" and not self.force_cpu

        compressed_batches: List[torch.Tensor] = []
        for start_idx in range(0, total_batch_size, batch_size):
            end_idx = min(start_idx + batch_size, total_batch_size)
            batch = x[start_idx:end_idx]

            with torch.no_grad():
                try:
                    if use_gpu:
                        yl, yh = self._apply_dwt_gpu_batch(batch)
                    else:
                        yl, yh = self._apply_dwt_cpu(batch)
                except RuntimeError as err:
                    cuda_error = "CUDA error" in str(err)
                    if use_gpu and cuda_error:
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        try:
                            yl, yh = self._apply_dwt_gpu_sequential(batch)
                        except RuntimeError as seq_err:
                            if self.allow_cpu_fallback:
                                print("Warning: Wavelet compression failed on GPU; falling back to CPU for remaining batches.")
                                OptimizedWaveletCompression._global_force_cpu = True
                                self.force_cpu = True
                                yl, yh = self._apply_dwt_cpu(batch)
                                cpu_dwt = self._ensure_cpu_dwt()
                                self.dwt = cpu_dwt
                                self.current_device = torch.device("cpu")
                                use_gpu = False
                            else:
                                raise RuntimeError(
                                    "CUDA wavelet transform failed even after retry with single-image batches. "
                                    "Either resolve the CUDA issue or instantiate OptimizedWaveletCompression "
                                    "with allow_cpu_fallback=True to enable CPU retries."
                                ) from seq_err
                    else:
                        raise

            batch_components: List[torch.Tensor] = []
            if self.process_LL:
                batch_components.append(yl)

            if self.subband_indices and self.level > 0:
                for level_idx in range(self.level):
                    level_yh = yh[level_idx]
                    for idx in self.subband_indices:
                        component = level_yh[:, :, idx]
                        if component.shape[2:] != yl.shape[2:]:
                            component = F.interpolate(
                                component,
                                size=yl.shape[2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        batch_components.append(component)

            compressed_batch = batch_components[0] if len(batch_components) == 1 else torch.cat(batch_components, dim=1)
            compressed_batches.append(compressed_batch.to(target_device))

            del batch, yl, yh, batch_components
            self._safe_empty_cache()

        result = torch.cat(compressed_batches, dim=0)
        del compressed_batches
        self._safe_empty_cache()
        return result

def apply_wavelet_compression(
    features: torch.Tensor,
    wavelet: str = "haar",
    level: int = 1,
    kept_subbands: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    compressor = WaveletFeatureCompression(
        wave=wavelet,
        level=level,
        kept_subbands=kept_subbands,
        device=device,
    )
    return compressor(features)
