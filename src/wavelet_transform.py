"""
Wavelet transform module for PaDiM with Wavelet-based dimensionality reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from typing import Tuple, List, Dict, Any, Optional, Union

# Try to import PyTorch Wavelets
try:
    from pytorch_wavelets import DWTForward, DWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    print("Warning: pytorch_wavelets not found. Please install with: pip install pytorch-wavelets")

class AdaptiveWaveletFusion(nn.Module):
    """
    Simplified Adaptive Wavelet Fusion that avoids dimension mismatches.
    """
    def __init__(
        self,
        wave: str = 'haar',
        levels: list = [1, 2],
        kept_subbands: list = ['LL'],
        learn_weights: bool = False,
        device: torch.device = None
    ):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.levels = levels
        self.kept_subbands = kept_subbands
        self.wave = wave
        
        # Create wavelet transformers for each level
        self.wavelets = nn.ModuleList([
            DWTForward(J=level, wave=wave, mode='zero').to(self.device)
            for level in levels
        ])
        
        # Flag for using weighted fusion
        self.learn_weights = learn_weights
        
        # Initialize weights if using weighted fusion
        if learn_weights:
            self.level_weights = nn.Parameter(torch.ones(len(levels), device=self.device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-level wavelet fusion.
        
        During evaluation, we don't need to track gradients, so we use no_grad
        to prevent gradient-related errors.
        """
        # Check if we're in evaluation mode
        if not self.training:
            with torch.no_grad():
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of the fusion logic."""
        batch_size, channels, height, width = x.shape
        
        # Process each level and store the resulting feature maps
        level_features = []
        
        for level_idx, level in enumerate(self.levels):
            # Apply wavelet transform for this level
            yl, yh = self.wavelets[level_idx](x)
            
            # We'll use only the LL component for simplicity
            level_ll = yl
            
            # Resize LL component to match original dimensions if needed
            if level_ll.shape[2:] != (height, width):
                level_ll = F.interpolate(
                    level_ll, 
                    size=(height, width),
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Store this level's features
            level_features.append(level_ll)
        
        # Combine features from all levels (weighted or simple average)
        if self.learn_weights:
            # Apply softmax to weights for normalization
            weights = F.softmax(self.level_weights, dim=0)
            
            # Weighted sum of level features
            combined_features = torch.zeros_like(level_features[0])
            for i, level_feat in enumerate(level_features):
                combined_features += level_feat * weights[i]
        else:
            # Simple averaging of level features
            combined_features = sum(level_features) / len(level_features)
        
        return combined_features
    def train_weights(self, train_dataloader, feature_extractor, num_epochs=5, learning_rate=0.01):
        """Train the fusion weights to optimize feature representation."""
        # Initialize level_weights if using weighted fusion
        if self.learn_weights and not hasattr(self, 'level_weights'):
            self.level_weights = nn.Parameter(torch.ones(len(self.levels), device=self.device))
            print(f"Initialized level weights: {self.level_weights}")
        
        # Verify parameters exist
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            print("No trainable parameters found in AdaptiveWaveletFusion, skipping training")
            return None
        self.train()
        
        optimizer = torch.optim.Adam([self.level_weights], lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for x, _, _ in train_dataloader:
                x = x.to(self.device)
                with torch.no_grad():
                    reference_features = feature_extractor.get_embedding_vectors(x)
                
                fused_features = self(x)
                loss = criterion(fused_features, reference_features)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        self.eval()
        with torch.no_grad():
            return F.softmax(self.level_weights, dim=0).cpu().numpy()

class WaveletFeatureCompression(nn.Module):
    """
    Wavelet-based feature compression for PaDiM.
    
    This module applies wavelet transforms to feature maps for dimensionality reduction.
    It can keep selected subbands and levels to preserve important information while
    reducing the feature dimension.
    """
    
    def __init__(
        self,
        wave: str = 'haar',
        level: int = 1,
        kept_subbands: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the wavelet feature compression module.
        
        Args:
            wave: Wavelet to use (default: 'haar'). Options include 'haar', 'db1', 'db2', etc.
            level: Number of decomposition levels (default: 1)
            kept_subbands: List of subbands to keep. If None, keeps all. 
                           Options: ['LL', 'LH', 'HL', 'HH'] for level=1
            device: Device to use (CPU or CUDA)
        """
        super().__init__()
        
        if not PYTORCH_WAVELETS_AVAILABLE:
            raise ImportError("pytorch_wavelets is required for wavelet transforms. "
                             "Please install with: pip install pytorch-wavelets")
        
        self.wave = wave
        self.level = level
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default kept subbands if not specified
        if kept_subbands is None:
            if level == 1:
                # Keep all subbands by default
                self.kept_subbands = ['LL', 'LH', 'HL', 'HH']
            else:
                # For multi-level, keep all by default
                # This gets more complex, so we'll handle it in the forward pass
                self.kept_subbands = None
        else:
            if level == 1:
                # Validate subbands for single level
                valid_subbands = ['LL', 'LH', 'HL', 'HH']
                for sb in kept_subbands:
                    if sb not in valid_subbands:
                        raise ValueError(f"Invalid subband: {sb}. Must be one of {valid_subbands}")
                self.kept_subbands = kept_subbands
            else:
                # For multi-level, more complex validation needed
                # For simplicity, we'll accept any list of subbands and validate in forward
                self.kept_subbands = kept_subbands
        
        # Initialize DWT
        self.dwt = DWTForward(J=level, wave=wave, mode='zero').to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet transform to feature maps.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Compressed tensor with reduced dimensions
        """
        batch_size, channels, height, width = x.shape
        
        # Apply DWT - returns low-frequency and list of high-frequency components
        # yl: (B, C, H/2^level, W/2^level)
        # yh: list of level tensors, each (B, C, 3, H/2^l, W/2^l) where 3 = LH, HL, HH
        yl, yh = self.dwt(x)
        
        # Single level handling (simpler case)
        if self.level == 1:
            # Extract components
            ll = yl  # LL component (low-low)
            lh, hl, hh = yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]  # High frequency components
            
            # Keep only selected subbands
            components = []
            subband_map = {'LL': ll, 'LH': lh, 'HL': hl, 'HH': hh}
            
            for sb in self.kept_subbands:
                components.append(subband_map[sb])
            
            # Combine kept components
            if len(components) == 1:
                # Single component case
                return components[0]
            else:
                # Multiple components - stack or concatenate based on what's kept
                # Here we concatenate along channel dimension
                return torch.cat(components, dim=1)
        
        # Multi-level handling (more complex)
        else:
            # For multi-level, we need more complex handling
            # This is a simplified implementation - adjust as needed
            components = [yl]  # Start with the lowest frequency component
            
            # Add high frequency components from each level
            for level_idx in range(self.level):
                high_comps = yh[level_idx]  # (B, C, 3, H, W)
                
                # If we have specific subbands to keep, filter them
                if self.kept_subbands:
                    # Simple approach: just check if we want LH, HL, or HH 
                    # and append the corresponding component
                    if 'LH' in self.kept_subbands:
                        components.append(high_comps[:, :, 0])
                    if 'HL' in self.kept_subbands:
                        components.append(high_comps[:, :, 1])
                    if 'HH' in self.kept_subbands:
                        components.append(high_comps[:, :, 2])
                else:
                    # Keep all components
                    for i in range(3):  # LH, HL, HH
                        components.append(high_comps[:, :, i])
            
            # Concatenate all kept components along channel dimension
            # Note: components may have different spatial dimensions
            # For simplicity, resize all to match the lowest frequency component
            resized_components = [components[0]]  # LL is already the right size
            
            for comp in components[1:]:
                # Resize to match LL component spatial dimensions
                target_h, target_w = components[0].shape[2:]
                comp_h, comp_w = comp.shape[2:]
                
                if comp_h != target_h or comp_w != target_w:
                    # Use interpolate to resize - adjust mode as needed
                    resized = torch.nn.functional.interpolate(
                        comp, 
                        size=(target_h, target_w),
                        mode='bilinear', 
                        align_corners=False
                    )
                    resized_components.append(resized)
                else:
                    resized_components.append(comp)
            
            # Concatenate along channel dimension
            return torch.cat(resized_components, dim=1)
    
    @staticmethod
    def get_compression_ratio(
        input_shape: Tuple[int, int, int, int],
        level: int = 1,
        kept_subbands: Optional[List[str]] = None
    ) -> float:
        """
        Calculate the compression ratio achieved by the wavelet transform.
        
        Args:
            input_shape: Input tensor shape (B, C, H, W)
            level: Number of decomposition levels
            kept_subbands: List of subbands to keep
            
        Returns:
            Compression ratio (original size / compressed size)
        """
        _, channels, height, width = input_shape
        original_size = channels * height * width
        
        # Calculate size after transform
        if kept_subbands is None:
            # If keeping all subbands, there's no compression
            return 1.0
        
        # For level 1
        if level == 1:
            # Size of each subband
            subband_size = channels * (height // 2) * (width // 2)
            
            # Count how many subbands we're keeping
            num_kept = len(kept_subbands)
            compressed_size = num_kept * subband_size
            
            return original_size / compressed_size
        
        # For multi-level, calculation is more complex
        # This is a simplified estimate
        compressed_size = 0
        
        # Size of LL band
        ll_h, ll_w = height // (2 ** level), width // (2 ** level)
        compressed_size += channels * ll_h * ll_w
        
        # Size of high frequency bands at each level
        if kept_subbands:
            for l in range(level):
                level_h, level_w = height // (2 ** (l+1)), width // (2 ** (l+1))
                num_kept_this_level = sum(1 for sb in kept_subbands if sb != 'LL' and f'L{l+1}' in sb)
                compressed_size += num_kept_this_level * channels * level_h * level_w
        
        return original_size / compressed_size

class OptimizedWaveletCompression:
    """
    Memory-efficient wavelet compression for feature maps.
    
    This implementation focuses on computational and memory efficiency
    when applying wavelet transforms for dimensionality reduction.
    """
    def __init__(
        self,
        wave: str = 'haar',
        level: int = 1,
        kept_subbands: List[str] = ['LL'],
        device: Optional[torch.device] = None
    ):
        self.wave = wave
        self.level = level
        self.kept_subbands = kept_subbands
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create wavelet transformer
        self.dwt = DWTForward(J=level, wave=wave, mode='zero').to(self.device)
        
        # Pre-compute subband indices and masks for efficiency
        self.process_LL = 'LL' in kept_subbands
        self.subband_indices = []
        
        if not self.process_LL and level > 0:
            raise ValueError("When process_LL is False, level must be at least 1")
            
        # Map subband names to indices for high frequency components
        subband_map = {'LH': 0, 'HL': 1, 'HH': 2}
        for sb in kept_subbands:
            if sb != 'LL' and sb in subband_map:
                self.subband_indices.append(subband_map[sb])
    
    def __call__(self, x: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        Apply wavelet compression to input tensor with batched processing.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            batch_size: Size of batches to process to limit memory usage
            
        Returns:
            Compressed tensor
        """
        total_batch_size, channels, height, width = x.shape
        device = x.device
        
        # Process in smaller batches to limit memory usage
        compressed_batches = []
        
        for start_idx in range(0, total_batch_size, batch_size):
            end_idx = min(start_idx + batch_size, total_batch_size)
            batch = x[start_idx:end_idx]
            
            # Apply wavelet transform
            with torch.no_grad():  # Ensure no gradients for memory efficiency
                yl, yh = self.dwt(batch)
            
            # Extract LL component (low frequency)
            batch_components = []
            if self.process_LL:
                batch_components.append(yl)
            
            # Extract selected high frequency components
            if self.subband_indices and self.level > 0:
                for level_idx in range(self.level):
                    # High frequency components at this level
                    level_yh = yh[level_idx]  # Shape: (batch, C, 3, H_j, W_j)
                    
                    # Extract requested subbands
                    for idx in self.subband_indices:
                        component = level_yh[:, :, idx]
                        
                        # Resize to match LL component size
                        if component.shape[2:] != yl.shape[2:]:
                            component = F.interpolate(
                                component,
                                size=yl.shape[2:],
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        batch_components.append(component)
            
            # Concatenate all components for this batch
            if len(batch_components) == 1:
                compressed_batch = batch_components[0]
            else:
                compressed_batch = torch.cat(batch_components, dim=1)
            
            compressed_batches.append(compressed_batch)
            
            # Clear variables to free memory
            del batch, yl, yh, batch_components
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all processed batches
        result = torch.cat(compressed_batches, dim=0)
        
        # Clear variables
        del compressed_batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result

# Example usage:
def apply_wavelet_compression(
    features: torch.Tensor,
    wavelet: str = 'haar',
    level: int = 1,
    kept_subbands: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Apply wavelet compression to feature maps.
    
    Args:
        features: Feature tensor of shape (B, C, H, W)
        wavelet: Wavelet to use
        level: Number of decomposition levels
        kept_subbands: List of subbands to keep
        device: Device to use
        
    Returns:
        Compressed feature tensor
    """
    compressor = WaveletFeatureCompression(
        wave=wavelet,
        level=level,
        kept_subbands=kept_subbands,
        device=device
    )
    
    return compressor(features)
