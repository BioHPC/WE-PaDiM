# models.py
"""
Models module for PaDiM with wavelets
Provides model setup, feature extraction, and related utilities.
(Revised for DWT-before-concatenation)
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional, Union, Any

# for resource monitoring
from utils import ResourceMonitor, log_gpu_memory, manage_gpu_memory

# import necessary components from other files
from wavelet_transform import WaveletFeatureCompression, OptimizedWaveletCompression

class FeatureExtractor:
    """Feature extractor class to handle different model architectures."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        """
        Initialize feature extractor with specified model.

        Args:
            model_name: Name of the model (e.g., 'resnet18', 'efficientnet-b0')
            device: Device to use (CPU or CUDA)
            resource_monitor: Optional resource monitor for tracking
        """
        self.model_name = model_name
        self.device = device
        self.resource_monitor = resource_monitor

        # set up model and target layers
        # _setup_model now also returns the names of the layers used
        self.model, self.feature_dims, self.target_layers = self._setup_model()
        self.model.to(self.device)
        self.model.eval()

        # note: self.total_dim is less directly relevant now as the final dimension
        # depends on the number of kept wavelet subbands per layer.
        # self.total_dim = sum(self.feature_dims) # original calculation

        print(f"Initialized FeatureExtractor for {model_name} with target layers: {self.target_layers}")

    def _setup_model(self) -> Tuple[torch.nn.Module, List[int], List[str]]:
        """
        Set up the model and return feature dimensions and target layer names.

        Returns:
            Tuple of (model, feature_dimensions, target_layer_names)
        """
        if self.resource_monitor:
            self.resource_monitor.log(phase=f"before_model_setup_{self.model_name}")

        model = None
        feature_dims = []
        target_layers = []

        # --- handle resnet models ---
        if self.model_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            target_layers = ['layer1', 'layer2', 'layer3']
            feature_dims = [64, 128, 256]

        elif self.model_name == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            target_layers = ['layer1', 'layer2', 'layer3']
            feature_dims = [64, 128, 256]

        elif self.model_name == "wide_resnet50_2":
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            target_layers = ['layer1', 'layer2', 'layer3']
            feature_dims = [256, 512, 1024]

        # --- handle efficientnet models ---
        elif self.model_name.startswith("efficientnet-b"):
            version = self.model_name.split("-")[1]

            # define target features block indices for efficientnets
            effnet_target_indices = [3, 4, 5, 6]
            # store names like 'features.3' - used for hooking and accessing output dict
            target_layers = [f'features.{i}' for i in effnet_target_indices]

            if version == "b0":
                from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                feature_dims = [40, 80, 112, 192]

            elif version == "b1":
                from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
                model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
                feature_dims = [40, 80, 112, 192]

            elif version == "b2":
                from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
                model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
                feature_dims = [48, 88, 120, 208]

            elif version == "b3":
                from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
                model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
                feature_dims = [48, 96, 136, 232]

            elif version == "b4":
                 from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
                 model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
                 feature_dims = [56, 112, 160, 272]

            elif version == "b5":
                from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
                model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
                feature_dims = [64, 128, 176, 304]

            elif version == "b6":
                from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
                model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
                feature_dims = [72, 144, 200, 344]

            else:
                raise ValueError(f"Unsupported EfficientNet version: {version}")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if self.resource_monitor:
            self.resource_monitor.log(phase=f"after_model_setup_{self.model_name}")

        # return the initialized model, the dims of the target layers, and the names of the target layers
        return model, feature_dims, target_layers

    def extract_features(self, dataloader) -> Dict[str, torch.Tensor]:
        """
        Extract features from the specified target layers of the model.

        Args:
            dataloader: DataLoader containing the images

        Returns:
            Dictionary where keys are target layer names (e.g., 'layer1', 'features.3')
            and values are the corresponding feature tensors (B, C, H, W) for the whole dataset.
        """
        if self.resource_monitor:
            self.resource_monitor.log(phase="before_feature_extraction")

        # use self.target_layers determined during setup
        # initialize lists to collect features for each layer across batches
        feature_outputs_batched = OrderedDict([(layer_name, []) for layer_name in self.target_layers])

        hooks = []
        # use a temporary dictionary per forward pass to store outputs captured by hooks
        temp_outputs_per_pass = {}

        # hook function factory
        def hook_fn(layer_identifier):  # use a unique identifier (e.g., layer name)
            def hook(module, input, output):
                # store output keyed by its identifier
                temp_outputs_per_pass[layer_identifier] = output.detach().clone()  # clone to prevent modification issues
            return hook

        # --- register hooks ---
        # use the target_layers list defined in _setup_model
        for layer_name in self.target_layers:
            module_to_hook = None
            if self.model_name.startswith("resnet"):
                # for resnets, layer names directly correspond to attributes
                if hasattr(self.model, layer_name):
                     module_to_hook = getattr(self.model, layer_name)
                else:
                     print(f"Warning: Could not find attribute '{layer_name}' in ResNet model.")
            elif self.model_name.startswith("efficientnet"):
                # for efficientnets, parse names like 'features.3'
                try:
                    parts = layer_name.split('.')
                    module = self.model
                    for part in parts:
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    module_to_hook = module
                except (AttributeError, IndexError, ValueError) as e:
                    print(f"Warning: Could not find module for EfficientNet layer '{layer_name}': {e}")

            if module_to_hook:
                hooks.append(module_to_hook.register_forward_hook(hook_fn(layer_name)))
            else:
                 print(f"Error: Failed to register hook for layer '{layer_name}' - layer not found or accessible.")

        if not hooks:
             raise RuntimeError("No hooks were successfully registered. Check target layer names and model structure.")

        print(f"Registered {len(hooks)} hooks for layers: {self.target_layers}")

        try:
            # --- extract features batch by batch ---
            for batch_idx, (x, _, _) in enumerate(tqdm(dataloader, desc=f"Extracting features ({self.model_name})")):
                x = x.to(self.device)
                temp_outputs_per_pass.clear()  # clear outputs from the previous batch

                with torch.no_grad():
                    _ = self.model(x)  # perform forward pass

                # append the captured outputs to the correct list in feature_outputs_batched
                for layer_name in self.target_layers:
                    if layer_name in temp_outputs_per_pass:
                        feature_outputs_batched[layer_name].append(temp_outputs_per_pass[layer_name])
                    else:
                        # this indicates a hook might have failed or wasn't registered correctly
                        print(f"Warning: No output captured for layer '{layer_name}' in batch {batch_idx}. Check hook registration.")

                if batch_idx > 0 and batch_idx % 10 == 0:  # manage memory periodically
                    manage_gpu_memory(f"Feature extraction batch {batch_idx}")

        finally:
            # --- always remove hooks ---
            for h in hooks:
                h.remove()

        # --- concatenate features across batches for each layer ---
        final_feature_outputs = OrderedDict()
        for layer_name, batch_features in feature_outputs_batched.items():
            if batch_features:  # ensure list is not empty
                 try:
                     final_feature_outputs[layer_name] = torch.cat(batch_features, 0)
                 except Exception as e:
                     print(f"Error concatenating features for layer {layer_name}: {e}")
                     # optionally, print shapes of tensors in the list for debugging
                     # for i, t in enumerate(batch_features): print(f"  batch {i} shape: {t.shape}")
                     final_feature_outputs[layer_name] = None  # indicate failure
            else:
                 print(f"Warning: No features collected for layer {layer_name}.")
                 final_feature_outputs[layer_name] = None

        del feature_outputs_batched, temp_outputs_per_pass  # free memory
        manage_gpu_memory("After feature extraction loop & concatenation")

        if self.resource_monitor:
            self.resource_monitor.log(phase="after_feature_extraction")

        # return dictionary keyed by layer name with tensors for the full dataset
        return final_feature_outputs

    def get_wavelet_features_before_concat(self, dataloader, wavelet_params: dict) -> torch.Tensor:
        """
        Extracts features, applies DWT per layer, selects subbands, aligns, and concatenates channel-wise.

        Args:
            dataloader: DataLoader for input images.
            wavelet_params: Dictionary containing 'wavelet_type', 'wavelet_level',
                            and 'wavelet_kept_subbands'.

        Returns:
            Tensor of concatenated DWT coefficients from selected layers and subbands.
            Shape: (B, C_total_dwt, H_out, W_out)
        """
        print(f"Starting DWT-before-concat feature processing for model {self.model_name}...")
        # 1. extract raw features from all target layers
        # the keys of this dict are the layer names (e.g., 'layer1', 'features.3')
        raw_feature_outputs = self.extract_features(dataloader)

        processed_layer_features = []
        layer_spatial_sizes: List[Tuple[int, int]] = []

        # 2. initialize dwt compressor (once) using provided params
        # use optimizedwaveletcompression for potentially better memory handling
        try:
            compressor = OptimizedWaveletCompression(
                wave=wavelet_params['wavelet_type'],
                level=wavelet_params['wavelet_level'],
                kept_subbands=wavelet_params['wavelet_kept_subbands'],
                device=self.device
            )
        except Exception as e:
             print(f"ERROR initializing OptimizedWaveletCompression: {e}")
             raise

        # 3. process features layer by layer using self.target_layers
        print(f"Processing features from layers: {self.target_layers}")
        for layer_name in self.target_layers:
            layer_features_raw = raw_feature_outputs.get(layer_name)

            # skip if features for this layer weren't extracted successfully
            if layer_features_raw is None:
                 print(f"Warning: Skipping layer '{layer_name}' due to missing raw features.")
                 continue

            print(f"  Processing layer {layer_name}, Raw shape: {layer_features_raw.shape}")

            # apply dwt and select subbands for *this single layer*
            with torch.no_grad():  # dwt application doesn't require gradients
                try:
                    # pass the raw features for this layer to the compressor
                    # the compressor returns the concatenated selected subbands for this layer
                    layer_dwt_coeffs = compressor(layer_features_raw)
                    print(f"    DWT output shape for {layer_name}: {layer_dwt_coeffs.shape}")
                except Exception as e:
                     print(f"ERROR applying DWT to layer {layer_name}: {e}")
                     # decide how to handle: skip layer or raise error? skipping for now.
                     continue  # skip to the next layer

            processed_layer_features.append((layer_name, layer_dwt_coeffs))
            layer_spatial_sizes.append((layer_dwt_coeffs.shape[2], layer_dwt_coeffs.shape[3]))

            # optional: aggressive memory clearing inside the loop
            del layer_features_raw, layer_dwt_coeffs
            if layer_name in raw_feature_outputs: del raw_feature_outputs[layer_name]
            manage_gpu_memory(f"After DWT/Align for {layer_name}")
        # 4. determine a shared target spatial size (use the smallest h/w to avoid upsampling blur)
        if not processed_layer_features:
             raise ValueError("No features were successfully processed and collected. Check layer names, extraction, and DWT steps.")

        target_height = min(size[0] for size in layer_spatial_sizes)
        target_width = min(size[1] for size in layer_spatial_sizes)
        target_spatial_size = (target_height, target_width)
        print(f"Aligning all layers to minimal spatial size: {target_spatial_size}")
        del layer_spatial_sizes

        aligned_features: List[torch.Tensor] = []
        for layer_name, feature_tensor in processed_layer_features:
            current_size = (feature_tensor.shape[2], feature_tensor.shape[3])
            if current_size != target_spatial_size:
                print(f"    Downsampling layer {layer_name} from {current_size} to {target_spatial_size}")
                try:
                    feature_tensor = F.interpolate(
                        feature_tensor,
                        size=target_spatial_size,
                        mode='bilinear',
                        align_corners=False
                    )
                except Exception as e:
                    print(f"ERROR interpolating layer {layer_name}: {e}. Skipping layer.")
                    continue
            aligned_features.append(feature_tensor)

        if not aligned_features:
            raise ValueError("No features remained after spatial alignment. Check DWT outputs.")

        # 5. concatenate the processed dwt coefficients along the channel dimension
        print(f"Concatenating DWT coefficients from {len(aligned_features)} layers channel-wise.")
        try:
            final_embedding_vector = torch.cat(aligned_features, dim=1)
            print(f"Final concatenated embedding shape: {final_embedding_vector.shape}")
        except Exception as e:
            print(f"Error during final channel concatenation: {e}")
            raise

        # optional: clear intermediate list and compressor object
        del aligned_features, processed_layer_features, raw_feature_outputs, compressor
        manage_gpu_memory("After final feature concatenation")

        return final_embedding_vector

    # --- keep the original get_embedding_vectors method name, but point it to the new logic ---
    # this maintains compatibility with existing code that calls get_embedding_vectors,
    # assuming the dwt-before-concat is the primary/only way features should now be obtained.
    # it now requires wavelet_params.
    def get_embedding_vectors(self, dataloader, wavelet_params: Optional[dict] = None) -> torch.Tensor:
        """
        Main method to get final embedding vectors.
        Implements DWT-before-concatenation workflow.

        Args:
            dataloader: DataLoader for input images.
            wavelet_params: Dictionary containing 'wavelet_type', 'wavelet_level',
                            and 'wavlet_kept_subbands'. Required for this method.

        Returns:
            Tensor of concatenated DWT coefficients from selected layers and subbands.
        """
        if wavelet_params is None:
            raise ValueError("`wavelet_params` dictionary is required for get_embedding_vectors (DWT-before-concat workflow).")

        # call the actual implementation
        return self.get_wavelet_features_before_concat(dataloader, wavelet_params)

# --- cached feature extractor ---
# wraps the featureextractor to add caching capabilities
class CachedFeatureExtractor:
    """Feature extractor with caching to avoid redundant computation."""
    def __init__(
        self,
        base_extractor: FeatureExtractor,  # expects an instance of the modified featureextractor
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        self.base_extractor = base_extractor
        self.cache_dir = cache_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache = {}  # in-memory cache

        # create cache directory if specified and doesn't exist
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using disk cache directory: {cache_dir}")
        else:
            print("Disk cache disabled.")

    def _generate_cache_key(self, class_name: Optional[str], wavelet_params: Optional[dict]) -> str:
        """Generates a unique cache key based on class and wavelet parameters."""
        key_parts = [self.base_extractor.model_name]
        if class_name:
            key_parts.append(class_name)
        if wavelet_params:
             # create a stable string representation of params
             params_str = f"wave_{wavelet_params.get('wavelet_type', 'na')}_lvl_{wavelet_params.get('wavelet_level', 'na')}_subs_{'_'.join(sorted(wavelet_params.get('wavelet_kept_subbands', [])))}"
             key_parts.append(params_str)
        # sanitize for filesystem compatibility
        raw_key = "_".join(key_parts)
        safe_key = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in raw_key)
        return safe_key

    def get_cache_path(self, cache_key: str) -> Optional[str]:
        """Get path for cached features using a cache key."""
        if not self.cache_dir or not cache_key:
            return None
        return os.path.join(self.cache_dir, f"{cache_key}_features.pt")

    # renamed method to match the base class, now requires wavelet_params
    def get_embedding_vectors(self, dataloader, class_name: Optional[str] = None, wavelet_params: Optional[dict] = None) -> torch.Tensor:
        """Get embedding vectors with caching, using the DWT-before-concat workflow."""
        if wavelet_params is None:
             # or fetch default params from config? forcing it seems safer.
            raise ValueError("`wavelet_params` dictionary is required for CachedFeatureExtractor.get_embedding_vectors.")

        # generate cache key based on class and wavelet params
        cache_key = self._generate_cache_key(class_name, wavelet_params)

        # 1. try to load from memory cache
        if cache_key in self.cache:
            print(f"Using in-memory cached embeddings for key: {cache_key}")
            return self.cache[cache_key].to(self.device)  # ensure it's on the correct device

        # 2. try to load from disk cache
        cache_path = self.get_cache_path(cache_key)
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached embeddings from disk: {cache_path}")
            try:
                # load directly to the target device if possible, otherwise load to cpu then move
                embeddings = torch.load(cache_path, map_location=self.device)
                print(f"  Loaded embeddings shape: {embeddings.shape}")
                # store in memory cache after loading from disk
                self.cache[cache_key] = embeddings
                return embeddings.to(self.device)  # ensure correct device
            except Exception as e:
                print(f"  Error loading cache file {cache_path}: {e}. Recomputing...")

        # 3. extract features if not cached - must call the base extractor's method
        print(f"Cache miss for key '{cache_key}'. Extracting features...")
        with MemoryUsageMonitor(f"feature_extraction_cached_{cache_key}"):
             # *** call the base extractor's method which implements dwt-before-concat ***
             embeddings = self.base_extractor.get_embedding_vectors(dataloader, wavelet_params)  # pass wavelet_params

        # 4. save to cache
        if cache_key:
            # store in memory cache (ensure it's on the target device)
            self.cache[cache_key] = embeddings.to(self.device)

            # save to disk cache (save cpu tensor to avoid gpu mismatch later)
            if cache_path:
                print(f"Saving embeddings to disk: {cache_path}")
                try:
                     # move to cpu before saving for better portability
                     torch.save(embeddings.cpu(), cache_path)
                except Exception as e:
                     print(f"  Error saving cache file {cache_path}: {e}")

        return embeddings.to(self.device)  # ensure result is on correct device

