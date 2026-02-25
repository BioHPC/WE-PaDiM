# config.py
"""
Configuration module for PaDiM with Wavelet implementation.
Contains default settings and configuration options.
"""

import os
import argparse
import json
from typing import List, Dict, Any, Optional, Union, Tuple

# default paths
DEFAULT_DATA_PATH = "./data/MVTec"
DEFAULT_SAVE_PATH = "./results"

# default batch sizes
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_TEST_BATCH_SIZE = 32

DEFAULT_MAX_ITER = 50
DEFAULT_REG_LAMBDA = 1e-3

# default wavelet parameters
DEFAULT_WAVELET_TYPE = 'haar'
DEFAULT_WAVELET_LEVEL = 1
DEFAULT_WAVELET_KEPT_SUBBANDS = ['LL', 'LH', 'HL']

# available model configurations
AVAILABLE_MODELS = {
    "resnet18": {"type": "resnet", "version": "18"},
    "resnet34": {"type": "resnet", "version": "34"},
    "wide_resnet50_2": {"type": "resnet", "version": "wide_resnet50_2"},
    "efficientnet-b0": {"type": "efficientnet", "version": "b0"},
    "efficientnet-b1": {"type": "efficientnet", "version": "b1"},
    "efficientnet-b2": {"type": "efficientnet", "version": "b2"},
    "efficientnet-b3": {"type": "efficientnet", "version": "b3"},
    "efficientnet-b4": {"type": "efficientnet", "version": "b4"},
    "efficientnet-b5": {"type": "efficientnet", "version": "b5"},
    "efficientnet-b6": {"type": "efficientnet", "version": "b6"},
}

# all mvtec classes
MVTEC_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

VISA_CLASSES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

# available wavelet types and subbands
AVAILABLE_WAVELETS = [
    'haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8',
    'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8',
    'coif1', 'coif2', 'coif3', 'coif4', 'coif5'
]
AVAILABLE_SUBBANDS = ['LL', 'LH', 'HL', 'HH']

# default wavelet parameter grid for comprehensive grid search (if used)
DEFAULT_WAVELET_PARAM_GRID = {
    'wavelet_type': ['haar', 'db2', 'db4', 'sym4'],
    'wavelet_level': [1, 2],
    'wavelet_kept_subbands': [
        ['LL'], ['LL', 'LH'], ['LL', 'LH', 'HL'], ['LL', 'LH', 'HL', 'HH']
    ],
    'sigma': [2.0, 4.0, 6.0],
    'cov_reg': [0.01, 0.001]
}

# experiment configurations for 'paper' type runs
# ensure these are updated if parameters change
EXPERIMENT_CONFIGS = {
    "main_comparison": [
        {
            "name": "WE-PaDiM (EffNet-B0)",
            "model": "efficientnet-b0",
            "wavelet_type": "db4",  # example best param
            "wavelet_level": 1,  # example best param
            "wavelet_kept_subbands": ["LL", "LH", "HL"],  # example best param
            "sigma": 4.0,  # example best param
            "cov_reg": 0.01  # example best param
        },
         {
            "name": "WE-PaDiM (ResNet-18)",
            "model": "resnet18",
            "wavelet_type": "haar",
            "wavelet_level": 1,
            "wavelet_kept_subbands": ["LL", "LH", "HL"],
            "sigma": 4.0,
            "cov_reg": 0.01
        }
        # add other paper experiment configs here
    ],
    "wavelet_ablation": [
        # define ablation experiments if needed
    ],
    # add other groups as needed
}

class Config:
    """Configuration class to hold all settings."""

    def __init__(self, args=None):
        # set default values
        self.data_path = DEFAULT_DATA_PATH
        self.save_path = DEFAULT_SAVE_PATH
        self.train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
        self.test_batch_size = DEFAULT_TEST_BATCH_SIZE
        self.memory_efficient = getattr(args, 'memory_efficient', True)

        self.models = ["resnet18"]
        self.classes = None

        self.experiment_type = "single"
        self.experiment_group = "main_comparison"

        self.save_visualizations = False

        self.sigma = 4.0
        self.cov_reg = 0.01

        self.wavelet_type = DEFAULT_WAVELET_TYPE
        self.wavelet_level = DEFAULT_WAVELET_LEVEL
        self.wavelet_kept_subbands = DEFAULT_WAVELET_KEPT_SUBBANDS

        self.gpu_id = 0
        self.enable_resource_monitoring = True
        self.dataset_type = 'auto'  # 'mvtec', 'visa', or 'auto'
        # apply args passed from command line or runner script
        if args:
            self._update_from_args(args)

        # validate the final configuration after defaults and args are applied
        # this must come after _update_from_args
        self._validate()

    def _update_from_args(self, args):
        """Update configuration attributes from parsed arguments (args Namespace)."""
        for key, value in vars(args).items():
             if value is not None and hasattr(self, key):
                 if key == 'classes' and value == ["all"]:
                     setattr(self, key, None)
                 else:
                     setattr(self, key, value)

    def _validate(self):
        """Validate configuration parameters, handling single values and lists for grid search."""
        # validate models
        models_to_check = self.models if isinstance(self.models, list) else [self.models]
        for model in models_to_check:
            if model not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model: {model}. Available: {list(AVAILABLE_MODELS.keys())}")

        # validate dataset type
        if self.dataset_type not in {'mvtec', 'visa', 'auto'}:
            raise ValueError("dataset_type must be one of {'mvtec', 'visa', 'auto'}")

        # validate classes (if not none)
        if self.classes is not None:
            if not isinstance(self.classes, list):
                raise TypeError(f"Invalid type for classes: {type(self.classes)}. Expected list or None.")
            if self.dataset_type == 'visa':
                valid_classes = set(VISA_CLASSES)
            elif self.dataset_type == 'mvtec':
                valid_classes = set(MVTEC_CLASSES)
            else:  # auto: accept either dataset's canonical classes
                valid_classes = set(VISA_CLASSES) | set(MVTEC_CLASSES)
            invalid = [cls for cls in self.classes if cls not in valid_classes]
            if invalid:
                raise ValueError(f"Unknown class(es) for dataset_type '{self.dataset_type}': {invalid}")

        # validate experiment type
        valid_experiment_types = ["single", "grid_search", "paper"]
        if self.experiment_type not in valid_experiment_types:
            raise ValueError(f"Invalid experiment_type: {self.experiment_type}. Must be one of {valid_experiment_types}")

        # validate experiment group if type is 'paper'
        if self.experiment_type == "paper":
            valid_groups = list(EXPERIMENT_CONFIGS.keys())
            if self.experiment_group not in valid_groups and self.experiment_group != "all":
                raise ValueError(f"Invalid experiment_group: {self.experiment_group}. Must be 'all' or one of {valid_groups}")

        # --- validate numeric params (handle single value or list) ---
        params_to_check_positive = {'sigma': self.sigma, 'cov_reg': self.cov_reg, 'wavelet_level': self.wavelet_level}
        for name, value in params_to_check_positive.items():
            # create a list of values to check, whether input was single or list
            values_to_check = value if isinstance(value, list) else [value]
            if not values_to_check and name in ['sigma', 'cov_reg', 'wavelet_level']:  # ensure required params aren't empty lists
                 raise ValueError(f"Parameter list for {name} cannot be empty.")
            for element in values_to_check:
                 if not isinstance(element, (int, float)):
                      raise TypeError(f"Invalid type for {name} element: {type(element)}. Expected int or float.")
                 if element <= 0:
                      raise ValueError(f"Invalid value for {name} element: {element}. Must be positive.")

        # --- validate wavelet type (handle single string or list) ---
        types_to_check = self.wavelet_type if isinstance(self.wavelet_type, list) else [self.wavelet_type]
        if not types_to_check: raise ValueError("wavelet_type list cannot be empty.")
        for wt in types_to_check:
            if not isinstance(wt, str) or wt not in AVAILABLE_WAVELETS:
                raise ValueError(f"Invalid wavelet_type value: {wt}. Must be one of {AVAILABLE_WAVELETS}")

        # --- validate wavelet subbands (handle single list or list of lists) ---
        if self.wavelet_kept_subbands:  # check if it's not none or empty list
             is_list_of_lists = isinstance(self.wavelet_kept_subbands, list) and \
                                len(self.wavelet_kept_subbands) > 0 and \
                                isinstance(self.wavelet_kept_subbands[0], list)

             if is_list_of_lists:  # grid search over subband combinations
                 if not self.wavelet_kept_subbands: raise ValueError("wavelet_kept_subbands list of lists cannot be empty.")
                 for inner_list in self.wavelet_kept_subbands:
                      if not isinstance(inner_list, list):
                           raise TypeError(f"Expected list of lists for wavelet_kept_subbands grid search, but found item: {inner_list}")
                      if not inner_list: raise ValueError("Empty subband combination ([]) is not allowed.")
                      for subband in inner_list:
                           if not isinstance(subband, str) or subband not in AVAILABLE_SUBBANDS:
                               raise ValueError(f"Invalid subband '{subband}' in combination list {inner_list}. Must be one of {AVAILABLE_SUBBANDS}")
             elif isinstance(self.wavelet_kept_subbands, list):  # single list of subbands
                  if not self.wavelet_kept_subbands: raise ValueError("wavelet_kept_subbands list cannot be empty.")
                  for subband in self.wavelet_kept_subbands:
                       if not isinstance(subband, str) or subband not in AVAILABLE_SUBBANDS:
                            raise ValueError(f"Invalid subband: {subband}. Must be one of {AVAILABLE_SUBBANDS}")
             else:  # invalid type
                  raise TypeError(f"Invalid type for wavelet_kept_subbands: {type(self.wavelet_kept_subbands)}. Expected list or list of lists.")
        # allow none for wavelet_kept_subbands? if so, __init__ should set a default. if not, raise error.
        elif self.wavelet_kept_subbands is None:
            raise ValueError("wavelet_kept_subbands must be specified (cannot be None).")
        else:  # handle empty list case explicitly (e.g., self.wavelet_kept_subbands = [])
            raise ValueError("wavelet_kept_subbands cannot be an empty list.")

    def save(self, path: str):
        """Saves the configuration dictionary to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        try:
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)
            # print(f"configuration saved to {path}") # reduce verbosity
        except TypeError as e:
            print(f"Warning: Error saving full configuration to {path}: {e}. Saving basic types only.")
            serializable_dict = {k: v for k, v in config_dict.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            with open(path, 'w') as f: json.dump(serializable_dict, f, indent=4, sort_keys=True)
        except Exception as e: print(f"Error saving configuration: {e}")

    @classmethod
    def load(cls, path: str):
        """Loads configuration from a JSON file."""
        if not os.path.exists(path): raise FileNotFoundError(f"Config file not found: {path}")
        try:
             with open(path, 'r') as f: config_dict = json.load(f)
             config_instance = cls()
             for key, value in config_dict.items(): setattr(config_instance, key, value)
             config_instance._validate()  # re-validate after loading
             # print(f"configuration loaded from {path}") # reduce verbosity
             return config_instance
        except Exception as e: print(f"Error loading configuration from {path}: {e}"); raise

    '''
    def get_classes(self) -> List[str]:
        """Returns the list of classes to be used."""
        if self.classes is None:
             if 'MVTEC_CLASSES' not in globals(): raise NameError("MVTEC_CLASSES not defined in config.py")
             return MVTEC_CLASSES
        elif isinstance(self.classes, list):
             return self.classes
        else: raise TypeError(f"self.classes type error: {type(self.classes)}")
    '''
    def get_classes(self) -> List[str]:
        if self.classes is None:
            dataset_type = self.dataset_type or 'auto'
            try:
                from dataset import get_all_class_names
                return get_all_class_names(self.data_path, dataset_type=dataset_type)
            except Exception as exc:
                print(f"Warning: Unable to infer class names from '{self.data_path}' ({exc}). Falling back to defaults.")
                if dataset_type == 'visa':
                    return VISA_CLASSES
                return MVTEC_CLASSES
        elif isinstance(self.classes, list):
            return self.classes
        else:
            raise TypeError(f"self.classes type error: {type(self.classes)}")

    def get_experiment_configs(self) -> Union[List[Dict], Dict]:
        """Get experiment configurations based on experiment type and group."""
        if self.experiment_type == "paper":
            if self.experiment_group == "all":
                configs = []
                for group, group_configs in EXPERIMENT_CONFIGS.items():
                    for cfg in group_configs: cfg['experiment_group'] = group
                    configs.extend(group_configs)
                return configs
            elif self.experiment_group in EXPERIMENT_CONFIGS:
                 group_configs = EXPERIMENT_CONFIGS[self.experiment_group]
                 for cfg in group_configs: cfg['experiment_group'] = self.experiment_group
                 return group_configs
            else: raise ValueError(f"Unknown experiment_group: {self.experiment_group}")

        elif self.experiment_type == "grid_search":
            grid = {
                'wavelet_type': self.wavelet_type if isinstance(self.wavelet_type, list) else [self.wavelet_type],
                'wavelet_level': self.wavelet_level if isinstance(self.wavelet_level, list) else [self.wavelet_level],
                # ensure structure matches grid search expectation (list of lists)
                'wavelet_kept_subbands': self.wavelet_kept_subbands if isinstance(self.wavelet_kept_subbands, list) and self.wavelet_kept_subbands and isinstance(self.wavelet_kept_subbands[0], list) else [self.wavelet_kept_subbands],
                'sigma': self.sigma if isinstance(self.sigma, list) else [self.sigma],
                'cov_reg': self.cov_reg if isinstance(self.cov_reg, list) else [self.cov_reg]
            }
            return grid

        elif self.experiment_type == "single":
            single_config = {
                "wavelet_type": self.wavelet_type,
                "wavelet_level": self.wavelet_level,
                "wavelet_kept_subbands": self.wavelet_kept_subbands,
                "sigma": self.sigma,
                "cov_reg": self.cov_reg,
            }
            return single_config
        else: raise ValueError(f"Unsupported experiment_type: {self.experiment_type}")

    def __repr__(self):
        """Return a string representation of the Config object."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"Config({attrs})"

def parse_args():  # make sure this is outside the config class
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PaDiM with Wavelet for Anomaly Detection')

    # --- paths ---
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help=f'Path to MVTec dataset (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH, help=f'Path to save results (default: {DEFAULT_SAVE_PATH})')

    # --- models and classes ---
    parser.add_argument('--models', type=str, nargs='+', default=["resnet18"], choices=list(AVAILABLE_MODELS.keys()), help='Models to use (default: resnet18)')
    parser.add_argument('--classes', type=str, nargs='+', default=None, help='MVTec classes (default: all)')

    # --- experiment settings ---
    parser.add_argument('--experiment_type', type=str, default="single", choices=["single", "grid_search", "paper"], help='Type of experiment (default: single)')
    parser.add_argument('--experiment_group', type=str, default="main_comparison", choices=list(EXPERIMENT_CONFIGS.keys()) + ["all"], help='Group for paper experiments (default: main_comparison)')

    # --- wavelet parameters ---
    parser.add_argument('--wavelet_type', type=str, default=DEFAULT_WAVELET_TYPE, choices=AVAILABLE_WAVELETS, help=f'Wavelet type (default: {DEFAULT_WAVELET_TYPE})')
    # changed default to none to allow easier detection if it was explicitly set vs default
    parser.add_argument('--wavelet_level', type=int, default=None, help=f'Wavelet level (default: {DEFAULT_WAVELET_LEVEL} if None)')
    parser.add_argument('--wavelet_kept_subbands', type=str, nargs='+', default=None, choices=AVAILABLE_SUBBANDS, help=f'Subbands to keep (default: {DEFAULT_WAVELET_KEPT_SUBBANDS} if None)')

    # --- padim parameters ---
    # changed default to none to allow easier detection if it was explicitly set vs default
    parser.add_argument('--sigma', type=float, default=None, help='Gaussian smoothing sigma (default: 4.0 if None)')
    parser.add_argument('--cov_reg', type=float, default=None, help='Covariance regularization (default: 0.01 if None)')

    # --- execution options ---
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (-1 for CPU, default: 0)')  # changed default to 0
    parser.add_argument('--train_batch_size', type=int, default=None, help=f'Training batch size (default: {DEFAULT_TRAIN_BATCH_SIZE} if None)')
    parser.add_argument('--test_batch_size', type=int, default=None, help=f'Test batch size (default: {DEFAULT_TEST_BATCH_SIZE} if None)')

    parser.add_argument('--dataset_type', type=str, default='auto',
        choices=['mvtec', 'visa', 'auto'],
        help='Dataset type: mvtec, visa, or auto for automatic detection (default: auto)')
    # --- flags ---
    # use booleanoptionalaction for clearer --no-<flag> syntax (requires python 3.9+)
    try:
        parser.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=False, help='Save visualization images (--no-save-visualizations to disable)')
        parser.add_argument('--enable_resource_monitoring', action=argparse.BooleanOptionalAction, default=True, help='Enable resource monitoring (--no-enable-resource-monitoring to disable)')
        # add other flags similarly
    except AttributeError:  # fallback for python < 3.9
        parser.add_argument('--save_visualizations', action='store_true', default=False)
        parser.add_argument('--no_save_visualizations', dest='save_visualizations', action='store_false')
        parser.add_argument('--enable_resource_monitoring', action='store_true', default=True)
        parser.add_argument('--no_enable_resource_monitoring', dest='enable_resource_monitoring', action='store_false')
    args = parser.parse_args()

    # --- post-process args to handle defaults for nones ---
    # this ensures __init__ gets the actual default if arg wasn't provided
    if args.wavelet_level is None: args.wavelet_level = DEFAULT_WAVELET_LEVEL
    if args.wavelet_kept_subbands is None: args.wavelet_kept_subbands = DEFAULT_WAVELET_KEPT_SUBBANDS
    if args.sigma is None: args.sigma = 4.0  # re-set default float here
    if args.cov_reg is None: args.cov_reg = 0.01  # re-set default float here
    if args.train_batch_size is None: args.train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
    if args.test_batch_size is None: args.test_batch_size = DEFAULT_TEST_BATCH_SIZE

    return args

# example usage block for testing config.py itself
if __name__ == "__main__":
    # example of creating config without parsing args (uses defaults)
    print("--- Default Config ---")
    default_config = Config()
    print(default_config)
    print("-" * 20)

    # example of creating config by parsing args (run with e.g, --sigma 3.0)
    print("--- Config from Args ---")
    try:
        args = parse_args()  # this should be outside the config class
        config_from_args = Config(args)
        print(config_from_args)
        config_from_args.save("example_config.json")
        loaded_config = Config.load("example_config.json")
        print("\n--- Loaded Config ---")
        print(loaded_config)
        print("\n--- Getting Classes ---")
        print(loaded_config.get_classes())
        print("\n--- Getting Experiment Configs (Single) ---")
        print(loaded_config.get_experiment_configs())
    except SystemExit:  # prevent argparse from exiting the script in simple tests
        print("Argparse called sys.exit(), likely due to -h or --help.")
    except Exception as e:
        print(f"Error during example usage: {e}")
