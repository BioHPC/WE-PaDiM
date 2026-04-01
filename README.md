# Wavelet-Enhanced PaDiM (WE-PaDiM)

[![arXiv](https://img.shields.io/badge/arXiv-2508.16034-b31b1b.svg)](https://arxiv.org/abs/2508.16034)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Wavelet-Enhanced PaDiM extends the PaDiM anomaly-detection framework by inserting a discrete wavelet transform (DWT) step before feature concatenation. Combining multi-layer CNN activations with frequency-structured subbands improves both image-level and pixel-level anomaly detection on industrial inspection benchmarks such as MVTec AD and VisA.

---

## ✨ Highlights

- 🔬 **DWT-before-concatenation** selects frequency-aware features instead of random channels
- 🎯 **Two-phase workflow**: reduced hyperparameter search followed by best-config evaluation and optional subband ablations
- 📊 **Built-in support** for both **MVTec AD** and **VisA** datasets
- 🚀 **Multiple CNN backbones**: ResNet-18, EfficientNet-B0 through B6
- 🐳 **Docker support**: Run experiments in containerized environments with GPU acceleration

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Docker Usage](#docker-usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## 🔧 Prerequisites

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **CUDA**: 11.8 or 12.1 (or compatible with your PyTorch version)
- **Docker** (optional): For containerized execution with GPU support
- **Storage**: ~10GB for datasets + additional space for results

---

## 📦 Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/WE-PaDiM_VisA.git
cd WE-PaDiM_VisA

# Create and activate conda environment
conda create -n wepadim python=3.10
conda activate wepadim

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pytorch-wavelets pywavelets scipy pandas scikit-learn \
            matplotlib seaborn tqdm psutil gputil pillow
```

### Option 2: Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-org/WE-PaDiM_VisA.git
cd WE-PaDiM_VisA

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-wavelets pywavelets scipy pandas scikit-learn \
            matplotlib seaborn tqdm psutil gputil pillow
```

### Option 3: CPU-Only Installation

```bash
# Install PyTorch CPU version
pip install torch torchvision

# Install other dependencies (same as above)
pip install pytorch-wavelets pywavelets scipy pandas scikit-learn \
            matplotlib seaborn tqdm psutil gputil pillow
```

> **Note**: CPU-only mode is significantly slower and not recommended for full experiments.

---

## 📊 Dataset Setup

Download and extract the datasets into the `data/` directory:

```
WE-PaDiM_VisA/
└── data/
    ├── MVTec/          # MVTec AD dataset
    │   ├── bottle/
    │   ├── cable/
    │   └── ...
    └── VisA/           # VisA dataset
        ├── candle/
        ├── capsules/
        └── ...
```

### Datasets

- **MVTec AD**: [Download here](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **VisA**: [Download here](https://github.com/amazon-science/spot-diff)

See [`data/README.md`](data/README.md) for detailed instructions and licensing information.

---

## 🚀 Quick Start

### Test Your Installation

Run a quick test on a single class to verify everything is working:

```bash
# Test with VisA dataset
PYTHONPATH=./src python scripts/run_experiment.py \
  --dataset_type visa \
  --data_path ./data/VisA \
  --models efficientnet-b0 \
  --experiment_type single \
  --classes candle \
  --gpu_id 0

# Test with MVTec dataset
PYTHONPATH=./src python scripts/run_experiment.py \
  --dataset_type mvtec \
  --data_path ./data/MVTec \
  --models efficientnet-b0 \
  --experiment_type single \
  --classes bottle \
  --gpu_id 0
```

Results will be saved in `./results/` with JSON metrics and optional visualizations.

---

## 🔬 Running Experiments

All experiment scripts are located in `scripts/` and require `PYTHONPATH=./src` to be set.

### General Runner (Recommended)

The unified runner (`run_experiment.py`) supports single runs, grid searches, and custom configurations.

#### Single Model Run

Evaluate a single model with specific parameters:

```bash
PYTHONPATH=./src python scripts/run_experiment.py \
  --dataset_type visa \
  --data_path ./data/VisA \
  --models efficientnet-b0 \
  --experiment_type single \
  --wavelet_type haar \
  --wavelet_level 1 \
  --wavelet_kept_subbands LL LH HL \
  --sigma 4.0 \
  --cov_reg 0.01 \
  --gpu_id 0
```

#### Grid Search

Search across multiple hyperparameter combinations:

```bash
PYTHONPATH=./src python scripts/run_experiment.py \
  --dataset_type mvtec \
  --data_path ./data/MVTec \
  --models resnet18 efficientnet-b0 \
  --experiment_type grid_search \
  --wavelet_type haar sym4 \
  --wavelet_level 1 2 \
  --wavelet_subband_sets LL|LH|HL LL|LH|HL|HH \
  --sigma 2.0 4.0 \
  --cov_reg 0.01 0.1 \
  --gpu_id 0
```

#### Common Parameters

- `--dataset_type`: Dataset to use (`mvtec` or `visa`)
- `--data_path`: Path to dataset root directory
- `--models`: CNN backbone(s) to use (space-separated)
- `--experiment_type`: Type of experiment (`single` or `grid_search`)
- `--classes`: Specific classes to evaluate (default: all)
- `--gpu_id`: GPU device ID (use `-1` for CPU)
- `--save_path`: Directory for saving results (default: `./results`)

### Phase 1: Reduced Hyperparameter Search

Perform a curated hyperparameter search optimized for each backbone:

```bash
PYTHONPATH=./src python scripts/phase1_reduced_grid_search.py \
  --dataset_type visa \
  --data_path ./data/VisA \
  --model efficientnet-b0 \
  --gpu_id 0
```

**Available backbones**: `resnet18`, `efficientnet-b0`, `efficientnet-b1`, ..., `efficientnet-b6`

### Phase 2: Best-Config Evaluation

Evaluate optimized configurations from Phase 1:

```bash
PYTHONPATH=./src python scripts/phase2_best_config_runner.py \
  --dataset_type visa \
  --gpu_id 0
```

### Phase 2: Subband Ablation Study

Explore different wavelet subband combinations:

```bash
PYTHONPATH=./src python scripts/phase2_subband_ablation_runner.py \
  --dataset_type visa \
  --data_path ./data/VisA \
  --model resnet18 \
  --base_params image \
  --save_dir_base ./results/Phase2_SubbandAblation_VisA \
  --gpu_id 0
```

**Parameters**:
- `--base_params`: Use `image` for image-level AUC optimization or `pixel` for pixel-level
- `--wavelet_types`: Wavelet types to test (default: `haar`, `db4`, `sym4`)
- `--wavelet_levels`: Decomposition levels to test

### Output Format

All experiments save results under `./results/` with the following structure:

```
results/
└── <experiment_name>/
    ├── config.json                    # Experiment configuration
    ├── <model>/
    │   ├── <model>_wavelet_experiment.json
    │   ├── <model>_resource_summary.json
    │   └── visualizations/            # Optional visualization outputs
    └── ...
```

---

## 🐳 Docker Usage

Run experiments in a containerized environment with GPU support.

### Prerequisites

- Docker installed
- NVIDIA Docker runtime (`nvidia-docker2`)
- NVIDIA GPU with appropriate drivers

### Using Pre-built NVIDIA PyTorch Container

```bash
# Pull the NVIDIA PyTorch image
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# Run experiment in container
docker run --gpus all --rm --ipc=host \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  bash -c "pip install -q pytorch-wavelets pywavelets scipy pandas scikit-learn seaborn tqdm psutil gputil pillow && \
  PYTHONPATH=./src python scripts/run_experiment.py \
    --dataset_type visa \
    --data_path ./data/VisA \
    --models efficientnet-b0 \
    --experiment_type single \
    --classes candle \
    --gpu_id 0"
```

### Building Custom Docker Image

Create a `Dockerfile`:

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir \
    pytorch-wavelets pywavelets scipy pandas \
    scikit-learn seaborn tqdm psutil gputil pillow

ENV PYTHONPATH=/workspace/src:$PYTHONPATH
CMD ["/bin/bash"]
```

Build and run:

```bash
docker build -t wepadim:latest .
docker run --gpus all --rm --ipc=host \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/results:/workspace/results \
  wepadim:latest \
  python scripts/run_experiment.py --dataset_type visa --gpu_id 0
```

---

## 📁 Project Structure

```
.
├── data/                     # External datasets (not included in repo)
│   ├── MVTec/               # MVTec AD dataset
│   └── VisA/                # VisA dataset
├── docs/                     # Additional documentation
├── results/                  # Experiment outputs (generated)
├── scripts/                  # High-level experiment runners
│   ├── run_experiment.py           # Unified experiment runner
│   ├── phase1_reduced_grid_search.py
│   ├── phase2_best_config_runner.py
│   └── phase2_subband_ablation_runner.py
├── src/                      # Core implementation
│   ├── config.py            # Configuration handling
│   ├── dataset.py           # Dataset loaders
│   ├── evaluator.py         # PaDiM evaluation logic
│   ├── experiment.py        # Experiment orchestration
│   ├── grid_search.py       # Grid search utilities
│   ├── main.py              # Main experiment entry point
│   ├── metrics.py           # Evaluation metrics
│   ├── models.py            # Feature extractors
│   ├── utils.py             # Utility functions
│   ├── visualization.py     # Result visualization
│   └── wavelet_transform.py # DWT implementation
└── README.md                 # This file
```

---

## 📊 Results

WE-PaDiM supports distinct operating points for image-level detection vs. pixel-level localization:

### Image-Optimized Configuration
| Dataset | Backbone | Image AUC | Pixel AUC |
|---------|----------|-----------|----------|
| MVTec AD | EfficientNet-b1 | **98.43%** | 89.41% |
| VisA | EfficientNet-b6 | **92.43%** | 63.64% |

### Pixel-Optimized Configuration
| Dataset | Backbone | Image AUC | Pixel AUC |
|---------|----------|-----------|----------|
| MVTec AD | ResNet-18 | 97.52% | **96.65%** |
| VisA | ResNet-18 | 90.90% | **96.41%** |

*Note: These represent single-configuration operating points applied uniformly across all categories. Per-category selection can achieve higher performance (see paper for details).*

For detailed per-category results and comparisons with other methods, please refer to our paper.

---

## 🐛 Troubleshooting

### CUDA Out of Memory

- Reduce batch size: `--train_batch_size 16 --test_batch_size 16`
- Use smaller backbone: `--models resnet18`
- Enable memory-efficient mode (enabled by default)

### pytorch_wavelets Import Error

Ensure you have a CUDA-enabled PyTorch installation:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-wavelets
```

### Dataset Not Found

Verify your dataset structure matches the expected format:

```bash
ls data/MVTec/  # Should show: bottle, cable, capsule, ...
ls data/VisA/   # Should show: candle, capsules, cashew, ...
```

---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@ARTICLE{11418894,
  author={Gardner, Cory and Dhiren, Aendri and Min, Byungseok and Ahn, Tae-Hyuk},
  journal={IEEE Access}, 
  title={Wavelet-Enhanced PaDiM for Industrial Anomaly Detection}, 
  year={2026},
  volume={14},
  number={},
  pages={37702-37718},
  keywords={Feature extraction;Anomaly detection;Discrete wavelet transforms;Wavelet analysis;Image reconstruction;Location awareness;Frequency-domain analysis;Benchmark testing;Gaussian distribution;Convolutional neural networks;Anomaly detection;anomaly localization;deep learning;discrete wavelet transform;EfficientNet;ResNet;feature selection;industrial inspection},
  doi={10.1109/ACCESS.2026.3669882}}

```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgements
This research was supported by the Technology Innovation Program (ATC+ Program, Project No. 20014131, "25nm X-ray Inspection System for Semiconductor Backend Process") funded by the Ministry of Trade, Industry and Energy (MOTIE, Korea). Additional support for C.G. and T.A. was provided by the National Science Foundation (NSF) under Grant No. 2430236, and B.M. was supported by the "Regional Innovation System & Education (RISE)" through the Seoul RISE Center, funded by the Ministry of Education (MOE) and the Seoul Metropolitan Government (2025-RISE-01-019-04).
