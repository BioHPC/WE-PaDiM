# Wavelet-Enhanced PaDiM (WE-PaDiM) for Industrial Anomaly Detection

This repository contains the PyTorch implementation for the paper: **Wavelet-Enhanced PaDiM for Industrial Anomaly Detection**.

WE-PaDiM integrates the Discrete Wavelet Transform (DWT) with multi-layer CNN features to provide a structured, frequency-domain alternative to random feature selection in the PaDiM framework. This "DWT-before-concatenation" strategy offers a method for feature selection and dimensionality management based on frequency content relevant to anomalies.

## Abstract

Anomaly detection and localization in industrial images are essential for automated quality inspection. PaDiM, a prominent method, models the distribution of normal image features extracted by pre-trained Convolutional Neural Networks (CNNs) but typically relies on random channel selection for dimensionality reduction, potentially discarding structured information. 

We propose Wavelet-Enhanced PaDiM (WE-PaDiM), which integrates Discrete Wavelet Transform (DWT) analysis with multi-layer CNN features in a structured manner. WE-PaDiM applies 2D DWT individually to feature maps extracted from multiple layers of a backbone CNN. Specific frequency subband coefficients (e.g., approximation LL, details LH, HL) are selected from each layer's DWT output, spatially aligned if necessary, and then concatenated channel-wise before being modeled using PaDiM's patch-based multivariate Gaussian approach. 

Our evaluation on the MVTec AD dataset shows that WE-PaDiM achieves high performance, yielding average results of approximately 99.32% Image-AUC and 92.10% Pixel-AUC with per-class optimized configurations.

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@misc{Gardner_Min_Ahn_2024,
      title={Wavelet-Enhanced PaDiM for Industrial Anomaly Detection}, 
      author={Gardner, Cory and Min, Byungseok and Ahn, Tae-Hyuk},
      year={2025},
      eprint={2508.16034},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.16034}, 
}
```

## Workflow Overview

The WE-PaDiM workflow consists of a training and a testing phase. In both phases, features are extracted from a CNN backbone, processed with a per-layer 2D DWT, and then specific subbands are selected and concatenated before statistical modeling or inference.

*(See Figure 2 in the paper for a visual representation of the workflow.)*

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-enabled GPU (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/WE-PaDiM.git
cd WE-PaDiM
```

### 2. Set Up the Environment

We recommend using a virtual environment (e.g., Conda or venv).

```bash
# Using Conda
conda create -n wepadim python=3.8
conda activate wepadim

# Install dependencies
# Adjust depending on CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install pytorch-wavelets pandas scikit-learn scipy matplotlib seaborn tqdm psutil gputil pillow numpy PyWavelets
```

### 3. Download the MVTec AD Dataset

Download the MVTec Anomaly Detection (MVTec AD) dataset from the official website: https://www.mvtec.com/company/research/datasets/mvtec-ad

Unzip the dataset and place it in the `data/` directory. The expected structure is:

```
./data/
└── mvtec_anomaly_detection/
    ├── bottle/
    ├── cable/
    ├── capsule/
    ├── ... (all 15 classes)
    └── LICENSE.txt
```

Ensure your scripts point to `./data/mvtec_anomaly_detection` as the `data_path`.

## How to Run Experiments

The `scripts/` directory contains runners for reproducing the key experiments from the paper. Results will be saved to the `./results/` directory by default.

### Phase 1: Hyperparameter Grid Search

This script runs a grid search over key hyperparameters (wavelet type, level, sigma, covariance regularization) for a specified backbone model.

**Usage:**

```bash
PYTHONPATH=./src python scripts/final_gridsearch.py \
    --data_path ./data/mvtec_anomaly_detection \
    --model efficientnet-b0 \
    --gpu_id 0
```

- `--model`: Choose a backbone from the available options (e.g., `resnet18`, `efficientnet-b0` to `efficientnet-b6`).
- `--gpu_id`: Specify the GPU to use.

### Phase 2: Subband Ablation Study

This script performs an ablation study by iterating through all 15 non-empty subband combinations, using a fixed set of "best" hyperparameters determined from Phase 1.

**Usage:**

```bash
python scripts/final_gridsearch_subband_ablation.py \
    --data_path ./data/mvtec_anomaly_detection \
    --model efficientnet-b0 \
    --base_params image \
    --gpu_id 0
```

- `--base_params`: Choose which set of optimized parameters to use as the base: `'image'` (for Image AUC) or `'pixel'` (for Pixel AUC).

### Evaluating Best Models

This script runs the final evaluation using the per-class optimized configurations identified from the comprehensive grid searches.

**Usage:**

```bash
python scripts/final_gridsearch_best_models.py --gpu_id 0
```

## Parsing Results

After running the grid search, you can parse the generated `_final.json` files into ranked CSV summaries using the provided scripts.

**Usage:**
Navigate to the directory containing the results folders (e.g., `./results/WEPaDiM_Phase1_Reduced/`) and run the shell scripts:

```bash
# To parse and rank results by Image AUC
bash results/parse_image_result.sh

# To parse and rank results by Pixel AUC
bash results/parse_pixel_result.sh
```

This will generate `_grid_summary_ranked.csv` and `_grid_summary_ranked_by_pixel.csv` files in each experiment directory.

## Repository Structure

```
.
├── data/                  # Placeholder for MVTec AD dataset
├── results/               # For parsing scripts and experiment outputs
├── scripts/               # High-level experiment runner scripts
├── src/                   # Core source code for WE-PaDiM
├── .gitignore
├── LICENSE
└── README.md
```

## Acknowledgments

This research was supported by the Technology Innovation Program (ATC+ Program, Project No. 20014131) funded by the Ministry of Trade, Industry and Energy (MOTIE, Korea). Additional support was provided by the National Science Foundation (NSF) under Grant No. 2430236, and the Faculty Research Fund of Sejong University.
