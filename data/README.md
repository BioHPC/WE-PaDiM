# Data Directory

This directory is intended to hold the datasets required for running the WE-PaDiM experiments.

## MVTec Anomaly Detection (MVTec AD) Dataset

The primary dataset used in this project is the MVTec AD dataset. Due to copyright issues, it is **not** included in this repository.

### Download Instructions

1. **Download the dataset**: You can download the dataset from the official MVTec website:
   https://www.mvtec.com/company/research/datasets/mvtec-ad

2. **Extract the files**: Unzip the downloaded file (`mvtec_anomaly_detection.tar.xz`).

### Required Directory Structure

After extraction, place the dataset folder inside this `data/` directory. The code expects the following structure:

```
WE-PaDiM/
└── data/
    └── mvtec_anomaly_detection/
        ├── bottle/
        │   ├── train/
        │   ├── test/
        │   └── ground_truth/
        ├── cable/
        ├── capsule/
        ├── ... (all 15 MVTec AD classes)
        └── LICENSE.txt
```

The runner scripts in the `scripts/` directory will look for the dataset at `./data/mvtec_anomaly_detection` by default. You can change this path using the `--data_path` argument when running the scripts.
