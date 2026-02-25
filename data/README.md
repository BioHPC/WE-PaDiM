# Data Directory

This directory holds the datasets required for running WE-PaDiM experiments on both **MVTec AD** and **VisA**.

## MVTec Anomaly Detection (MVTec AD)

Due to licensing restrictions, the dataset is **not** included in this repository.

### Download Instructions

1. **Download the dataset**: You can download the dataset from the official MVTec website:
   https://www.mvtec.com/company/research/datasets/mvtec-ad

2. **Extract the files**: Unzip the downloaded file (`mvtec_anomaly_detection.tar.xz`).

### Required Directory Structure

After extraction, place the dataset folder inside this `data/` directory. The code expects the following structure:

```
WE-PaDiM_VisA/
└── data/
    └── MVTec/
        ├── bottle/
        │   ├── train/
        │   ├── test/
        │   └── ground_truth/
        ├── cable/
        ├── capsule/
        ├── ... (all 15 MVTec AD classes)
        └── LICENSE.txt
```

The runner scripts look for MVTec AD at `./data/MVTec` by default. Override this path with `--data_path` if needed.

## VisA: High-Quality Visual Anomaly Dataset

VisA is an additional benchmark supported by this repository. Download it from the official project page or mirrored academic sources (e.g., [https://github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)). VisA is released under its own license—review and comply before use.

### Download Instructions
1. **Download the dataset** (usually distributed as multiple ZIP files).
2. **Extract** all archives while preserving the folder hierarchy.

### Required Directory Structure

```
WE-PaDiM_VisA/
└── data/
    └── VisA/
        ├── candle/
        ├── capsules/
        ├── cashew/
        ├── ... (all VisA subsets)
        ├── split_csv/
        └── LICENSE-DATASET
```

Runners expect VisA at `./data/VisA` by default. Use `--data_path` to point elsewhere if necessary.

---

For both datasets, keep the original train/test/ground-truth folder structures untouched. The scripts rely on the canonical file layout to locate samples and annotations.
