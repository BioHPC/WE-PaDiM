"""
Generate a small set of qualitative anomaly detection visualizations without
running the full benchmark. Uses the existing WE-PaDiM pipeline but limits the
number of samples loaded per class.

Example:
    PYTHONPATH=./src python scripts/quick_visuals.py \\
        --dataset_type mvtec \\
        --data_path ./data/MVTec \\
        --classes bottle \\
        --model efficientnet-b0 \\
        --wavelet_type haar --wavelet_level 1 --subbands LL LH HL \\
        --sigma 2.0 --cov_reg 0.1 \\
        --max_train 40 --max_test 30 --max_visuals 16
"""

import argparse
import os
import time
import torch

from models import FeatureExtractor
from evaluator import PaDiMEvaluator
from dataset import get_all_class_names
from utils import setup_device, set_gpu_environment

def parse_args():
    p = argparse.ArgumentParser(description="Quick qualitative visual generator")
    p.add_argument("--dataset_type", choices=["mvtec", "visa"], default="mvtec")
    p.add_argument("--data_path", default="./data/MVTec")
    p.add_argument("--classes", nargs="*", default=None, help="Space-separated class list; defaults to all")
    p.add_argument("--model", default="efficientnet-b0")
    p.add_argument("--wavelet_type", default="haar")
    p.add_argument("--wavelet_level", type=int, default=1)
    p.add_argument("--subbands", nargs="+", default=["LL", "LH", "HL"])
    p.add_argument("--sigma", type=float, default=4.0, help="Gaussian sigma for score smoothing")
    p.add_argument("--cov_reg", type=float, default=0.01, help="Diagonal regularization for covariance")
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--test_bs", type=int, default=16)
    p.add_argument("--max_train", type=int, default=40, help="Max training samples per class (0 = all)")
    p.add_argument("--max_test", type=int, default=30, help="Max test samples per class (0 = all)")
    p.add_argument("--visuals_per_class", type=int, default=3, help="Number of visuals to save per class (2-3 recommended)")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--save_dir", default=None, help="Optional custom save dir")
    return p.parse_args()

def main():
    args = parse_args()

    # select device
    set_gpu_environment(args.gpu_id)
    device = setup_device(args.gpu_id)
    torch.manual_seed(1024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1024)

    # resolve classes
    if args.classes:
        class_names = args.classes
    else:
        class_names = get_all_class_names(args.data_path, dataset_type=args.dataset_type)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_root = args.save_dir or os.path.join(
        "results",
        "quick_visuals",
        args.dataset_type,
        f"{args.model}_{timestamp}"
    )
    os.makedirs(save_root, exist_ok=True)

    # initialize extractor and evaluator
    feature_extractor = FeatureExtractor(
        model_name=args.model,
        device=device,
        resource_monitor=None
    )
    evaluator = PaDiMEvaluator(
        feature_extractor=feature_extractor,
        data_path=args.data_path,
        save_path=save_root,
        test_batch_size=args.test_bs,
        resource_monitor=None,
        device=device,
        dataset_type=args.dataset_type
    )

    wavelet_params = {
        "wavelet_type": args.wavelet_type,
        "wavelet_level": args.wavelet_level,
        "wavelet_kept_subbands": args.subbands,
    }

    print(f"\nSaving visuals to: {save_root}")
    for cls in class_names:
        print(f"\n=== {cls} ===")
        cls_dir = os.path.join(save_root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        # choose tight limits so we only visualize a couple examples per class
        effective_max_test = args.max_test if args.max_test > 0 else args.visuals_per_class
        effective_max_visuals = args.visuals_per_class
        res = evaluator.evaluate_single_class(
            cls,
            sigma=args.sigma,
            cov_reg=args.cov_reg,
            save_dir=cls_dir,
            train_batch_size=args.train_bs,
            wavelet_type=wavelet_params["wavelet_type"],
            wavelet_level=wavelet_params["wavelet_level"],
            wavelet_kept_subbands=wavelet_params["wavelet_kept_subbands"],
            max_train_samples=None if args.max_train <= 0 else args.max_train,
            max_test_samples=None if effective_max_test <= 0 else effective_max_test,
            max_visuals=None if effective_max_visuals <= 0 else effective_max_visuals,
        )
        print(f"Image AUC (subset): {res.get('img_auc', 0):.4f} | Pixel AUC (subset): {res.get('pixel_auc', 0):.4f}")
        print(f"Saved visuals under: {cls_dir}")

    print("\nDone. Attach the saved PNG grids to the paper.")

if __name__ == "__main__":
    main()
