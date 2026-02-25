# dataset.py
"""
Dataset module for PaDiM with Incremental implementation.
Provides dataset loading and batch preparation for MVTec.
"""

import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from typing import List, Tuple, Any, Dict, Optional

def _infer_dataset_type(dataset_path: str) -> str:
    """Best-effort detection of dataset layout."""
    if not os.path.isdir(dataset_path):
        return 'mvtec'

    class_dirs = [
        os.path.join(dataset_path, d)
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    for cls_dir in class_dirs:
        if os.path.isdir(os.path.join(cls_dir, 'train')) and os.path.isdir(os.path.join(cls_dir, 'test')):
            return 'mvtec'

    for cls_dir in class_dirs:
        if os.path.isdir(os.path.join(cls_dir, 'Data')):
            return 'visa'

    return 'mvtec'

def resolve_dataset_type(dataset_path: str, dataset_type: str) -> str:
    """Resolve desired dataset type, supporting automatic detection."""
    normalized = dataset_type.lower().strip()
    if normalized in {'mvtec', 'visa'}:
        return normalized
    if normalized not in {'auto', ''}:
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'.")
    return _infer_dataset_type(dataset_path)

def _select_dataset_class(dataset_path: str, dataset_type: str):
    """Return the Dataset subclass to instantiate for the requested dataset."""
    resolved = resolve_dataset_type(dataset_path, dataset_type)
    if resolved == 'visa':
        return VisADataset, resolved
    return MVTecDataset, 'mvtec'

# this class goes inside we-padim/src/dataset.py

class VisADataset(Dataset):
    """VisA dataset for anomaly detection."""

    def __init__(
        self,
        dataset_path: str,
        class_name: str,
        is_train: bool = True,
        resize: int = 256,
        cropsize: int = 224,
        split_csv_name: Optional[str] = None
    ):
        """
        Initialize VisA dataset.
        """
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        env_split = os.getenv('VISA_SPLIT_CSV')
        chosen_split = split_csv_name or env_split or '1cls.csv'
        if chosen_split and not chosen_split.endswith('.csv'):
            chosen_split = f"{chosen_split}.csv"
        self.split_csv_name = chosen_split
        self.split_csv_path = self._resolve_split_csv_path(chosen_split)

        # load dataset paths and labels
        self.x, self.y, self.mask = self.load_dataset_folder()

        # image transformations (these can remain the same)
        self.transform_x = T.Compose([
            T.Resize(resize, Image.LANCZOS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_mask = T.Compose([
            T.Resize(resize, Image.NEAREST),
            T.CenterCrop(cropsize),
            T.ToTensor()
        ])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a single item from the dataset.
        """
        x_path, y, mask_path = self.x[idx], self.y[idx], self.mask[idx]

        # visa images are .jpg, so we ensure they are converted to rgb
        x = Image.open(x_path).convert('RGB')
        x = self.transform_x(x)

        if y == 0:  # normal sample
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:  # anomalous sample
            # visa masks are grayscale, so we convert to 'l' mode
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)
            mask = (mask > 0).float()

        return x, y, mask

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.x)

    def load_dataset_folder(self) -> Tuple[List[str], List[int], List[Any]]:
        """
        Load dataset file paths and labels for the VisA dataset.
        This is the core logic that handles the different folder structure.
        """
        split_key = 'train' if self.is_train else 'test'

        if self.split_csv_path and os.path.isfile(self.split_csv_path):
            return self._load_from_split_csv(split_key)

        return self._load_from_directory_structure(split_key)

    def _resolve_split_csv_path(self, split_csv_name: Optional[str]) -> Optional[str]:
        if not split_csv_name:
            return None

        if os.path.isabs(split_csv_name):
            return split_csv_name if os.path.isfile(split_csv_name) else None

        candidates = [
            os.path.join(self.dataset_path, 'split_csv', split_csv_name),
            os.path.join(self.dataset_path, split_csv_name)
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        return None

    def _resolve_resource_path(self, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.dataset_path, relative_path)

    def _load_from_split_csv(self, split_key: str) -> Tuple[List[str], List[int], List[Any]]:
        x: List[str] = []
        y: List[int] = []
        mask: List[Optional[str]] = []

        with open(self.split_csv_path, mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if (row.get('object') or '').strip() != self.class_name:
                    continue
                if (row.get('split') or '').strip().lower() != split_key:
                    continue

                label = (row.get('label') or '').strip().lower()
                image_rel = (row.get('image') or '').strip()
                mask_rel = (row.get('mask') or '').strip()

                if not image_rel:
                    raise ValueError(
                        f"Split file '{self.split_csv_path}' contains an entry without an image path for class '{self.class_name}'."
                    )

                image_path = self._resolve_resource_path(image_rel)
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"VisA image listed in split file not found: {image_path}")

                if label in {'normal', 'good', '0'}:
                    x.append(image_path)
                    y.append(0)
                    mask.append(None)
                else:
                    anomaly_mask_path = self._resolve_resource_path(mask_rel) if mask_rel else None
                    if anomaly_mask_path is None or not os.path.isfile(anomaly_mask_path):
                        raise FileNotFoundError(
                            f"VisA anomaly mask missing for '{image_rel}' using entry '{mask_rel}'."
                        )
                    x.append(image_path)
                    y.append(1)
                    mask.append(anomaly_mask_path)

        if not x:
            raise FileNotFoundError(
                f"No VisA samples found for class '{self.class_name}' in split '{split_key}' using '{self.split_csv_path}'."
            )

        assert len(x) == len(y) == len(mask), 'VisA split produced inconsistent sample counts'
        return list(x), list(y), list(mask)

    def _load_from_directory_structure(self, split_key: str) -> Tuple[List[str], List[int], List[Any]]:
        x: List[str] = []
        y: List[int] = []
        mask: List[Optional[str]] = []

        if split_key == 'train':
            img_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'train', 'normal')
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(
                    f"VisA training directory not found and no split CSV available. Expected directory: {img_dir}"
                )

            img_fpath_list = sorted(
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.lower().endswith('.jpg') or f.lower().endswith('.png')
            )
            if not img_fpath_list:
                raise FileNotFoundError(
                    f"No training images found for class '{self.class_name}' using directory layout in {img_dir}"
                )

            x.extend(img_fpath_list)
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))

        else:
            normal_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'test', 'normal')
            if not os.path.isdir(normal_dir):
                raise FileNotFoundError(
                    f"VisA test 'normal' directory not found and no split CSV available. Expected directory: {normal_dir}"
                )
            img_fpath_list_normal = sorted(
                os.path.join(normal_dir, f)
                for f in os.listdir(normal_dir)
                if f.lower().endswith('.jpg') or f.lower().endswith('.png')
            )
            x.extend(img_fpath_list_normal)
            y.extend([0] * len(img_fpath_list_normal))
            mask.extend([None] * len(img_fpath_list_normal))

            anomaly_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'test', 'anomaly')
            if not os.path.isdir(anomaly_dir):
                raise FileNotFoundError(
                    f"VisA test 'anomaly' directory not found and no split CSV available. Expected directory: {anomaly_dir}"
                )
            img_fpath_list_anomaly = sorted(
                os.path.join(anomaly_dir, f)
                for f in os.listdir(anomaly_dir)
                if f.lower().endswith('.jpg') or f.lower().endswith('.png')
            )
            x.extend(img_fpath_list_anomaly)
            y.extend([1] * len(img_fpath_list_anomaly))

            gt_dir_candidates = [
                os.path.join(self.dataset_path, self.class_name, 'Seg_mask', 'anomaly'),
                os.path.join(self.dataset_path, self.class_name, 'Data', 'Masks', 'Anomaly')
            ]
            gt_dir = next((d for d in gt_dir_candidates if os.path.isdir(d)), None)
            if gt_dir is None:
                raise FileNotFoundError(
                    f"VisA ground-truth mask directory not found near '{self.dataset_path}/{self.class_name}'."
                )

            mask_fpath_list = []
            for f in img_fpath_list_anomaly:
                base_name = os.path.splitext(os.path.basename(f))[0]
                mask_candidates = [
                    os.path.join(gt_dir, base_name + '.png'),
                    os.path.join(gt_dir, base_name + '_mask.png')
                ]
                mask_path = next((m for m in mask_candidates if os.path.isfile(m)), None)
                if mask_path is None:
                    raise FileNotFoundError(f"Missing VisA anomaly mask for image '{f}'.")
                mask_fpath_list.append(mask_path)
            mask.extend(mask_fpath_list)

        assert len(x) == len(y) == len(mask), 'Number of images, labels, and masks should be the same'
        return list(x), list(y), list(mask)

class MVTecDataset(Dataset):
    """MVTec AD dataset for anomaly detection."""

    def __init__(
        self,
        dataset_path: str,
        class_name: str,
        is_train: bool = True,
        resize: int = 256,
        cropsize: int = 224
    ):
        """
        Initialize MVTec dataset.

        Args:
            dataset_path: Path to MVTec dataset
            class_name: Name of the class to load
            is_train: Whether to load training set or test set
            resize: Size to resize the image to
            cropsize: Size to center crop the image to
        """
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset paths and labels
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([
            T.Resize(resize, Image.LANCZOS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_mask = T.Compose([
            T.Resize(resize, Image.NEAREST),
            T.CenterCrop(cropsize),
            T.ToTensor()
        ])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image, label, mask)
        """
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:  # normal sample
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:  # anomalous sample
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
            mask = (mask > 0).float()

        return x, y, mask

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.x)

    def load_dataset_folder(self) -> Tuple[List[str], List[int], List[Any]]:
        """
        Load dataset file paths and labels.

        Returns:
            Tuple of (image_paths, labels, mask_paths)
        """
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        # set up directory paths
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        # load each image type (good, crack, etc.)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # get image type directory
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            # get all png images in this directory
            img_fpath_list = sorted([
                os.path.join(img_type_dir, f)
                for f in os.listdir(img_type_dir)
                if f.endswith('.png')
            ])
            x.extend(img_fpath_list)

            # load ground truth masks
            if img_type == 'good':
                # normal samples have no masks
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                # anomalous samples have mask files
                y.extend([1] * len(img_fpath_list))

                # get corresponding mask files
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + '_mask.png')
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)

def load_dataset(
    data_path: str,
    class_name: str,
    train_batch_size: int,
    test_batch_size: int,
    dataset_type: str = 'auto',
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Load train and test datasets for a specific class.

    Args:
        data_path: Path to MVTec dataset
        class_name: Name of the class to load
        train_batch_size: Batch size for training set
        test_batch_size: Batch size for test set

    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # create train dataset
    dataset_cls, _ = _select_dataset_class(data_path, dataset_type)

    train_dataset = dataset_cls(
        dataset_path=data_path,
        class_name=class_name,
        is_train=True
    )

    # create test dataset
    test_dataset = dataset_cls(
        dataset_path=data_path,
        class_name=class_name,
        is_train=False
    )

    # create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

def get_all_class_names(data_path: str, dataset_type: str = 'mvtec') -> List[str]:
    """Return all class directories for the requested dataset type."""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    _, resolved_type = _select_dataset_class(data_path, dataset_type)

    class_names: List[str] = []
    for entry in sorted(os.listdir(data_path)):
        if entry.startswith('.'):
            continue
        class_dir = os.path.join(data_path, entry)
        if not os.path.isdir(class_dir):
            continue
        if resolved_type == 'mvtec':
            if os.path.isdir(os.path.join(class_dir, 'train')) and os.path.isdir(os.path.join(class_dir, 'test')):
                class_names.append(entry)
        else:
            if os.path.isdir(os.path.join(class_dir, 'Data')):
                class_names.append(entry)

    if not class_names:
        raise ValueError(f"No valid classes found in '{data_path}' for dataset type '{resolved_type}'.")

    return class_names

def create_efficient_dataloaders(
    data_path: str,
    class_name: str,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int = 2,
    dataset_type: str = 'auto',
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create memory-efficient dataloaders with optional downsampling.

    The optional ``max_*_samples`` arguments let us grab only a small subset
    of images when we just need quick qualitative outputs instead of a full
    benchmark run.
    """
    dataset_cls, _ = _select_dataset_class(data_path, dataset_type)

    train_dataset = dataset_cls(
        dataset_path=data_path,
        class_name=class_name,
        is_train=True
    )

    test_dataset = dataset_cls(
        dataset_path=data_path,
        class_name=class_name,
        is_train=False
    )

    def _maybe_subset(ds: Dataset, max_samples: Optional[int], seed: int = 1024) -> Dataset:
        if max_samples is None or max_samples <= 0 or len(ds) <= max_samples:
            return ds
        import random
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), k=max_samples)
        from torch.utils.data import Subset
        return Subset(ds, indices)

    train_dataset = _maybe_subset(train_dataset, max_train_samples)
    test_dataset = _maybe_subset(test_dataset, max_test_samples)

    # use persistent workers and proper prefetching
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False  # changed from true to false
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False  # changed from true to false
    )

    return train_loader, test_loader
