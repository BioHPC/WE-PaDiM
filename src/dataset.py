# dataset.py 
""" **
Dataset module for PaDiM with Incremental implementation.
Provides dataset loading and batch preparation for MVTec.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from typing import List, Tuple, Any, Dict, Optional


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

        # Load dataset paths and labels
        self.x, self.y, self.mask = self.load_dataset_folder()

        # Set transforms
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

        if y == 0:  # Normal sample
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:  # Anomalous sample
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

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

        # Set up directory paths
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        # Load each image type (good, crack, etc.)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # Get image type directory
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
                
            # Get all PNG images in this directory
            img_fpath_list = sorted([
                os.path.join(img_type_dir, f)
                for f in os.listdir(img_type_dir)
                if f.endswith('.png')
            ])
            x.extend(img_fpath_list)

            # Load ground truth masks
            if img_type == 'good':
                # Normal samples have no masks
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                # Anomalous samples have mask files
                y.extend([1] * len(img_fpath_list))
                
                # Get corresponding mask files
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
    test_batch_size: int
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
    # Create train dataset
    train_dataset = MVTecDataset(
        dataset_path=data_path,
        class_name=class_name,
        is_train=True
    )
    
    # Create test dataset
    test_dataset = MVTecDataset(
        dataset_path=data_path,
        class_name=class_name,
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_all_class_names(data_path: str) -> List[str]:
    """
    Get all class names from the MVTec dataset directory.
    
    Args:
        data_path: Path to MVTec dataset
        
    Returns:
        List of class names
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"MVTec dataset path does not exist: {data_path}")
        
    return [d for d in os.listdir(data_path) 
            if os.path.isdir(os.path.join(data_path, d))]

def create_efficient_dataloaders(
    data_path: str, 
    class_name: str, 
    train_batch_size: int, 
    test_batch_size: int,
    num_workers: int = 2  # Reduced from 4 to 2
) -> Tuple[DataLoader, DataLoader]:
    """Create memory-efficient dataloaders."""
    train_dataset = MVTecDataset(
        dataset_path=data_path,
        class_name=class_name,
        is_train=True
    )
    
    test_dataset = MVTecDataset(
        dataset_path=data_path,
        class_name=class_name,
        is_train=False
    )
    
    # Use persistent workers and proper prefetching
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False  # Changed from True to False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False  # Changed from True to False
    )
    
    return train_loader, test_loader
