"""
Visual Model - Data Loader
Loads plot images from Combinations folder for training
"""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from visual_mapping import (
    VISUAL_MAPPING, BASE_FOLDERS,
    get_anomaly_label, get_label_index,
    ANOMALY_CLASSES
)


class TimeSeriesPlotDataset(Dataset):
    """Dataset for loading time series plot images"""

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to plot images
            labels: List of corresponding anomaly labels (as indices)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[idx]

        return image, label


def get_image_transforms(image_size=224, augment=True):
    """Get image transformation pipeline"""
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    return transform


def load_combination_images(combinations_dir, max_images_per_combo=None):
    """
    Load all plot images from combinations folder

    Args:
        combinations_dir: Path to Combinations directory
        max_images_per_combo: Optional limit on images per combination

    Returns:
        image_paths: List of image file paths
        labels: List of corresponding label indices
        label_names: List of corresponding label names
    """
    combinations_dir = Path(combinations_dir)

    image_paths = []
    labels = []
    label_names = []

    stats = defaultdict(int)

    print("\n" + "=" * 70)
    print("  LOADING PLOT IMAGES FROM COMBINATIONS")
    print("=" * 70)

    for base_folder, combo_folders in BASE_FOLDERS.items():
        print(f"\n  {base_folder}:")

        for combo_name in combo_folders:
            # Get anomaly label
            try:
                anomaly_label = get_anomaly_label(combo_name)
                label_idx = get_label_index(anomaly_label)
            except ValueError as e:
                print(f"    WARNING: {combo_name}: {e}")
                continue

            # Find all plot folders recursively
            combo_base_path = combinations_dir / base_folder / combo_name
            plot_folders = list(combo_base_path.rglob("plots"))

            # Collect all PNG images from all plot folders
            combo_images = []
            for plot_folder in plot_folders:
                png_files = list(plot_folder.glob("*.png"))
                combo_images.extend(png_files)

            if not combo_images:
                print(f"    WARNING: {combo_name}: No plots found")
                continue

            # Optionally limit images per combination
            if max_images_per_combo and len(combo_images) > max_images_per_combo:
                combo_images = np.random.choice(
                    combo_images,
                    size=max_images_per_combo,
                    replace=False
                ).tolist()

            # Add to dataset
            for img_path in combo_images:
                image_paths.append(str(img_path))
                labels.append(label_idx)
                label_names.append(anomaly_label)

            stats[anomaly_label] += len(combo_images)
            print(f"    OK: {combo_name}: {len(combo_images)} images -> {anomaly_label}")

    print("\n" + "=" * 70)
    print("  DATASET STATISTICS")
    print("=" * 70)
    print(f"\n  Total images: {len(image_paths)}")
    print(f"\n  {'Anomaly Type':<25} {'Images':>10}")
    print("  " + "-" * 40)

    for anomaly in ANOMALY_CLASSES:
        count = stats.get(anomaly, 0)
        print(f"  {anomaly:<25} {count:>10}")

    print("=" * 70)

    return image_paths, labels, label_names


def create_dataloaders(image_paths, labels,
                       train_split=0.8,
                       batch_size=32,
                       image_size=224,
                       num_workers=4):
    """
    Create train and validation dataloaders

    Args:
        image_paths: List of image paths
        labels: List of label indices
        train_split: Fraction of data for training
        batch_size: Batch size
        image_size: Image size for resizing
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader
    """
    # Shuffle and split
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create datasets
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_dataset = TimeSeriesPlotDataset(
        train_paths,
        train_labels,
        transform=get_image_transforms(image_size, augment=True)
    )

    val_dataset = TimeSeriesPlotDataset(
        val_paths,
        val_labels,
        transform=get_image_transforms(image_size, augment=False)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\n" + "=" * 70)
    print("  DATALOADERS CREATED")
    print("=" * 70)
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print("=" * 70)

    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    combinations_dir = Path("C:/Users/user/Desktop/STATIONARY/Combinations")

    image_paths, labels, label_names = load_combination_images(
        combinations_dir,
        max_images_per_combo=100
    )

    print(f"\n  Sample:")
    for i in range(min(5, len(image_paths))):
        print(f"    {Path(image_paths[i]).name} -> {label_names[i]}")
