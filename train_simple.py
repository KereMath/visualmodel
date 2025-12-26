"""
Visual Model - Simple Training Script
Automatically trains on available anomaly classes (4 classes with current data)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from visual_model.visual_mapping import VISUAL_MAPPING, BASE_FOLDERS, get_anomaly_label
from visual_model.data_loader import load_combination_images, create_dataloaders
from visual_model.model import create_model


def get_available_classes(combinations_dir):
    """
    Scan combinations to find which classes have images

    Returns:
        active_classes: List of classes that have plot images
        class_to_idx: Mapping from class name to index
    """
    combinations_dir = Path(combinations_dir)
    found_classes = set()

    print("\n" + "=" * 70)
    print("  SCANNING FOR AVAILABLE ANOMALY CLASSES")
    print("=" * 70)

    for base_folder, combo_folders in BASE_FOLDERS.items():
        for combo_name in combo_folders:
            try:
                anomaly_label = get_anomaly_label(combo_name)
                combo_base_path = combinations_dir / base_folder / combo_name
                plot_folders = list(combo_base_path.rglob("plots"))

                # Check if any plots exist
                has_plots = False
                for plot_folder in plot_folders:
                    if list(plot_folder.glob("*.png")):
                        has_plots = True
                        break

                if has_plots:
                    found_classes.add(anomaly_label)
            except:
                continue

    active_classes = sorted(list(found_classes))
    class_to_idx = {cls: idx for idx, cls in enumerate(active_classes)}

    print(f"\n  Found {len(active_classes)} classes with plot images:")
    for cls in active_classes:
        print(f"    - {cls}")
    print("=" * 70)

    return active_classes, class_to_idx


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    return running_loss / total, correct / total


def validate(model, val_loader, criterion, device):
    """Validate model and compute detailed metrics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Compute F1, Precision, Recall (macro average)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels


def main():
    """Main training function"""
    # Configuration
    COMBINATIONS_DIR = Path("C:/Users/user/Desktop/STATIONARY/Combinations")
    OUTPUT_DIR = Path("c:/Users/user/Desktop/STATIONARY/tsfresh ensemble/visual_model/trained_models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224
    TRAIN_SPLIT = 0.8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Using device: {device}")

    # Find available classes
    active_classes, class_to_idx = get_available_classes(COMBINATIONS_DIR)
    num_classes = len(active_classes)

    if num_classes == 0:
        print("\n  ERROR: No classes with plot images found!")
        return

    # Load images and remap labels to active classes
    print("\n" + "=" * 70)
    print("  LOADING IMAGES")
    print("=" * 70)

    # First pass: collect all images per class
    class_images = {cls: [] for cls in active_classes}

    for base_folder, combo_folders in BASE_FOLDERS.items():
        for combo_name in combo_folders:
            try:
                anomaly_label = get_anomaly_label(combo_name)

                if anomaly_label not in class_to_idx:
                    continue

                combo_base_path = COMBINATIONS_DIR / base_folder / combo_name
                plot_folders = list(combo_base_path.rglob("plots"))

                combo_images = []
                for plot_folder in plot_folders:
                    png_files = list(plot_folder.glob("*.png"))
                    combo_images.extend(png_files)

                if combo_images:
                    class_images[anomaly_label].extend(combo_images)
                    print(f"  {combo_name}: {len(combo_images)} images -> {anomaly_label}")
            except:
                continue

    # Show original distribution
    print(f"\n  Original class distribution:")
    for cls in active_classes:
        print(f"    {cls}: {len(class_images[cls])} images")

    # Balance classes: sample equally from each class
    min_samples = min(len(imgs) for imgs in class_images.values())
    print(f"\n  Balancing classes to {min_samples} images per class...")

    image_paths = []
    labels = []
    stats = Counter()

    np.random.seed(42)  # For reproducibility
    for cls in active_classes:
        label_idx = class_to_idx[cls]
        imgs = class_images[cls]

        # Sample min_samples from this class
        if len(imgs) > min_samples:
            selected_imgs = np.random.choice(imgs, size=min_samples, replace=False)
        else:
            selected_imgs = imgs

        for img_path in selected_imgs:
            image_paths.append(str(img_path))
            labels.append(label_idx)

        stats[cls] = len(selected_imgs)

    print(f"\n  Balanced class distribution:")
    for cls in active_classes:
        print(f"    {cls}: {stats[cls]} images")
    print(f"\n  Total images after balancing: {len(image_paths)}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        image_paths, labels,
        train_split=TRAIN_SPLIT,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    # Create model
    model = create_model(
        num_classes=num_classes,
        pretrained=True,
        backbone='resnet50',
        device=device
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
    print("=" * 70)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_f1)  # Use F1 for scheduling

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")
        print(f"Val   F1:   {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # Save best model (based on F1 score)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'num_classes': num_classes,
                'active_classes': active_classes,
                'class_to_idx': class_to_idx
            }
            torch.save(checkpoint, OUTPUT_DIR / 'best_model.pth')
            print(f">>> NEW BEST MODEL! F1: {val_f1:.4f}, Acc: {val_acc*100:.2f}%")

    # Final evaluation on best model
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION (BEST MODEL)")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    _, final_acc, final_prec, final_rec, final_f1, final_preds, final_labels = validate(
        model, val_loader, criterion, device
    )

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:  {final_acc*100:.2f}%")
    print(f"    Precision: {final_prec:.4f}")
    print(f"    Recall:    {final_rec:.4f}")
    print(f"    F1-Score:  {final_f1:.4f}")

    # Per-class metrics
    print(f"\n  Per-Class Metrics:")
    print(f"  {'-'*70}")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-'*70}")

    # Compute per-class metrics
    for idx, cls_name in enumerate(active_classes):
        cls_mask = np.array(final_labels) == idx
        cls_preds = np.array(final_preds)[cls_mask]
        cls_labels = np.array(final_labels)[cls_mask]

        if len(cls_labels) > 0:
            cls_prec = precision_score(cls_labels, cls_preds, average='binary', pos_label=idx, zero_division=0)
            cls_rec = recall_score(cls_labels, cls_preds, average='binary', pos_label=idx, zero_division=0)
            cls_f1 = f1_score(cls_labels, cls_preds, average='binary', pos_label=idx, zero_division=0)
            support = len(cls_labels)
            print(f"  {cls_name:<25} {cls_prec:>10.4f} {cls_rec:>10.4f} {cls_f1:>10.4f} {support:>10}")

    # Confusion Matrix
    cm = confusion_matrix(final_labels, final_preds)
    print(f"\n  Confusion Matrix:")
    print(f"  {'-'*70}")
    header = "True / Pred"
    print(f"  {header:<20}", end='')
    for cls in active_classes:
        print(f"{cls[:8]:>10}", end='')
    print()
    print(f"  {'-'*70}")
    for i, true_cls in enumerate(active_classes):
        print(f"  {true_cls:<20}", end='')
        for j in range(len(active_classes)):
            print(f"{cm[i][j]:>10}", end='')
        print()
    print(f"  {'-'*70}")

    # Save results
    model_info = {
        'num_classes': num_classes,
        'active_classes': active_classes,
        'class_to_idx': class_to_idx,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'total_images': len(image_paths),
        'train_images': len(train_loader.dataset),
        'val_images': len(val_loader.dataset),
        'class_distribution': dict(stats),
        'confusion_matrix': cm.tolist()
    }

    with open(OUTPUT_DIR / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best F1-Score: {best_val_f1:.4f}")
    print(f"  Best Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Model saved to: {OUTPUT_DIR / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
