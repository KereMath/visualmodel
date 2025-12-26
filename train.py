"""
Visual Model - Training Script
Train CNN on plot images for 8-class anomaly classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import create_model
from data_loader import load_combination_images, create_dataloaders
from visual_mapping import ANOMALY_CLASSES, IDX_TO_CLASS


class AnomalyTrainer:
    """Trainer for anomaly classification model"""

    def __init__(self, model, device, output_dir):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels

    def train(self, train_loader, val_loader, num_epochs, learning_rate):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

        print("\n" + "=" * 70)
        print("  STARTING TRAINING")
        print("=" * 70)
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {self.device}")
        print("=" * 70)

        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"  Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")

            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer
            )

            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(
                val_loader, criterion
            )

            # Update scheduler
            scheduler.step(val_acc)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print epoch summary
            print(f"\n  Results:")
            print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"    Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"    ✓ New best validation accuracy: {val_acc*100:.2f}%")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)

        # Save final results
        self.save_training_history()

        print("\n" + "=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print("=" * 70)

        return self.history

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }

        if is_best:
            save_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
        else:
            save_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, save_path)

    def save_training_history(self):
        """Save training history as JSON"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  Training history saved to: {history_path}")


def main():
    """Main training function"""
    # Configuration
    COMBINATIONS_DIR = Path("C:/Users/user/Desktop/STATIONARY/Combinations")
    OUTPUT_DIR = Path("c:/Users/user/Desktop/STATIONARY/tsfresh ensemble/visual_model/trained_models")

    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224
    TRAIN_SPLIT = 0.8
    MAX_IMAGES_PER_COMBO = None  # None = use all images

    BACKBONE = 'resnet50'  # Options: 'resnet50', 'resnet34', 'efficientnet_b0'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Using device: {device}")

    # Load images
    image_paths, labels, label_names = load_combination_images(
        COMBINATIONS_DIR,
        max_images_per_combo=MAX_IMAGES_PER_COMBO
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        image_paths, labels,
        train_split=TRAIN_SPLIT,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=4
    )

    # Create model
    model = create_model(
        num_classes=len(ANOMALY_CLASSES),
        pretrained=True,
        backbone=BACKBONE,
        device=device
    )

    # Train
    trainer = AnomalyTrainer(model, device, OUTPUT_DIR)
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # Save final model info
    model_info = {
        'num_classes': len(ANOMALY_CLASSES),
        'classes': ANOMALY_CLASSES,
        'backbone': BACKBONE,
        'image_size': IMAGE_SIZE,
        'best_val_acc': trainer.best_val_acc,
        'total_images': len(image_paths),
        'train_images': len(train_loader.dataset),
        'val_images': len(val_loader.dataset)
    }

    with open(OUTPUT_DIR / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n  Model info saved to: {OUTPUT_DIR / 'model_info.json'}")


if __name__ == "__main__":
    main()
