"""
Comprehensive test of visual model on ALL images from ResNet50 dataset
Tests the 100% validation accuracy model on full train+test sets
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architecture and mapping
from model import AnomalyClassifierCNN
from visual_mapping import VISUAL_MAPPING


class VisualModelTester:
    def __init__(self, model_path, model_info_path, images_root):
        """
        Args:
            model_path: Path to best_model.pth
            model_info_path: Path to model_info.json
            images_root: Root directory containing train/test folders
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)

        self.classes = self.model_info['active_classes']
        self.class_to_idx = self.model_info['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Load model
        self.model = AnomalyClassifierCNN(num_classes=len(self.classes))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.images_root = Path(images_root)

    def extract_label_from_path(self, image_path):
        """Extract ground truth label using VISUAL_MAPPING
        Path format: .../Base Type/Combination Name/**/plots/image.png
        """
        path_parts = image_path.parts

        # Find the combination folder (direct child of Base Type)
        # Look backwards from 'plots' folder
        for i, part in enumerate(reversed(path_parts)):
            if part == 'plots':
                # Found plots folder, now check all ancestors
                plots_index = len(path_parts) - i - 1

                # Check all parent directories from plots upward
                for parent_level in range(1, plots_index):
                    parent_name = path_parts[plots_index - parent_level]
                    if parent_name in VISUAL_MAPPING:
                        return VISUAL_MAPPING[parent_name]

        return None

    def predict_image(self, image_path):
        """Predict single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = self.idx_to_class[predicted_idx.item()]
            confidence_value = confidence.item()

            return predicted_class, confidence_value, probabilities.cpu().numpy()[0]

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, 0.0, None

    def collect_all_images(self):
        """Collect all images from Combinations folder structure
        Structure: Combinations/Base Type/Base + Anomaly/**/plots/*.png
        """
        all_images = []

        # Iterate through base trend folders
        for base_folder in self.images_root.iterdir():
            if not base_folder.is_dir():
                continue

            # Iterate through combination folders (e.g., "Linear + Collective Anomaly")
            for combo_folder in base_folder.iterdir():
                if not combo_folder.is_dir():
                    continue

                # Look for all plots subfolders (including nested ones)
                plots_folders = list(combo_folder.rglob('plots'))

                for plots_folder in plots_folders:
                    if not plots_folder.is_dir():
                        continue

                    # Collect all PNG images
                    for image_path in plots_folder.glob('*.png'):
                        all_images.append({
                            'path': image_path,
                            'split': 'all',  # No train/test split in Combinations
                            'folder': combo_folder.name
                        })

        return all_images

    def run_comprehensive_test(self):
        """Run test on all images and generate detailed report"""
        print("=" * 80)
        print("COMPREHENSIVE VISUAL MODEL TEST")
        print("=" * 80)
        print(f"Model: {self.model_info['best_val_acc']*100:.2f}% validation accuracy")
        print(f"Classes: {', '.join(self.classes)}")
        print(f"Device: {self.device}")
        print()

        # Collect all images
        print("Collecting images...")
        all_images = self.collect_all_images()
        print(f"Found {len(all_images)} images")
        print()

        # Run predictions
        results = []
        y_true = []
        y_pred = []
        confidences = []

        print("Running predictions...")
        for i, img_data in enumerate(all_images):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(all_images)} images...")

            image_path = img_data['path']
            true_label = self.extract_label_from_path(image_path)

            if true_label is None:
                print(f"Warning: Could not extract label from {image_path}")
                continue

            pred_label, confidence, probs = self.predict_image(image_path)

            if pred_label is None:
                continue

            results.append({
                'image': image_path.name,
                'split': img_data['split'],
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'correct': true_label == pred_label,
                'probabilities': probs.tolist() if probs is not None else None
            })

            y_true.append(self.class_to_idx[true_label])
            y_pred.append(self.class_to_idx[pred_label])
            confidences.append(confidence)

        print(f"Completed {len(results)} predictions")
        print()

        # Calculate metrics
        return self.generate_report(results, y_true, y_pred, confidences)

    def generate_report(self, results, y_true, y_pred, confidences):
        """Generate comprehensive report with metrics and visualizations"""

        # Overall metrics
        overall_acc = accuracy_score(y_true, y_pred)

        print("=" * 80)
        print("OVERALL RESULTS")
        print("=" * 80)
        print(f"Total Images: {len(results)}")
        print(f"Overall Accuracy: {overall_acc*100:.2f}%")
        print(f"Correct Predictions: {sum(r['correct'] for r in results)}")
        print(f"Incorrect Predictions: {sum(not r['correct'] for r in results)}")
        print(f"Average Confidence: {np.mean(confidences):.4f}")
        print()

        # Split-wise performance
        print("SPLIT-WISE PERFORMANCE")
        print("-" * 80)
        for split in ['train', 'test']:
            split_results = [r for r in results if r['split'] == split]
            if split_results:
                split_acc = sum(r['correct'] for r in split_results) / len(split_results)
                split_conf = np.mean([r['confidence'] for r in split_results])
                print(f"{split.upper()}: {len(split_results)} images, "
                      f"Accuracy: {split_acc*100:.2f}%, "
                      f"Avg Confidence: {split_conf:.4f}")
        print()

        # Per-class performance
        print("PER-CLASS PERFORMANCE")
        print("-" * 80)

        # Get unique labels present in the data
        unique_labels = sorted(set(y_true + y_pred))
        unique_class_names = [self.idx_to_class[idx] for idx in unique_labels]

        class_report = classification_report(
            y_true, y_pred,
            labels=unique_labels,
            target_names=unique_class_names,
            digits=4,
            zero_division=0
        )
        print(class_report)
        print()

        # Confusion matrix (only for classes present in data)
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        print("CONFUSION MATRIX")
        print("-" * 80)
        print("      ", "  ".join(f"{c[:8]:>8}" for c in unique_class_names))
        for i, row in enumerate(cm):
            print(f"{unique_class_names[i][:8]:>8}", "  ".join(f"{val:>8}" for val in row))
        print()

        # Calculate per-class accuracy
        print("PER-CLASS ACCURACY")
        print("-" * 80)
        for i, class_name in enumerate(unique_class_names):
            class_total = np.sum(cm[i, :])
            class_correct = cm[i, i]
            class_acc = class_correct / class_total if class_total > 0 else 0
            print(f"{class_name:20} {class_correct:4}/{class_total:4} = {class_acc*100:6.2f}%")
        print()

        # Misclassification analysis
        errors = [r for r in results if not r['correct']]
        if errors:
            print("MISCLASSIFICATION ANALYSIS")
            print("-" * 80)
            print(f"Total Errors: {len(errors)}")
            print()

            # Most common confusion pairs
            from collections import Counter
            confusion_pairs = Counter([(r['true_label'], r['pred_label']) for r in errors])
            print("Top 10 Confusion Pairs:")
            for (true_label, pred_label), count in confusion_pairs.most_common(10):
                print(f"  {true_label:20} -> {pred_label:20} : {count:4} times")
            print()

            # Low confidence errors
            low_conf_errors = [r for r in errors if r['confidence'] < 0.7]
            print(f"Low Confidence Errors (<0.7): {len(low_conf_errors)}/{len(errors)}")
            if low_conf_errors:
                avg_low_conf = np.mean([r['confidence'] for r in low_conf_errors])
                print(f"Average confidence: {avg_low_conf:.4f}")
            print()

        # Save results
        output = {
            'summary': {
                'total_images': len(results),
                'overall_accuracy': overall_acc,
                'correct_predictions': sum(r['correct'] for r in results),
                'incorrect_predictions': sum(not r['correct'] for r in results),
                'average_confidence': float(np.mean(confidences))
            },
            'split_performance': {},
            'confusion_matrix': cm.tolist(),
            'class_report': classification_report(
                y_true, y_pred,
                labels=unique_labels,
                target_names=unique_class_names,
                output_dict=True,
                zero_division=0
            ),
            'detailed_results': results[:100]  # First 100 for space
        }

        for split in ['train', 'test']:
            split_results = [r for r in results if r['split'] == split]
            if split_results:
                output['split_performance'][split] = {
                    'count': len(split_results),
                    'accuracy': sum(r['correct'] for r in split_results) / len(split_results),
                    'avg_confidence': float(np.mean([r['confidence'] for r in split_results]))
                }

        # Save to JSON
        output_path = Path('results/comprehensive_test_results.json')
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {output_path}")

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, output_path.parent / 'confusion_matrix_full.png', unique_class_names)

        return output

    def plot_confusion_matrix(self, cm, save_path, class_names):
        """Plot and save confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix - Combinations Dataset Test')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to: {save_path}")


if __name__ == '__main__':
    # Paths
    MODEL_PATH = 'trained_models/best_model.pth'
    MODEL_INFO_PATH = 'trained_models/model_info.json'
    IMAGES_ROOT = r'C:\Users\user\Desktop\STATIONARY\Combinations'

    # Run test
    tester = VisualModelTester(MODEL_PATH, MODEL_INFO_PATH, IMAGES_ROOT)
    results = tester.run_comprehensive_test()

    print()
    print("=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
