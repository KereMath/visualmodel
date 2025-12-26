# Visual Model - Image-Based Anomaly Classification

## Overview

This module trains a CNN-based classifier to identify **8 anomaly types** from time series plots, **excluding deterministic_trend classification**.

## Anomaly Classes (8 Total)

1. `collective_anomaly`
2. `contextual_anomaly`
3. `mean_shift`
4. `point_anomaly`
5. `stochastic_trend`
6. `trend_shift`
7. `variance_shift`
8. `volatility`

**Note**: `deterministic_trend` is NOT classified by this model.

## Dataset

- **Source**: Plot images from `C:/Users/user/Desktop/STATIONARY/Combinations/`
- **21 Combinations** mapped to 8 anomaly types
- Images are automatically loaded from `plots/` folders within each combination

### Combination Mapping

```
Cubic Base:
  - cubic_collective_anomaly → collective_anomaly
  - Cubic + Mean Shift → mean_shift
  - Cubic + Point Anomaly → point_anomaly
  - Cubic + Variance Shift → variance_shift

Damped Base:
  - Damped + Collective Anomaly → collective_anomaly
  - Damped + Mean Shift → mean_shift
  - Damped + Point Anomaly → point_anomaly
  - Damped + Variance Shift → variance_shift

(... and so on for all 21 combinations)
```

## File Structure

```
visual_model/
├── README.md                    # This file
├── visual_mapping.py            # Combination → anomaly mappings
├── data_loader.py               # Image loading and preprocessing
├── model.py                     # CNN architecture (ResNet-based)
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script (to be created)
└── trained_models/              # Saved models and checkpoints
    ├── best_model.pth           # Best model checkpoint
    ├── model_info.json          # Model metadata
    └── training_history.json    # Training metrics
```

## Quick Start

### 1. Check Mappings

```bash
cd "c:\Users\user\Desktop\STATIONARY\tsfresh ensemble\visual_model"
python visual_mapping.py
```

**Output**: Shows all 21 combinations and their anomaly labels.

### 2. Test Data Loading

```bash
python data_loader.py
```

**Output**: Loads and counts images from all combinations.

### 3. Test Model Architecture

```bash
python model.py
```

**Output**: Creates model and tests forward pass.

### 4. Train Model

```bash
python train.py
```

**What happens**:
1. Loads all plot images from Combinations folder
2. Splits into train/val (80/20 by default)
3. Trains ResNet50-based classifier for 30 epochs
4. Saves best model based on validation accuracy

**Training Configuration** (edit in [train.py](train.py:276-284)):
```python
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
TRAIN_SPLIT = 0.8
BACKBONE = 'resnet50'  # Options: 'resnet50', 'resnet34', 'efficientnet_b0'
```

## Model Architecture

### Backbone Options

1. **ResNet50** (default) - 25M parameters, best accuracy
2. **ResNet34** - 21M parameters, faster training
3. **EfficientNet-B0** - 5M parameters, most efficient

### Classifier Head

```
Backbone (pretrained) → Dropout(0.3) → Linear(512) → ReLU → BatchNorm
                      → Dropout(0.3) → Linear(256) → ReLU → BatchNorm
                      → Dropout(0.2) → Linear(8)
```

### Data Augmentation (Training Only)

- Random horizontal flip (p=0.3)
- Random rotation (±5°)
- Color jitter (brightness & contrast ±20%)
- Normalization (ImageNet mean/std)

## Expected Performance

### Baseline Expectations

- **Validation Accuracy**: 70-85% (depends on image quality and dataset balance)
- **Training Time**: ~10-20 minutes (GPU), ~1-2 hours (CPU)

### Per-Class Performance

Some anomalies may be easier to classify visually:
- **Easy**: `point_anomaly` (clear spikes), `mean_shift` (level changes)
- **Medium**: `variance_shift`, `collective_anomaly`
- **Harder**: `contextual_anomaly`, `volatility`, `trend_shift`

## Troubleshooting

### Error: "CUDA out of memory"

Reduce batch size in [train.py](train.py:277):
```python
BATCH_SIZE = 16  # or 8
```

### Error: "No module named 'torch'"

Install PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Low Validation Accuracy (<60%)

1. **Check class imbalance**: Run `python visual_mapping.py` to see distribution
2. **Increase epochs**: Try 50-100 epochs
3. **Adjust learning rate**: Try 0.0001 or 0.0005
4. **Use larger backbone**: Switch to ResNet50 if using ResNet34

### Training too slow on CPU

1. Use smaller backbone: `BACKBONE = 'efficientnet_b0'`
2. Reduce batch size: `BATCH_SIZE = 16`
3. Limit images per combo: `MAX_IMAGES_PER_COMBO = 50`

## Next Steps

After training, you can:

1. **Evaluate on test set**: Create `evaluate.py` to test on held-out combinations
2. **Visualize predictions**: Plot confusion matrix to see which classes are confused
3. **Compare with TSFresh**: Compare accuracy with TSFresh ensemble model
4. **Deploy**: Use trained model for real-time classification on new plots

## Advantages vs TSFresh Ensemble

✅ **Much faster inference** (~0.01s vs 1.5s per sample)
✅ **No feature extraction needed** (just load image)
✅ **Learns visual patterns directly** (may capture patterns TSFresh misses)
✅ **Easier to debug** (can visualize what model sees)

## Disadvantages vs TSFresh Ensemble

❌ **Requires plots** (can't work on raw time series)
❌ **Less interpretable** (can't see which features matter)
❌ **May overfit to plot style** (grid, colors, etc.)

## Configuration Reference

All main parameters are in [train.py](train.py:276-284):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_EPOCHS` | 30 | Number of training epochs |
| `BATCH_SIZE` | 32 | Batch size for training |
| `LEARNING_RATE` | 0.001 | Initial learning rate (Adam optimizer) |
| `IMAGE_SIZE` | 224 | Input image size (224x224) |
| `TRAIN_SPLIT` | 0.8 | Train/val split ratio |
| `BACKBONE` | 'resnet50' | CNN backbone architecture |
| `MAX_IMAGES_PER_COMBO` | None | Limit images per combination (None = all) |

## Output Files

After training, check `trained_models/`:

1. **best_model.pth**: Best model checkpoint (use for inference)
2. **model_info.json**: Model metadata and hyperparameters
3. **training_history.json**: Loss and accuracy curves for all epochs
4. **checkpoint_epoch_X.pth**: Checkpoints saved every 5 epochs

## Citation

This visual model complements the TSFresh ensemble by providing:
- Direct image-based classification
- Faster inference for real-time applications
- Alternative approach when TSFresh features are insufficient
# visualmodel
