# Quick Start - Visual Model

## What is this?

Visual model for classifying anomaly types directly from time series plots using CNN (ResNet50).

**Key Difference**: This model does NOT classify `deterministic_trend`. It only classifies the anomaly types.

## Currently Available Classes (4 out of 8)

Based on available plot data:
- ✅ `collective_anomaly` (68 images)
- ✅ `mean_shift` (152 images)
- ✅ `point_anomaly` (68 images)
- ✅ `variance_shift` (68 images)

Not available (no plots found):
- ❌ `contextual_anomaly` (0 images)
- ❌ `stochastic_trend` (0 images)
- ❌ `trend_shift` (0 images)
- ❌ `volatility` (0 images)

**Total**: 356 images from 21 combination folders

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow tqdm numpy
```

## Usage

### 1. Check Mappings

```bash
cd "c:\Users\user\Desktop\STATIONARY\tsfresh ensemble\visual_model"
python visual_mapping.py
```

### 2. Train Model (Simple Version - Recommended)

```bash
python train_simple.py
```

This will:
- Automatically detect which classes have images (currently 4 classes)
- Load all 356 images
- Split 80/20 train/val
- Train ResNet50 for 30 epochs
- Save best model to `trained_models/best_model.pth`

**Expected training time**:
- GPU: ~5-10 minutes
- CPU: ~30-60 minutes

**Expected accuracy**: 70-90% (4-class classification)

### 3. Monitor Training

Output will show:
```
Epoch 1/30
--------------------------------------------------
Training: 100%|████████| 9/9 [00:15<00:00,  1.72s/it]
Validating: 100%|████████| 3/3 [00:02<00:00,  1.12it/s]
Train Loss: 1.2345, Train Acc: 45.67%
Val Loss:   1.1234, Val Acc:   52.34%
>>> New best model saved! Val Acc: 52.34%
```

## Output Files

After training, check `trained_models/`:

1. **best_model.pth**: Trained model checkpoint
2. **model_info.json**: Model metadata
   ```json
   {
     "num_classes": 4,
     "active_classes": ["collective_anomaly", "mean_shift", "point_anomaly", "variance_shift"],
     "best_val_acc": 0.85,
     "total_images": 356
   }
   ```
3. **training_history.json**: Loss/accuracy curves

## Troubleshooting

### Error: "No module named 'torch'"

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "CUDA out of memory"

Edit [train_simple.py](train_simple.py:173) and reduce batch size:
```python
BATCH_SIZE = 16  # or even 8
```

### Training very slow on CPU

This is normal. ResNet50 is heavy. Options:
1. Use GPU if available
2. Reduce epochs to 10-15 for quick test
3. Switch to smaller backbone (edit model.py)

### Low accuracy (<60%)

Possible causes:
1. **Small dataset**: Only 356 images total, ~70 per class on average
2. **Class imbalance**: mean_shift has 152 images, others have ~68
3. **Plot style variations**: Different combination folders may have different plot styles

Solutions:
- Increase epochs to 50
- Add more augmentation
- Collect more plot images

## Comparison: Visual Model vs TSFresh Ensemble

| Metric | TSFresh Ensemble | Visual Model |
|--------|------------------|--------------|
| **Input** | Raw time series (CSV) | Plot images (PNG) |
| **Feature extraction** | 1.5s (TSFresh) | 0s (just load image) |
| **Inference speed** | ~1.8s per sample | ~0.01s per sample |
| **Accuracy** | 90%+ (9-class) | 70-90% (4-class, limited data) |
| **Classes** | 9 classes | 4 classes (limited by available plots) |
| **Interpretability** | High (feature importances) | Low (black box CNN) |
| **Use case** | Production, analysis | Quick screening, real-time |

## Next Steps

1. **Evaluate model**: Create `evaluate.py` to test on held-out data
2. **Visualize predictions**: Plot confusion matrix
3. **Generate more plots**: Create plots for the 4 missing classes
4. **Deploy**: Use for real-time classification on new plots

## File Structure

```
visual_model/
├── QUICK_START.md           # This file
├── README.md                # Full documentation
├── visual_mapping.py        # Combination → anomaly mappings
├── data_loader.py           # Image loading and preprocessing
├── model.py                 # CNN architecture
├── train.py                 # Full training script (advanced)
├── train_simple.py          # Simple training script (recommended)
├── config.py                # Configuration file
└── trained_models/          # Model outputs
    ├── best_model.pth
    ├── model_info.json
    └── training_history.json
```

## Advanced Configuration

Edit hyperparameters in [train_simple.py](train_simple.py:169-174):

```python
NUM_EPOCHS = 30           # Number of training epochs
BATCH_SIZE = 32           # Batch size
LEARNING_RATE = 0.001     # Learning rate
IMAGE_SIZE = 224          # Input image size
TRAIN_SPLIT = 0.8         # Train/val split
```
