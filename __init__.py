"""
Visual Model - Image-Based Anomaly Classification
CNN-based classifier for 8 anomaly types (excluding deterministic_trend)
"""

from .visual_mapping import (
    VISUAL_MAPPING,
    ANOMALY_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    get_anomaly_label,
    get_label_index
)

from .model import AnomalyClassifierCNN, create_model

__all__ = [
    'VISUAL_MAPPING',
    'ANOMALY_CLASSES',
    'CLASS_TO_IDX',
    'IDX_TO_CLASS',
    'get_anomaly_label',
    'get_label_index',
    'AnomalyClassifierCNN',
    'create_model'
]
