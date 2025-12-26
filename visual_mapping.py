"""
Visual Model - Combination to Anomaly Label Mapping
Maps 21 combinations to their anomaly types (excluding deterministic_trend)
"""

# Mapping from combination folder name to anomaly label
VISUAL_MAPPING = {
    # CUBIC BASE (4 combinations)
    'cubic_collective_anomaly': 'collective_anomaly',
    'Cubic + Mean Shift': 'mean_shift',
    'Cubic + Point Anomaly': 'point_anomaly',
    'Cubic + Variance Shift': 'variance_shift',

    # DAMPED BASE (4 combinations)
    'Damped + Collective Anomaly': 'collective_anomaly',
    'Damped + Mean Shift': 'mean_shift',
    'Damped + Point Anomaly': 'point_anomaly',
    'Damped + Variance Shift': 'variance_shift',

    # EXPONENTIAL BASE (4 combinations)
    'exponential_collective_anomaly': 'collective_anomaly',
    'Exponential + Mean Shift': 'mean_shift',
    'exponential_point_anomaly': 'point_anomaly',
    'exponential_variance_shift': 'variance_shift',

    # LINEAR BASE (5 combinations)
    'Linear + Collective Anomaly': 'collective_anomaly',
    'Linear + Mean Shift': 'mean_shift',
    'Linear + Point Anomaly': 'point_anomaly',
    'Linear + Trend Shift': 'trend_shift',
    'Linear + Variance Shift': 'variance_shift',

    # QUADRATIC BASE (4 combinations)
    'Quadratic + Collective anomaly': 'collective_anomaly',
    'Quadratic + Mean Shift': 'mean_shift',
    'Quadratic + Point Anomaly': 'point_anomaly',
    'Quadratic + Variance Shift': 'variance_shift',
}

# 8 Anomaly classes (no deterministic_trend)
ANOMALY_CLASSES = [
    'collective_anomaly',
    'contextual_anomaly',
    'mean_shift',
    'point_anomaly',
    'stochastic_trend',
    'trend_shift',
    'variance_shift',
    'volatility'
]

# Class to index mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ANOMALY_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(ANOMALY_CLASSES)}

# Base folders in Combinations directory
BASE_FOLDERS = {
    'Cubic Base': [
        'cubic_collective_anomaly',
        'Cubic + Mean Shift',
        'Cubic + Point Anomaly',
        'Cubic + Variance Shift'
    ],
    'Damped Base': [
        'Damped + Collective Anomaly',
        'Damped + Mean Shift',
        'Damped + Point Anomaly',
        'Damped + Variance Shift'
    ],
    'Exponential Base': [
        'exponential_collective_anomaly',
        'Exponential + Mean Shift',
        'exponential_point_anomaly',
        'exponential_variance_shift'
    ],
    'Linear Base': [
        'Linear + Collective Anomaly',
        'Linear + Mean Shift',
        'Linear + Point Anomaly',
        'Linear + Trend Shift',
        'Linear + Variance Shift'
    ],
    'Quadratic Base': [
        'Quadratic + Collective anomaly',
        'Quadratic + Mean Shift',
        'Quadratic + Point Anomaly',
        'Quadratic + Variance Shift'
    ]
}


def get_anomaly_label(folder_name):
    """Get anomaly label from combination folder name"""
    if folder_name in VISUAL_MAPPING:
        return VISUAL_MAPPING[folder_name]
    else:
        raise ValueError(f"Unknown combination folder: {folder_name}")


def get_label_index(label):
    """Get index for anomaly label"""
    if label in CLASS_TO_IDX:
        return CLASS_TO_IDX[label]
    else:
        raise ValueError(f"Unknown anomaly class: {label}")


def get_anomaly_stats():
    """Get statistics about anomaly distribution in combinations"""
    from collections import Counter

    anomaly_counts = Counter(VISUAL_MAPPING.values())

    print("\n" + "=" * 70)
    print("  ANOMALY DISTRIBUTION IN 21 COMBINATIONS")
    print("=" * 70)
    print(f"\n  {'Anomaly Type':<25} {'Count':>10}")
    print("  " + "-" * 40)

    for anomaly, count in sorted(anomaly_counts.items()):
        print(f"  {anomaly:<25} {count:>10}")

    print("  " + "-" * 40)
    print(f"  {'Total Combinations':<25} {len(VISUAL_MAPPING):>10}")
    print("=" * 70)

    return anomaly_counts


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  VISUAL MODEL - COMBINATION MAPPINGS")
    print("=" * 70)

    print(f"\n  Total combinations: {len(VISUAL_MAPPING)}")
    print(f"  Anomaly classes: {len(ANOMALY_CLASSES)}")

    print("\n  Mappings:")
    print("  " + "-" * 60)
    for combo, label in VISUAL_MAPPING.items():
        print(f"  {combo:<40} -> {label}")

    print("\n")
    get_anomaly_stats()
