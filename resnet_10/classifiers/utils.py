"""Utility functions for classifier training and validation."""


def one_vs_rest(dataset, target_class):
    """Convert multi-class dataset to binary classification (one vs rest)

    Args:
        dataset: PyTorch dataset with targets attribute
        target_class: Class to use as positive class (all others become negative)

    Returns:
        Modified dataset with binary labels
    """
    new_targets = []
    for _, label in dataset:
        new_label = float(1.0) if label == target_class else float(0.0)
        new_targets.append(new_label)

    dataset.targets = new_targets  # Replace the original labels with the binary ones
    return dataset
