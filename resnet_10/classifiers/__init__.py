"""
ResNet-10 Classifier Components

This package provides modular components for creating, training, and validating
binary and multiclass classifiers using ResNet-10 encoders.
"""

from .heads import (create_binary_classifier_head, create_binary_classifier_with_spatial_embeddings,
                    create_classifier_head, create_multiclass_classifier_head,
                    create_multiclass_classifier_with_spatial_embeddings)
from .training import train_binary_classifier, train_multiclass_classifier
from .utils import one_vs_rest
from .validation import validate_binary_classifier, validate_multiclass_classifier

__all__ = [
    # Classifier heads
    "create_classifier_head",
    "create_binary_classifier_head",
    "create_multiclass_classifier_head",
    "create_binary_classifier_with_spatial_embeddings",
    "create_multiclass_classifier_with_spatial_embeddings",
    # Training functions
    "train_binary_classifier",
    "train_multiclass_classifier",
    # Validation functions
    "validate_binary_classifier",
    "validate_multiclass_classifier",
    # Utilities
    "one_vs_rest",
]
