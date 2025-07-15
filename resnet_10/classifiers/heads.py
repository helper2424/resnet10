"""Classifier head creation functions for ResNet-10."""

import torch.nn as nn

from resnet_10.spatial_embeddings import SpatialLearnedEmbeddings


def create_classifier_head(
    input_features=512,
    hidden_features=256,
    dropout_rate=0.1,
    num_classes=1,
    use_spatial_embeddings=False,
    spatial_height=4,
    spatial_width=4,
):
    """Create a unified classifier head for binary or multiclass classification.

    Args:
        input_features: Number of input features from the encoder
        hidden_features: Number of hidden features in the intermediate layer
        dropout_rate: Dropout rate for regularization
        num_classes: Number of output classes (1 for binary, >1 for multiclass)
        use_spatial_embeddings: Whether to use spatial learned embeddings
        spatial_height: Height for spatial embeddings (when use_spatial_embeddings=True)
        spatial_width: Width for spatial embeddings (when use_spatial_embeddings=True)

    Returns:
        nn.Sequential: The classifier head model
    """
    layers = []

    # Add spatial embeddings if requested
    if use_spatial_embeddings:
        layers.append(
            SpatialLearnedEmbeddings(
                height=spatial_height,
                width=spatial_width,
                channel=input_features,
                num_features=hidden_features,
            )
        )

    # Add standard layers
    layers.extend(
        [
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=input_features, out_features=hidden_features),
            nn.LayerNorm(normalized_shape=hidden_features),
            nn.Tanh(),
            nn.Linear(in_features=hidden_features, out_features=num_classes),
        ]
    )

    return nn.Sequential(*layers)


# Convenience functions for backward compatibility and ease of use
def create_binary_classifier_head(input_features=512, hidden_features=256, dropout_rate=0.1):
    """Create a binary classification head (convenience function)."""
    return create_classifier_head(
        input_features=input_features,
        hidden_features=hidden_features,
        dropout_rate=dropout_rate,
        num_classes=1,
        use_spatial_embeddings=False,
    )


def create_multiclass_classifier_head(input_features=512, hidden_features=256, dropout_rate=0.1, num_classes=10):
    """Create a multiclass classification head (convenience function)."""
    return create_classifier_head(
        input_features=input_features,
        hidden_features=hidden_features,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        use_spatial_embeddings=False,
    )


def create_binary_classifier_with_spatial_embeddings(
    input_features=512, hidden_features=256, dropout_rate=0.1, spatial_height=4, spatial_width=4
):
    """Create a binary classification head with spatial embeddings (convenience function)."""
    return create_classifier_head(
        input_features=input_features,
        hidden_features=hidden_features,
        dropout_rate=dropout_rate,
        num_classes=1,
        use_spatial_embeddings=True,
        spatial_height=spatial_height,
        spatial_width=spatial_width,
    )


def create_multiclass_classifier_with_spatial_embeddings(
    input_features=512, hidden_features=256, dropout_rate=0.1, num_classes=10, spatial_height=4, spatial_width=4
):
    """Create a multiclass classification head with spatial embeddings (convenience function)."""
    return create_classifier_head(
        input_features=input_features,
        hidden_features=hidden_features,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        use_spatial_embeddings=True,
        spatial_height=spatial_height,
        spatial_width=spatial_width,
    )
