"""Validation functions for binary and multiclass classifiers."""

import torch
import torch.nn as nn


def validate_binary_classifier(model, binary_classifier_head, test_loader, device):
    """
    Validate the binary classifier

    Args:
        model: Pre-trained ResNet-10 model
        binary_classifier_head: Trained binary classification head
        test_loader: Test data loader
        device: Device to run validation on

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    binary_classifier_head.eval()

    test_loss = 0.0
    test_labels = []
    test_predictions = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(torch.float32).to(device)

            # Get features from encoder
            features = model(data).pooler_output
            features = features.view(features.shape[0], -1)

            # Forward pass through classification head
            outputs = binary_classifier_head(features).squeeze()
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Convert to probabilities and predictions
            probs = torch.sigmoid(outputs)

            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(probs.cpu().numpy())

    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    test_predictions_binary = [1 if pred > 0.5 else 0 for pred in test_predictions]
    accuracy = sum([1 if pred == label else 0 for pred, label in zip(test_predictions_binary, test_labels)]) / len(
        test_labels
    )

    return avg_test_loss, accuracy * 100


def validate_multiclass_classifier(model, multiclass_classifier_head, test_loader, device):
    """
    Validate the multiclass classifier

    Args:
        model: Pre-trained ResNet-10 model
        multiclass_classifier_head: Trained multiclass classification head
        test_loader: Test data loader
        device: Device to run validation on

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    multiclass_classifier_head.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # Get features from encoder
            features = model(data).pooler_output
            features = features.view(features.shape[0], -1)

            # Forward pass through classification head
            outputs = multiclass_classifier_head(features)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_test_loss, accuracy
