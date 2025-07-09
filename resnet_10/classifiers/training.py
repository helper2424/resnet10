"""Training functions for binary and multiclass classifiers."""

import torch
import torch.nn as nn
import torch.optim as optim

from .heads import create_binary_classifier_head, create_multiclass_classifier_head
from .validation import validate_binary_classifier, validate_multiclass_classifier


def train_binary_classifier(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    learning_rate=0.001,
    input_features=512,
    hidden_features=256,
    dropout_rate=0.1,
):
    """
    Train a binary classifier using the ResNet-10 encoder

    Args:
        model: Pre-trained ResNet-10 model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        input_features: Number of input features from the encoder
        hidden_features: Number of hidden features in the intermediate layer
        dropout_rate: Dropout rate for regularization
    """
    # Create binary classification head
    binary_classifier_head = create_binary_classifier_head(input_features, hidden_features, dropout_rate)
    binary_classifier_head.to(device)

    # Freeze the encoder and only train the classification head
    for param in model.parameters():
        param.requires_grad = False

    # Set up optimizer and loss function
    optimizer = optim.Adam(binary_classifier_head.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    model.eval()  # Keep encoder in eval mode

    print("Starting binary classifier training...")
    print(f"Device: {device}")
    print(f"Training for {num_epochs} epochs with learning rate {learning_rate}")

    for epoch in range(num_epochs):
        # Training phase
        binary_classifier_head.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(torch.float32).to(device)

            optimizer.zero_grad()

            # Get features from the encoder
            with torch.no_grad():
                features = model(data).pooler_output
                features = features.view(features.shape[0], -1)

            # Forward pass through classification head
            outputs = binary_classifier_head(features).squeeze()
            loss = criterion(outputs, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            predicted = torch.sigmoid(outputs) > 0.5
            total_train += target.size(0)
            correct_train += (predicted == target.bool()).sum().item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train

        # Validation phase
        val_loss, val_accuracy = validate_binary_classifier(model, binary_classifier_head, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print("-" * 60)

    return binary_classifier_head


def train_multiclass_classifier(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    learning_rate=0.001,
    input_features=512,
    hidden_features=256,
    dropout_rate=0.1,
    num_classes=10,
):
    """
    Train a multiclass classifier using the ResNet-10 encoder

    Args:
        model: Pre-trained ResNet-10 model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        input_features: Number of input features from the encoder
        hidden_features: Number of hidden features in the intermediate layer
        dropout_rate: Dropout rate for regularization
        num_classes: Number of output classes
    """
    # Create multiclass classification head
    multiclass_classifier_head = create_multiclass_classifier_head(
        input_features, hidden_features, dropout_rate, num_classes
    )
    multiclass_classifier_head.to(device)

    # Freeze the encoder and only train the classification head
    for param in model.parameters():
        param.requires_grad = False

    # Set up optimizer and loss function
    optimizer = optim.Adam(multiclass_classifier_head.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.eval()  # Keep encoder in eval mode

    print("Starting multiclass classifier training...")
    print(f"Device: {device}")
    print(f"Training for {num_epochs} epochs with learning rate {learning_rate}")
    print(f"Number of classes: {num_classes}")

    for epoch in range(num_epochs):
        # Training phase
        multiclass_classifier_head.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Get features from the encoder
            with torch.no_grad():
                features = model(data).pooler_output
                features = features.view(features.shape[0], -1)

            # Forward pass through classification head
            outputs = multiclass_classifier_head(features)
            loss = criterion(outputs, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train

        # Validation phase
        val_loss, val_accuracy = validate_multiclass_classifier(model, multiclass_classifier_head, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print("-" * 60)

    return multiclass_classifier_head
