#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from transformers import AutoModel

BATCH_SIZE = 128
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def one_vs_rest(dataset, target_class):
    """Convert multi-class dataset to binary classification (one vs rest)"""
    new_targets = []
    for _, label in dataset:
        new_label = float(1.0) if label == target_class else float(0.0)
        new_targets.append(new_label)

    dataset.targets = new_targets  # Replace the original labels with the binary ones
    return dataset


def create_binary_classifier_head(input_features=512, hidden_features=256, dropout_rate=0.1):
    """Create the binary classification head layers

    Args:
        input_features: Number of input features from the encoder
        hidden_features: Number of hidden features in the intermediate layer
        dropout_rate: Dropout rate for regularization
    """
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features=input_features, out_features=hidden_features),
        nn.LayerNorm(normalized_shape=hidden_features),
        nn.Tanh(),
        nn.Linear(in_features=hidden_features, out_features=1),
    )


def train_binary_classifier(
    model,
    train_loader,
    val_loader,
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
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        input_features: Number of input features from the encoder
        hidden_features: Number of hidden features in the intermediate layer
        dropout_rate: Dropout rate for regularization
    """
    # Create binary classification head
    binary_classifier_head = create_binary_classifier_head(input_features, hidden_features, dropout_rate)
    binary_classifier_head.to(DEVICE)

    # Freeze the encoder and only train the post-processing layers
    for param in model.parameters():
        param.requires_grad = False

    # Set up optimizer and loss function
    optimizer = optim.Adam(binary_classifier_head.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    model.eval()  # Keep encoder in eval mode

    print("Starting training...")
    print(f"Device: {DEVICE}")
    print(f"Training for {num_epochs} epochs with learning rate {learning_rate}")

    for epoch in range(num_epochs):
        # Training phase
        binary_classifier_head.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(torch.float32).to(DEVICE)

            optimizer.zero_grad()

            # Get features from the encoder
            with torch.no_grad():
                features = model(data).pooler_output
                features = features.view(features.shape[0], -1)

            # Forward pass through post-processing layers
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
        train_loss / len(train_loader)
        100.0 * correct_train / total_train

    return binary_classifier_head


def validate_binary_classifier(model, post_steps, test_loader):
    """
    Validate the binary classifier

    Args:
        model: Pre-trained ResNet-10 model
        post_steps: Trained post-processing layers
        test_loader: Test data loader

    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    post_steps.eval()

    test_loss = 0.0
    test_labels = []
    test_predictions = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(torch.float32).to(DEVICE)

            # Get features from encoder
            features = model(data).last_hidden_state
            features = features.view(features.shape[0], -1)

            # Forward pass through post-processing layers
            outputs = post_steps(features).squeeze()
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


def main():
    # Validate that model works as expected
    # Let's do a binary classification task
    # And check convergence
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=("The model name to download from the hub."),
    )
    parser.add_argument("--train", action="store_true", help="Train the binary classifier instead of just validating")
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--input_features", default=512, type=int, help="Number of input features from the encoder")
    parser.add_argument(
        "--hidden_features", default=256, type=int, help="Number of hidden features in the classification head"
    )
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate for regularization")

    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(DEVICE)

    target_binary_class = 3

    # Load and prepare datasets
    binary_train_dataset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    binary_test_dataset = CIFAR10(root="data", train=False, download=True, transform=ToTensor())

    # Apply one-vs-rest labeling
    binary_train_dataset = one_vs_rest(binary_train_dataset, target_binary_class)
    binary_test_dataset = one_vs_rest(binary_test_dataset, target_binary_class)

    binary_train_loader = DataLoader(binary_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    binary_test_loader = DataLoader(binary_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train the binary classifier
    binary_classifier_head = train_binary_classifier(
        model,
        binary_train_loader,
        binary_test_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        input_features=args.input_features,
        hidden_features=args.hidden_features,
        dropout_rate=args.dropout_rate,
    )

    # Final evaluation
    print("\nFinal evaluation:")
    test_loss, test_accuracy = validate_binary_classifier(model, binary_classifier_head, binary_test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
