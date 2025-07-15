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
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from transformers import AutoModel

# Import classifier components
from resnet_10.classifiers import (
    one_vs_rest,
    train_binary_classifier,
    train_multiclass_classifier,
    validate_binary_classifier,
    validate_multiclass_classifier,
)

BATCH_SIZE = 128
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


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
    parser.add_argument("--epochs", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--input_features", default=512, type=int, help="Number of input features from the encoder")
    parser.add_argument(
        "--hidden_features", default=256, type=int, help="Number of hidden features in the classification head"
    )
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate for regularization")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of classes for multiclass classification")
    parser.add_argument(
        "--target_class", default=3, type=int, help="Target class for binary classification (one vs rest)"
    )

    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(DEVICE)

    print("=" * 80)
    print("COMPREHENSIVE RESNET-10 ENCODER VALIDATION")
    print("=" * 80)
    print("Running complete evaluation pipeline:")
    print("1. Binary Classification Training & Validation")
    print("2. Multiclass Classification Training & Validation")
    print("=" * 80)

    # =========================================================================
    # PHASE 1: BINARY CLASSIFICATION
    # =========================================================================
    print("\n" + "=" * 50)
    print("PHASE 1: BINARY CLASSIFICATION")
    print("=" * 50)
    print(f"Task: Class {args.target_class} vs rest")

    # Load and prepare binary datasets
    binary_train_dataset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    binary_test_dataset = CIFAR10(root="data", train=False, download=True, transform=ToTensor())

    # Apply one-vs-rest labeling
    binary_train_dataset = one_vs_rest(binary_train_dataset, args.target_class)
    binary_test_dataset = one_vs_rest(binary_test_dataset, args.target_class)

    binary_train_loader = DataLoader(binary_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    binary_test_loader = DataLoader(binary_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train binary classifier
    print("\n🚀 Training Binary Classifier...")
    binary_classifier_head = train_binary_classifier(
        model,
        binary_train_loader,
        binary_test_loader,
        DEVICE,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        input_features=args.input_features,
        hidden_features=args.hidden_features,
        dropout_rate=args.dropout_rate,
    )

    # Validate binary classifier
    print("\n🔍 Final Binary Classification Evaluation:")
    binary_test_loss, binary_test_accuracy = validate_binary_classifier(
        model, binary_classifier_head, binary_test_loader, DEVICE
    )
    print(f"📊 Binary Test Loss: {binary_test_loss:.4f}")
    print(f"📊 Binary Test Accuracy: {binary_test_accuracy:.2f}%")
    print("📊 Binary Random Baseline: 50.00%")

    # =========================================================================
    # PHASE 2: MULTICLASS CLASSIFICATION
    # =========================================================================
    print("\n" + "=" * 50)
    print("PHASE 2: MULTICLASS CLASSIFICATION")
    print("=" * 50)
    print(f"Task: All {args.num_classes} CIFAR-10 classes")

    # Load multiclass datasets (no label transformation needed)
    multiclass_train_dataset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    multiclass_test_dataset = CIFAR10(root="data", train=False, download=True, transform=ToTensor())

    multiclass_train_loader = DataLoader(multiclass_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    multiclass_test_loader = DataLoader(multiclass_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train multiclass classifier
    print("\n🚀 Training Multiclass Classifier...")
    multiclass_classifier_head = train_multiclass_classifier(
        model,
        multiclass_train_loader,
        multiclass_test_loader,
        DEVICE,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        input_features=args.input_features,
        hidden_features=args.hidden_features,
        dropout_rate=args.dropout_rate,
        num_classes=args.num_classes,
    )

    # Validate multiclass classifier
    print("\n🔍 Final Multiclass Classification Evaluation:")
    multiclass_test_loss, multiclass_test_accuracy = validate_multiclass_classifier(
        model, multiclass_classifier_head, multiclass_test_loader, DEVICE
    )
    print(f"📊 Multiclass Test Loss: {multiclass_test_loss:.4f}")
    print(f"📊 Multiclass Test Accuracy: {multiclass_test_accuracy:.2f}%")
    print(f"📊 Multiclass Random Baseline: {100.0/args.num_classes:.2f}%")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 50)
    print("📋 FINAL EVALUATION SUMMARY")
    print("=" * 50)
    print("CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
    print(f"Target class for binary: {args.target_class}")
    print("")
    print("📊 RESULTS:")
    print(f"   Binary Classification:    {binary_test_accuracy:.2f}% (baseline: 50.00%)")
    print(f"   Multiclass Classification: {multiclass_test_accuracy:.2f}% (baseline: {100.0/args.num_classes:.2f}%)")
    print("")

    # Performance analysis
    binary_improvement = binary_test_accuracy - 50.0
    multiclass_improvement = multiclass_test_accuracy - (100.0 / args.num_classes)

    print("🎯 PERFORMANCE ANALYSIS:")
    print(f"   Binary improvement over random: +{binary_improvement:.2f}%")
    print(f"   Multiclass improvement over random: +{multiclass_improvement:.2f}%")

    if binary_test_accuracy > 85 and multiclass_test_accuracy > 70:
        print("✅ EXCELLENT: Your ResNet-10 encoder shows strong feature learning!")
    elif binary_test_accuracy > 75 and multiclass_test_accuracy > 50:
        print("✅ GOOD: Your ResNet-10 encoder is learning meaningful representations!")
    elif binary_test_accuracy > 60 and multiclass_test_accuracy > 30:
        print("⚠️  MODERATE: Your ResNet-10 encoder shows some learning but could be improved!")
    else:
        print("❌ POOR: Your ResNet-10 encoder may need more training or architectural changes!")

    print("=" * 50)


if __name__ == "__main__":
    main()
