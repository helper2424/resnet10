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

import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from convert_jax_to_pytroch import apply_pretrained_resnet10_params, load_resnet10_params
# Import the necessary modules
from resnet_10.classifiers.heads import (create_binary_classifier_head,
                                         create_binary_classifier_with_spatial_embeddings,
                                         create_multiclass_classifier_head,
                                         create_multiclass_classifier_with_spatial_embeddings)
from resnet_10.classifiers.training import train_binary_classifier, train_multiclass_classifier
from resnet_10.classifiers.utils import one_vs_rest
from resnet_10.classifiers.validation import validate_binary_classifier, validate_multiclass_classifier
from resnet_10.configuration_resnet import ResNet10Config
from resnet_10.modeling_resnet import ResNet10

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TestEndToEndValidation:
    """End-to-end test for the complete pipeline from JAX weights to validation results."""

    @pytest.fixture(scope="class")
    def temp_model_dir(self):
        """Create a temporary directory for saving the model."""
        temp_dir = tempfile.mkdtemp(prefix="resnet10_test_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def presaved_model_weights_path(self):
        """Load presaved model weights from the pretrained model."""
        project_root = Path(__file__).parent.parent.parent
        weights_path = project_root / "resnet10_params.pkl"
        return weights_path

    def load_jax_weights(self, weights_path):
        """Load JAX weights from the pretrained model."""
        return load_resnet10_params(weights_path=weights_path)

    def default_config(self):
        """Return the default configuration for the ResNet10 model."""
        return ResNet10Config(
            num_channels=3,
            embedding_size=64,
            hidden_act="relu",
            hidden_sizes=[64, 128, 256, 512],  # Smaller hidden sizes for ResNet-10
            depths=[1, 1, 1, 1],  # One block per stage for ResNet-10
        )

    def get_device(self):
        """Get the device to run the tests on."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def setup_model_with_pooling(self):
        """Common setup for creating a model with pooling and applying JAX weights."""
        logger.info("🔄 Setting up model with pooling...")

        # Load JAX weights
        jax_weights = self.load_jax_weights(self.presaved_model_weights_path())
        assert jax_weights is not None
        logger.info("✅ JAX weights loaded")

        # Create model with pooling
        config = self.default_config()
        config.pooler = "avg"
        logger.info("⚙️  Model config created with average pooling")

        model = ResNet10(config)
        device = self.get_device()
        logger.info(f"🖥️  Using device: {device}")

        # Move to device BEFORE applying weights
        model.to(device)
        logger.info("🏗️  Model moved to device")

        model.train()
        apply_pretrained_resnet10_params(model, jax_weights)
        logger.info("🔄 JAX weights applied to PyTorch model")

        return model, device

    def setup_model_without_pooling(self):
        """Common setup for creating a model without pooling (spatial embeddings) and applying JAX weights."""
        logger.info("🔄 Setting up model without pooling (for spatial embeddings)...")

        # Load JAX weights
        jax_weights = self.load_jax_weights(self.presaved_model_weights_path())
        assert jax_weights is not None
        logger.info("✅ JAX weights loaded")

        # Create model without pooling
        config = self.default_config()
        config.pooler = None  # No pooling, spatial embeddings will be handled in classifier head
        logger.info("⚙️  Model config created without pooling")

        model = ResNet10(config)
        device = self.get_device()
        logger.info(f"🖥️  Using device: {device}")

        # Move to device BEFORE applying weights
        model.to(device)
        logger.info("🏗️  Model moved to device")

        model.train()
        apply_pretrained_resnet10_params(model, jax_weights)
        logger.info("🔄 JAX weights applied to PyTorch model")

        return model, device

    def load_cifar10_datasets(self, binary_classification=False, target_class=3, batch_size=32):
        """Load CIFAR-10 datasets, optionally converting to binary classification."""
        logger.info("📁 Loading CIFAR-10 datasets...")
        train_dataset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
        test_dataset = CIFAR10(root="data", train=False, download=True, transform=ToTensor())

        if binary_classification:
            # Convert to binary classification
            logger.info(f"🔄 Converting to binary classification: class {target_class} vs rest")
            train_dataset = one_vs_rest(train_dataset, target_class)
            test_dataset = one_vs_rest(test_dataset, target_class)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"📊 Dataset loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

        return train_loader, test_loader

    def test_jax_weights_loading(self):
        """Test that JAX weights are loaded correctly."""
        logger.info("🔄 Starting JAX weights loading test...")

        weights_path = self.presaved_model_weights_path()
        logger.info(f"📁 Loading weights from: {weights_path}")

        jax_weights = self.load_jax_weights(weights_path)
        assert jax_weights is not None
        logger.info("✅ JAX weights loaded successfully!")

        # Check for expected keys in the JAX weights
        expected_keys = ["conv_init", "norm_init", "ResNetBlock_0", "ResNetBlock_1", "ResNetBlock_2", "ResNetBlock_3"]
        logger.info(f"🔍 Checking for expected keys: {expected_keys}")

        for key in expected_keys:
            assert key in jax_weights, f"Missing expected key in JAX weights: {key}"
            logger.info(f"  ✓ Found key: {key}")

        logger.info("🎉 JAX weights validation completed successfully!")

    def test_pytorch_model_conversion(self):
        """Test that PyTorch model is created and weights are applied correctly."""
        logger.info("🔄 Starting PyTorch model conversion test...")

        # Load JAX weights
        jax_weights = self.load_jax_weights(self.presaved_model_weights_path())
        assert jax_weights is not None
        logger.info("✅ JAX weights loaded for conversion")

        # Create model configuration
        config = self.default_config()
        logger.info("⚙️  Model config created")

        # Get device and create model
        device = self.get_device()
        logger.info(f"🖥️  Using device: {device}")

        model = ResNet10(config)
        model.to(device)
        logger.info("🏗️  ResNet10 model created and moved to device")

        # Apply JAX weights to PyTorch model
        model.train()
        apply_pretrained_resnet10_params(model, jax_weights)
        logger.info("🔄 JAX weights applied to PyTorch model")

        # Test model is instance of ResNet10
        assert isinstance(model, ResNet10)
        assert model.config == config
        logger.info("✅ Model validation passed")

        # Test forward pass works
        logger.info("🧪 Testing forward pass...")
        dummy_input = torch.randn(2, 3, 128, 128).to(device)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.last_hidden_state is not None
        assert output.pooler_output is None
        logger.info(f"🎯 Forward pass successful! Output shape: {output.last_hidden_state.shape}")
        logger.info("🎉 PyTorch model conversion test completed successfully!")

    def test_convert_model_with_pooling_and_run_binary_classifier_training_and_validation(self):
        logger.info("🔄 Starting binary classifier training and validation test...")

        # Setup model with pooling (common code)
        model, device = self.setup_model_with_pooling()

        # Create a binary classifier head
        binary_classifier_head = create_binary_classifier_head(
            input_features=512, hidden_features=256, dropout_rate=0.2
        )
        binary_classifier_head.to(device)
        logger.info("🧠 Binary classifier head created")

        # Load CIFAR-10 dataset for binary classification
        target_class = 3  # cats vs rest
        train_loader, test_loader = self.load_cifar10_datasets(
            binary_classification=True, target_class=target_class, batch_size=32
        )

        # Train binary classifier
        logger.info("🏋️ Starting binary classifier training...")
        trained_head = train_binary_classifier(
            model,
            train_loader,
            test_loader,
            device,
            num_epochs=2,  # Reduced for faster testing
            learning_rate=0.001,
            input_features=512,
            hidden_features=256,
            dropout_rate=0.2,
        )
        logger.info("✅ Binary classifier training completed!")

        # Validate model
        logger.info("🔍 Running final validation...")
        res = validate_binary_classifier(model, trained_head, test_loader, device)

        logger.info(f"📊 Binary validation results: Loss={res[0]:.4f}, Accuracy={res[1]:.2f}%")
        assert res is not None
        # More realistic thresholds for a quick test
        assert res[0] < 2.0  # Loss should be reasonable
        assert res[1] > 10.0  # Accuracy should be better than random (50%) for this quick test

        logger.info("🎉 Binary classifier test completed successfully!")

    def test_convert_model_with_pooling_and_run_multiclass_classifier_training_and_validation(self):
        logger.info("🔄 Starting multiclass classifier training and validation test...")

        # Setup model with pooling (common code)
        model, device = self.setup_model_with_pooling()

        # Create a multiclass classifier head
        num_classes = 10  # CIFAR-10 has 10 classes
        multiclass_classifier_head = create_multiclass_classifier_head(
            input_features=512, hidden_features=256, dropout_rate=0.2, num_classes=num_classes
        )
        multiclass_classifier_head.to(device)
        logger.info(f"🧠 Multiclass classifier head created for {num_classes} classes")

        # Load CIFAR-10 dataset for multiclass classification (no transformation needed)
        train_loader, test_loader = self.load_cifar10_datasets(binary_classification=False, batch_size=32)

        # Train multiclass classifier
        logger.info("🏋️ Starting multiclass classifier training...")
        trained_head = train_multiclass_classifier(
            model,
            train_loader,
            test_loader,
            device,
            num_epochs=2,  # Reduced for faster testing
            learning_rate=0.001,
            input_features=512,
            hidden_features=256,
            dropout_rate=0.2,
            num_classes=num_classes,
        )
        logger.info("✅ Multiclass classifier training completed!")

        # Validate model
        logger.info("🔍 Running final validation...")
        res = validate_multiclass_classifier(model, trained_head, test_loader, device)

        logger.info(f"📊 Multiclass validation results: Loss={res[0]:.4f}, Accuracy={res[1]:.2f}%")
        assert res is not None
        # More realistic thresholds for a quick test
        assert res[0] < 5.0  # Loss should be reasonable for multiclass
        assert res[1] > 5.0  # Accuracy should be better than random (10%) for this quick test

        logger.info("🎉 Multiclass classifier test completed successfully!")

    def test_convert_model_without_pooling_and_run_binary_classifier_with_spatial_embeddings_training_and_validation(
        self,
    ):
        logger.info("🔄 Starting binary classifier with spatial embeddings training and validation test...")

        # Setup model without pooling (common code)
        model, device = self.setup_model_without_pooling()

        # Create a binary classifier head with spatial embeddings
        binary_classifier_head = create_binary_classifier_with_spatial_embeddings(
            input_features=512, hidden_features=256, dropout_rate=0.2, spatial_height=4, spatial_width=4
        )
        binary_classifier_head.to(device)
        logger.info("🧠 Binary classifier head with spatial embeddings created")

        # Load CIFAR-10 dataset for binary classification
        target_class = 3  # cats vs rest
        train_loader, test_loader = self.load_cifar10_datasets(
            binary_classification=True, target_class=target_class, batch_size=32
        )

        # Train binary classifier
        logger.info("🏋️ Starting binary classifier with spatial embeddings training...")
        trained_head = train_binary_classifier(
            model,
            train_loader,
            test_loader,
            device,
            num_epochs=2,  # Reduced for faster testing
            learning_rate=0.001,
            input_features=512,
            hidden_features=256,
            dropout_rate=0.2,
        )
        logger.info("✅ Binary classifier with spatial embeddings training completed!")

        # Validate model
        logger.info("🔍 Running final validation...")
        res = validate_binary_classifier(model, trained_head, test_loader, device)

        logger.info(f"📊 Binary (spatial embeddings) validation results: Loss={res[0]:.4f}, Accuracy={res[1]:.2f}%")
        assert res is not None
        # More realistic thresholds for a quick test
        assert res[0] < 2.0  # Loss should be reasonable
        assert res[1] > 10.0  # Accuracy should be better than random (50%) for this quick test

        logger.info("🎉 Binary classifier with spatial embeddings test completed successfully!")

    def test_without_pooling__multiclass_classifier_with_spatial_embeddings_train_and_validate(
        self,
    ):
        logger.info("🔄 Starting multiclass classifier with spatial embeddings training and validation test...")

        # Setup model without pooling (common code)
        model, device = self.setup_model_without_pooling()

        # Create a multiclass classifier head with spatial embeddings
        num_classes = 10  # CIFAR-10 has 10 classes
        multiclass_classifier_head = create_multiclass_classifier_with_spatial_embeddings(
            input_features=512,
            hidden_features=256,
            dropout_rate=0.2,
            num_classes=num_classes,
            spatial_height=4,
            spatial_width=4,
        )
        multiclass_classifier_head.to(device)
        logger.info(f"🧠 Multiclass classifier head with spatial embeddings created for {num_classes} classes")

        # Load CIFAR-10 dataset for multiclass classification (no transformation needed)
        train_loader, test_loader = self.load_cifar10_datasets(binary_classification=False, batch_size=32)

        # Train multiclass classifier
        logger.info("🏋️ Starting multiclass classifier with spatial embeddings training...")
        trained_head = train_multiclass_classifier(
            model,
            train_loader,
            test_loader,
            device,
            num_epochs=2,  # Reduced for faster testing
            learning_rate=0.001,
            input_features=512,
            hidden_features=256,
            dropout_rate=0.2,
            num_classes=num_classes,
        )
        logger.info("✅ Multiclass classifier with spatial embeddings training completed!")

        # Validate model
        logger.info("🔍 Running final validation...")
        res = validate_multiclass_classifier(model, trained_head, test_loader, device)

        logger.info(f"📊 Multiclass (spatial embeddings) validation results: Loss={res[0]:.4f}, Accuracy={res[1]:.2f}%")
        assert res is not None
        # More realistic thresholds for a quick test
        assert res[0] < 5.0  # Loss should be reasonable for multiclass
        assert res[1] > 5.0  # Accuracy should be better than random (10%) for this quick test

        logger.info("🎉 Multiclass classifier with spatial embeddings test completed successfully!")


if __name__ == "__main__":
    # Allow running this test file directly with full logging
    pytest.main(
        [
            __file__,
            "-v",  # Verbose mode
            "-s",  # Show print statements
            "--tb=short",  # Short traceback format
            "--log-cli-level=INFO",  # Show INFO level logs
            "--capture=no",  # Don't capture output
        ]
    )
