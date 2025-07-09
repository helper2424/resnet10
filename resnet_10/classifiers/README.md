# ResNet-10 Classifiers Package

This package provides modular components for creating, training, and validating binary and multiclass classifiers using ResNet-10 encoders.

## Structure

```
resnet_10/classifiers/
├── __init__.py          # Package exports
├── heads.py             # Classifier head architectures
├── training.py          # Training functions
├── validation.py        # Validation functions
├── utils.py             # Utility functions
└── README.md           # This documentation
```

## Quick Start

```python
from resnet_10.classifiers import (
    train_binary_classifier,
    train_multiclass_classifier,
    validate_binary_classifier,
    validate_multiclass_classifier,
    one_vs_rest
)

# Load your pre-trained ResNet-10 model
model = AutoModel.from_pretrained("your_model", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# For binary classification
binary_head = train_binary_classifier(
    model, train_loader, val_loader, device,
    num_epochs=10, learning_rate=0.001
)

# For multiclass classification
multiclass_head = train_multiclass_classifier(
    model, train_loader, val_loader, device,
    num_epochs=10, learning_rate=0.001, num_classes=10
)
```

## Components

### 🧠 Classifier Heads (`heads.py`)

#### `create_binary_classifier_head(input_features=512, hidden_features=256, dropout_rate=0.1)`
Creates a binary classification head with:
- Dropout layer for regularization
- Linear layer (input_features → hidden_features)
- LayerNorm for stable training
- Tanh activation
- Final linear layer (hidden_features → 1)

#### `create_multiclass_classifier_head(input_features=512, hidden_features=256, dropout_rate=0.1, num_classes=10)`
Creates a multiclass classification head with same architecture but outputs `num_classes` logits.

### 🏋️ Training Functions (`training.py`)

#### `train_binary_classifier(model, train_loader, val_loader, device, **kwargs)`
Trains a binary classifier head on top of frozen ResNet-10 encoder.

**Parameters:**
- `model`: Pre-trained ResNet-10 model
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `device`: Device to train on
- `num_epochs=10`: Number of training epochs
- `learning_rate=0.001`: Learning rate for Adam optimizer
- `input_features=512`: Encoder output features
- `hidden_features=256`: Hidden layer size
- `dropout_rate=0.1`: Dropout rate

**Returns:** Trained binary classification head

#### `train_multiclass_classifier(model, train_loader, val_loader, device, **kwargs)`
Trains a multiclass classifier head. Same parameters as binary classifier plus:
- `num_classes=10`: Number of output classes

**Returns:** Trained multiclass classification head

### 🔍 Validation Functions (`validation.py`)

#### `validate_binary_classifier(model, classifier_head, test_loader, device)`
Evaluates binary classifier performance.

**Returns:** `(test_loss, test_accuracy)`

#### `validate_multiclass_classifier(model, classifier_head, test_loader, device)`
Evaluates multiclass classifier performance.

**Returns:** `(test_loss, test_accuracy)`

### 🛠️ Utilities (`utils.py`)

#### `one_vs_rest(dataset, target_class)`
Converts multiclass dataset to binary classification (target class vs all others).

**Parameters:**
- `dataset`: PyTorch dataset with `.targets` attribute
- `target_class`: Class ID to use as positive class

**Returns:** Modified dataset with binary labels (1.0 for target class, 0.0 for others)

## Architecture Details

### Classification Head Design
Both binary and multiclass heads use the same architecture:

```
Input Features (512)
    ↓
Dropout (rate=0.1)
    ↓
Linear (512 → 256)
    ↓
LayerNorm (256)
    ↓
Tanh Activation
    ↓
Linear (256 → output_size)
```

- **Binary**: `output_size = 1` with BCEWithLogitsLoss
- **Multiclass**: `output_size = num_classes` with CrossEntropyLoss

### Training Strategy
1. **Freeze encoder**: Only classification heads are trainable
2. **Adam optimizer**: Adaptive learning rates for stable training
3. **Validation monitoring**: Track performance on validation set each epoch
4. **Progress reporting**: Batch-level and epoch-level metrics

## Usage Examples

### Binary Classification (One vs Rest)
```python
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Load CIFAR-10 and convert to binary task
train_dataset = CIFAR10(root="data", train=True, transform=transforms.ToTensor())
test_dataset = CIFAR10(root="data", train=False, transform=transforms.ToTensor())

# Convert to binary: class 3 (cats) vs rest
train_binary = one_vs_rest(train_dataset, target_class=3)
test_binary = one_vs_rest(test_dataset, target_class=3)

train_loader = DataLoader(train_binary, batch_size=128, shuffle=True)
test_loader = DataLoader(test_binary, batch_size=128, shuffle=False)

# Train binary classifier
binary_head = train_binary_classifier(
    model, train_loader, test_loader, device,
    num_epochs=20, learning_rate=0.001
)

# Evaluate
test_loss, test_acc = validate_binary_classifier(
    model, binary_head, test_loader, device
)
print(f"Binary accuracy: {test_acc:.2f}%")
```

### Multiclass Classification
```python
# Use original CIFAR-10 (no label transformation needed)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Train multiclass classifier
multiclass_head = train_multiclass_classifier(
    model, train_loader, test_loader, device,
    num_epochs=20, learning_rate=0.001, num_classes=10
)

# Evaluate
test_loss, test_acc = validate_multiclass_classifier(
    model, multiclass_head, test_loader, device
)
print(f"Multiclass accuracy: {test_acc:.2f}%")
```

## Integration with Main Validation Script

The main `validate.py` script uses this package:

```bash
python validate.py \
    --model_name "your_resnet10_model" \
    --epochs 20 \
    --lr 0.001 \
    --target_class 3
```

This runs both binary and multiclass classification experiments automatically.

## Performance Expectations

### CIFAR-10 Baselines
- **Random Binary**: 50.00%
- **Random Multiclass**: 10.00%

### Good Performance Thresholds
- **Binary**: >60% (good), >75% (excellent)
- **Multiclass**: >30% (good), >50% (excellent)

## Dependencies

- `torch` - PyTorch framework
- `torch.nn` - Neural network modules
- `torch.optim` - Optimizers
