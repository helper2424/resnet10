# ResNet 10

This project provides tools to convert JAX-based ResNet-10 weights from the [HIL-SERL](https://github.com/rail-berkeley/hil-serl) implementation to PyTorch format. The converted model is compatible with the Hugging Face Transformers library.

## Features

- Convert ResNet-10 weights from JAX to PyTorch format
- Automatic download of pretrained weights
- Integration with Hugging Face Hub
- Type-checked and well-tested codebase

## Usage

The model is available on Hugging Face Hub. You can use it as follows:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("helper2424/resnet10-imagenet-1k", trust_remote_code=True)
```

### Installation

1. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/helper2424/resnet-10.git
   cd resnet_10
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```


### How to convert

```bash
poetry run python convert_jax_to_pytorch.py --model_name helper2424/resnet10-imagenet-1k --push_to_hub True
```

### Validation

This script will download the model from the hub and validate that it works as expected.

```bash
poetry run python resnet_10/validate.py --model_name helper2424/resnet10-imagenet-1k
```
