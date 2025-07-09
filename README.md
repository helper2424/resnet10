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

model = AutoModel.from_pretrained("helper2424/resnet10", trust_remote_code=True)
```

### Installation

1. Install Poetry:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/helper2424/resnet-10.git
   cd resnet_10
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```


### How to convert

```bash
uv run python ./convert_jax_to_pytroch.py --model_name helper2424/resnet10 --push_to_hub True
```

### Validation

This script will download the model from the hub and validate that inference from the jax Resnet10 version is the same as the Pytorch

```bash
uv run python validate_outputs_are_same.py --model_name helper2424/resnet10
```

Also, you can run the following script to check Pytorch Resnet10 convrgenes for small classification task:

```bash
uv run python validate_convergenes.py --model_name helper2424/resnet10
```

### Update

To update source code of the model use the following script:

```bash
uv run python update_model_source.py --model_name helper2424/resnet10
```

This script will download the model from HF hub, update the

### Citation

```bibtex
@misc{resnet10,
   title = "Resnet10",
   author = "Eugene Mironov and Khalil Meftah and Adil Zouitine and Michel Aractingi and Ke Wang",
   month = jan,
   year = "2025",
   address = "Online",
   publisher = "Hugging Face",
   url = "https://huggingface.co/helper2424/resnet10",
}
```
