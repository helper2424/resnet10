# Claude Context for ResNet-10 Project

## Project Overview
This is a ResNet-10 implementation project that converts JAX-based weights to PyTorch format for use with Hugging Face Transformers. The project provides tools for conversion, validation, and training of ResNet-10 models.

## Key Files & Structure
- `resnet_10/`: Main package containing the model implementation
  - `modeling_resnet.py`: PyTorch ResNet-10 model implementation
  - `configuration_resnet.py`: Model configuration
  - `spatial_embeddings.py`: Spatial embedding utilities
- `convert_jax_to_pytroch.py`: JAX to PyTorch weight conversion script
- `validate_outputs_are_same.py`: Validation script to compare JAX and PyTorch outputs
- `validate.py`: Model validation utilities
- `run_end_to_end_test.py`: End-to-end testing script

## Development Setup
- **Package Manager**: uv (UV package manager)
- **Python Dependencies**: See `requirements.txt` and `requirements-dev.txt`
- **Lock File**: `uv.lock`

## Common Commands
```bash
# Install dependencies
uv sync

# Convert JAX weights to PyTorch
uv run python ./convert_jax_to_pytroch.py --model_name helper2424/resnet10 --push_to_hub True

# Validate outputs match between JAX and PyTorch
uv run python validate_outputs_are_same.py --model_name helper2424/resnet10

# Run tests
pytest

# Run end-to-end test
uv run python run_end_to_end_test.py
```

## Testing & Validation
- Uses pytest for testing (configured in `pytest.ini`)
- Validation includes comparing JAX vs PyTorch outputs
- End-to-end testing available
- Model convergence validation for classification tasks

## Data & Weights
- `data/`: Contains CIFAR-10 dataset files
- `weights/`: Pre-trained model weights
- `resnet10_diffs/`: Visualization files showing model differences

## Hugging Face Integration
- Model available at: `helper2424/resnet10`
- Supports `AutoModel.from_pretrained()` with `trust_remote_code=True`
- Includes model upload/update scripts

## Notes
- Project uses modern Python tooling (uv, pytest, pre-commit)
- Includes comprehensive validation and testing pipeline
- Supports both local development and Hugging Face Hub integration
