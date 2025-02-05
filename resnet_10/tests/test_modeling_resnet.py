import pytest
import torch

from resnet_10.configuration_resnet import ResNet10Config
from resnet_10.modeling_resnet import ResNet10


@pytest.fixture
def config():
    return ResNet10Config(
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[64, 128, 256, 512],
        hidden_act="relu",
        output_hidden_states=False,
    )


@pytest.fixture
def model(config):
    return ResNet10(config)


@pytest.fixture
def input_data():
    batch_size = 2
    input_size = 224
    return torch.randn(batch_size, 3, input_size, input_size)


def test_model_initialization(model, config):
    """Test if model initializes correctly with given config"""
    assert isinstance(model, ResNet10)
    assert model.config == config


def test_forward_pass_shape(model, input_data):
    """Test if forward pass produces correct output shape"""
    output = model(input_data)

    # After initial conv and maxpool: 224 -> 112 -> 56
    # After 4 stages with stride=2: 56 -> 28 -> 14 -> 7
    expected_size = 7
    batch_size = input_data.shape[0]

    assert output.last_hidden_state.shape == (batch_size, 512, expected_size, expected_size)


def test_hidden_states_output(model, input_data):
    """Test if hidden states are returned when requested"""
    output = model(input_data, output_hidden_states=True)

    # Should have hidden states from:
    # 1. After embedder (before encoder)
    # 2. After each encoder stage (4 stages)
    # Total: 5 hidden states
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 5


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_different_batch_sizes(model, batch_size):
    """Test if model handles different batch sizes"""
    input_size = 224
    x = torch.randn(batch_size, 3, input_size, input_size)
    output = model(x)
    assert output.last_hidden_state.shape[0] == batch_size


def test_model_parameters_not_none(model):
    """Test if all model parameters are initialized (not None)"""
    for name, param in model.named_parameters():
        assert param is not None
        assert param.data is not None
        assert not torch.isnan(param.data).any()


def test_input_channel_validation(model):
    """Test if model properly handles incorrect number of input channels"""
    wrong_channels = 4  # Config expects 3
    batch_size = 2
    input_size = 224
    x = torch.randn(batch_size, wrong_channels, input_size, input_size)

    with pytest.raises(RuntimeError):
        _ = model(x)


@pytest.mark.parametrize("input_size", [160, 224, 256])
def test_different_input_sizes(model, input_size):
    """Test if model handles different input sizes"""
    batch_size = 2
    x = torch.randn(batch_size, 3, input_size, input_size)
    output = model(x)
    expected_size = input_size // 32  # Due to 5 downsampling operations
    assert output.last_hidden_state.shape[-1] == expected_size


def test_pooler_output(model, input_data):
    """Test if pooler output is correct"""
    output = model(input_data)
    assert output.pooler_output.shape == (input_data.shape[0], 512, 1, 1)
