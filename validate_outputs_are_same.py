import functools as ft
import pickle
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from flax.core import freeze, unfreeze
from transformers import AutoImageProcessor, AutoModel

ModuleDef = Any

jax_hidden_states = {}
torch_hidden_states = {}


class AddSpatialCoordinates(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        grid = jnp.array(
            np.stack(
                np.meshgrid(*[np.arange(s) / (s - 1) * 2 - 1 for s in x.shape[-3:-1]]),
                axis=-1,
            ),
            dtype=self.dtype,
        ).transpose(1, 0, 2)

        if x.ndim == 4:
            grid = jnp.broadcast_to(grid, [x.shape[0], *grid.shape])

        return jnp.concatenate([x, grid], axis=-1)


class SpatialSoftmax(nn.Module):
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, features):
        if self.temperature == -1:
            from jax.nn import initializers

            temperature = self.param("softmax_temperature", initializers.ones, (1), jnp.float32)
        else:
            temperature = 1.0

        # add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        assert len(features.shape) == 4
        batch_size, num_featuremaps = features.shape[0], features.shape[3]
        features = features.transpose(0, 3, 1, 2).reshape(batch_size, num_featuremaps, self.height * self.width)

        softmax_attention = nn.softmax(features / temperature)
        expected_x = jnp.sum(self.pos_x * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_y = jnp.sum(self.pos_y * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        expected_xy = jnp.reshape(expected_xy, [batch_size, 2 * num_featuremaps])

        if no_batch_dim:
            expected_xy = expected_xy[0]
        return expected_xy


class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable = nn.initializers.lecun_normal()
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        """
        features is B x H x W X C
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.height, self.width, self.channel, self.num_features),
            self.param_dtype,
        )

        # add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        batch_size = features.shape[0]
        assert len(features.shape) == 4
        features = jnp.sum(jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2))
        features = jnp.reshape(features, [batch_size, -1])

        if no_batch_dim:
            features = features[0]

        return features


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1), self.strides, name="conv_proj")(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    norm: str = "group"
    add_spatial_coordinates: bool = False
    pooling_method: str = "avg"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    use_multiplicative_cond: bool = False
    num_spatial_blocks: int = 8
    use_film: bool = False
    bottleneck_dim: Optional[int] = None
    pre_pooling: bool = True
    image_size: tuple = (128, 128)

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        train: bool = True,
        cond_var=None,
        stop_gradient=False,
    ):
        global jax_hidden_states
        # put inputs in [-1, 1]
        # x = observations.astype(jnp.float32) / 127.5 - 1.0
        # if observations.shape[-3:-1] != self.image_size:
        #     observations = resize(observations, self.image_size)

        # imagenet mean and std # TODO: add this back
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (observations.astype(jnp.float32) / 255.0 - mean) / std

        jax_hidden_states["preprocessing"] = x

        if self.add_spatial_coordinates:
            x = AddSpatialCoordinates(dtype=self.dtype)(x)

        conv = partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.norm == "batch":
            raise NotImplementedError
        elif self.norm == "group":
            norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        elif self.norm == "layer":
            norm = partial(
                nn.LayerNorm,
                epsilon=1e-5,
                dtype=self.dtype,
            )
        else:
            raise ValueError("norm not found")

        act = getattr(nn, self.act)

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)

        jax_hidden_states["conv_init"] = x

        x = norm(name="norm_init")(x)
        jax_hidden_states["norm_init"] = x
        x = act(x)
        jax_hidden_states["act_init"] = x
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        jax_hidden_states["max_pool_init"] = x
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)
                jax_hidden_states[f"ResNetBlock_{i}"] = x
                if self.use_multiplicative_cond:
                    assert cond_var is not None, "Cond var is None, nothing to condition on"
                    cond_out = nn.Dense(x.shape[-1], kernel_init=nn.initializers.xavier_normal())(cond_var)
                    x_mult = jnp.expand_dims(jnp.expand_dims(cond_out, 1), 1)
                    x = x * x_mult
        if self.pre_pooling:
            return jax.lax.stop_gradient(x)
            # return x

        if self.pooling_method == "spatial_learned_embeddings":
            height, width, channel = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)
        elif self.pooling_method == "spatial_softmax":
            height, width, channel = x.shape[-3:]
            pos_x, pos_y = jnp.meshgrid(jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width))
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            x = SpatialSoftmax(height, width, channel, pos_x, pos_y, self.softmax_temperature)(x)
        elif self.pooling_method == "avg":
            x = jnp.mean(x, axis=(-3, -2))
        elif self.pooling_method == "max":
            x = jnp.max(x, axis=(-3, -2))
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError("pooling method not found")

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x


class PreTrainedResNetEncoder(nn.Module):
    pooling_method: str = "avg"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    num_spatial_blocks: int = 8
    bottleneck_dim: Optional[int] = None
    pretrained_encoder: nn.module = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        encode: bool = True,
        train: bool = True,
    ):
        x = observations
        if encode:
            x = self.pretrained_encoder(x, train=train)

        if self.pooling_method == "spatial_learned_embeddings":
            height, width, channel = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)
        elif self.pooling_method == "spatial_softmax":
            height, width, channel = x.shape[-3:]
            pos_x, pos_y = jnp.meshgrid(jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width))
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            x = SpatialSoftmax(height, width, channel, pos_x, pos_y, self.softmax_temperature)(x)
        elif self.pooling_method == "avg":
            x = jnp.mean(x, axis=(-3, -2))
        elif self.pooling_method == "max":
            x = jnp.max(x, axis=(-3, -2))
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError("pooling method not found")

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x


resnetv1_configs = {
    "resnetv1-10": ft.partial(ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock),
    "resnetv1-10-frozen": ft.partial(ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock, pre_pooling=True),
}


def load_pretrain_model_jax(filepath="weights/resnet10_params.pkl"):
    """Load pretrained model parameters from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def initialize_and_load_weights():
    # Create model
    model = resnetv1_configs["resnetv1-10-frozen"]()

    # Create dummy input (batch_size=1, height=128, width=128, channels=3)
    dummy_input = jnp.ones((1, 128, 128, 3), dtype=jnp.float32)

    # Initialize model to get parameter structure
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_input, train=False)
    init_params = variables["params"]

    # Load pretrained params
    pretrained_params = load_pretrain_model_jax()

    # Merge pretrained params with initialized params
    new_params = unfreeze(init_params)
    for k, v in pretrained_params.items():
        if k in new_params:
            new_params[k] = v
            print(f"Loaded key {k}")
        else:
            print(f"Skipping key {k} (not in model)")

    # Freeze again for Flax
    new_params = freeze(new_params)

    return model, new_params


def jax_to_torch(x):
    return torch.from_numpy(np.array(x)).permute(
        0, 3, 1, 2
    )  # JAX NHWC and PyTorch NCHW => (0, 3, 1, 2). The order of HW should stay the same


def visualize_tensor(tensor, title=None, save_path=None):
    """
    Visualize a tensor as an image.

    Args:
        tensor: torch.Tensor of shape (C, H, W) or (H, W)
        title: Optional title for the plot
        save_path: Optional path to save the image
    """
    # Convert to numpy and handle different tensor shapes
    if tensor.dim() == 3:  # (C, H, W)
        if tensor.shape[0] == 1:  # Single channel
            img = tensor.squeeze(0)
        else:  # Multiple channels (assume RGB)
            img = tensor.permute(1, 2, 0)  # Convert to (H, W, C)
    else:  # Already (H, W)
        img = tensor

    # Convert to numpy and ensure proper range
    img = img.detach().cpu().numpy()

    # Normalize if needed
    if img.max() > 1.0 or img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min())

    # Plot
    plt.figure(figsize=(8, 8))
    if len(img.shape) == 2:  # Grayscale
        plt.imshow(img, cmap="gray")
    else:  # RGB
        plt.imshow(img)

    if title:
        plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.show()


# 3. Batch visualization
def visualize_batches(batches, cols=2, rows=3, title="Batch Visualization"):
    """Visualize a batch of images."""

    # Calculate grid size
    # fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    # fig.suptitle(title)

    # batch = batches[0]

    # for rw in range(rows):
    #     for col in range(cols):
    #         item = batch[col][rw]
    #         item = row.view(1, row.shape[0], row.shape[1])

    #         row.squeeze(0).numpy()
    #         axes[rw, col].imshow(item, cmap="gray")

    #         # axes[row, col].axis('off')

    # plt.tight_layout()
    # # plt.colorbar()
    # plt.show()


# 4. Using torchvision's make_grid
def visualize_with_grid(batch, nrow=8, title="Grid Visualization"):
    """Visualize images in a grid using torchvision."""
    grid = T.make_grid(batch, nrow=nrow, normalize=True, padding=2)
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.show()


# Example with feature maps
def visualize_feature_maps(feature_maps, max_features=16):
    pass


if __name__ == "__main__":
    jax_model = resnetv1_configs["resnetv1-10-frozen"]()

    jax_model, new_params = initialize_and_load_weights()

    # Test with real input (batch_size=2)
    real_input_1 = jnp.zeros((1, 128, 128, 3), dtype=jnp.float32)
    real_input_2 = jnp.ones((1, 128, 128, 3), dtype=jnp.float32)

    real_input = jnp.concatenate([real_input_1], axis=0)

    # Run inference
    outputs = jax_model.apply({"params": new_params}, real_input, train=False)

    print("Model output shape:", outputs.shape)

    processor = AutoImageProcessor.from_pretrained("lilkm/resnet10")
    model = AutoModel.from_pretrained("lilkm/resnet10", trust_remote_code=True)

    dummy_input_1 = torch.zeros(1, 3, 128, 128)
    dummy_input_2 = torch.ones(1, 3, 128, 128)
    dummy_input = torch.cat([dummy_input_1], dim=0)

    processsed_input = processor(dummy_input, return_tensors="pt")
    processsed_input = processsed_input["pixel_values"]

    torch_hidden_states = {}
    torch_hidden_states["preprocessing"] = processsed_input
    torch_hidden_states["conv_init"] = model.embedder[0](processsed_input)
    torch_hidden_states["norm_init"] = model.embedder[1](torch_hidden_states["conv_init"])
    torch_hidden_states["act_init"] = model.embedder[2](torch_hidden_states["norm_init"])
    torch_hidden_states["max_pool_init"] = model.embedder[3](torch_hidden_states["act_init"])

    torch_hidden_states["ResNetBlock_0"] = model.encoder.stages[0](torch_hidden_states["max_pool_init"])
    torch_hidden_states["ResNetBlock_1"] = model.encoder.stages[1](torch_hidden_states["ResNetBlock_0"])
    torch_hidden_states["ResNetBlock_2"] = model.encoder.stages[2](torch_hidden_states["ResNetBlock_1"])
    torch_hidden_states["ResNetBlock_3"] = model.encoder.stages[3](torch_hidden_states["ResNetBlock_2"])

    torch_model_output = model(processsed_input, output_hidden_states=True)
    pred = torch_model_output.last_hidden_state

    # In order to compare the outputs, we need to convert the PyTorch output to JAX
    # We have to permute the channel dimension of torch array because jax is channel-last

    outputs = jax_to_torch(outputs)

    assert outputs.shape == pred.shape
    # Compare the outputs
    # assert np.allclose(outputs.detach().numpy(), pred.detach().numpy(), atol=1e-6)

    for k, v in torch_hidden_states.items():
        print(k)
        jax_tensor = jax_to_torch(jax_hidden_states[k])
        print(jax_tensor.shape)
        print(v.shape)
        print(f"{k} mean diff: {np.mean(np.abs(jax_tensor.detach().numpy() - v.detach().numpy()))}")
        print(f"{k} max diff: {np.max(np.abs(jax_tensor.detach().numpy() - v.detach().numpy()))}")
        # assert jax_tensor.shape == v.shape
        # assert np.allclose(jax_tensor.detach().numpy(), v.detach().numpy(), atol=1e-7)
