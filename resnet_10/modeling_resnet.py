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

import math
from typing import Optional

import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention

from .configuration_resnet import ResNet10Config


class MaxPool2dJax(nn.Module):
    """Mimics JAX's MaxPool with padding='SAME' for exact parity."""

    def __init__(self, kernel_size, stride=2):
        super().__init__()

        # Ensure kernel_size and stride are tuples
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.maxpool = nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # No padding
        )

    def _compute_padding(self, input_height, input_width):
        """Calculate asymmetric padding to match JAX's 'SAME' behavior."""

        # Compute padding needed for height and width
        pad_h = max(
            0, (math.ceil(input_height / self.stride[0]) - 1) * self.stride[0] + self.kernel_size[0] - input_height
        )
        pad_w = max(
            0, (math.ceil(input_width / self.stride[1]) - 1) * self.stride[1] + self.kernel_size[1] - input_width
        )

        # Asymmetric padding (JAX-style: more padding on the bottom/right if needed)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return (pad_left, pad_right, pad_top, pad_bottom)

    def forward(self, x):
        """Apply asymmetric padding before convolution."""
        _, _, h, w = x.shape

        # Compute asymmetric padding
        pad_left, pad_right, pad_top, pad_bottom = self._compute_padding(h, w)
        x = nn.functional.pad(
            x, (pad_left, pad_right, pad_top, pad_bottom), value=-float("inf")
        )  # Pad right/bottom by 1 to match JAX's maxpooling padding="SAME"

        return nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)


class Conv2dJax(nn.Module):
    """Mimics JAX's Conv2D with padding='SAME' for exact parity."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super().__init__()

        # Ensure kernel_size and stride are tuples
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # No padding
            bias=bias,
        )

    def _compute_padding(self, input_height, input_width):
        """Calculate asym
        metric padding to match JAX's 'SAME' behavior."""

        # Compute padding needed for height and width
        pad_h = max(
            0, (math.ceil(input_height / self.stride[0]) - 1) * self.stride[0] + self.kernel_size[0] - input_height
        )
        pad_w = max(
            0, (math.ceil(input_width / self.stride[1]) - 1) * self.stride[1] + self.kernel_size[1] - input_width
        )

        # Asymmetric padding (JAX-style: more padding on the bottom/right if needed)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return (pad_left, pad_right, pad_top, pad_bottom)

    def forward(self, x):
        """Apply asymmetric padding before convolution."""
        _, _, h, w = x.shape

        # Compute asymmetric padding
        pad_left, pad_right, pad_top, pad_bottom = self._compute_padding(h, w)
        x = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, stride=1, norm_groups=4):
        super().__init__()

        self.conv1 = Conv2dJax(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        self.act1 = ACT2FN[activation]
        self.act2 = ACT2FN[activation]
        self.conv2 = Conv2dJax(out_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv2dJax(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        return self.act2(out)


class Encoder(nn.Module):
    def __init__(self, config: ResNet10Config):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList([])

        for i, size in enumerate(self.config.hidden_sizes):
            if i == 0:
                self.stages.append(
                    BasicBlock(
                        self.config.embedding_size,
                        size,
                        activation=self.config.hidden_act,
                    )
                )
            else:
                self.stages.append(
                    BasicBlock(
                        self.config.hidden_sizes[i - 1],
                        size,
                        activation=self.config.hidden_act,
                        stride=2,
                    )
                )

    def forward(self, hidden_state: Tensor, output_hidden_states: bool = False) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class ResNet10(PreTrainedModel):
    config_class = ResNet10Config

    def __init__(self, config):
        super().__init__(config)

        self.embedder = nn.Sequential(
            nn.Conv2d(
                self.config.num_channels,
                self.config.embedding_size,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            # The original code has a small trick -
            # https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/vision/resnet_v1.py#L119
            # class MyGroupNorm(nn.GroupNorm):
            #     def __call__(self, x):
            #         if x.ndim == 3:
            #             x = x[jnp.newaxis]
            #             x = super().__call__(x)
            #             return x[0]
            #         else:
            #             return super().__call__(x)
            nn.GroupNorm(num_groups=4, eps=1e-5, num_channels=self.config.embedding_size),
            ACT2FN[self.config.hidden_act],
            MaxPool2dJax(kernel_size=3, stride=2),
        )

        self.encoder = Encoder(self.config)
        self.pooler = nn.AdaptiveAvgPool2d(output_size=1)

    def _init_pooler(self):
        if self.config.pooler == "avg":
            self.pooler = nn.AdaptiveAvgPool2d(output_size=1)
        elif self.config.pooler == "max":
            self.pooler = nn.MaxPool2d(kernel_size=3, stride=2)
        elif self.config.pooler == "spatial_learned_embeddings":
            raise ValueError("Invalid pooler, it exist in the hil serl version, but weights are missing")

            # In the original HIl-SERL code is used SpatialLearnedEmbeddings as pooliing method
            # Check https://github.com/rail-berkeley/hil-serl/blob/7d17d13560d85abffbd45facec17c4f9189c29c0/serl_launcher/serl_launcher/agents/continuous/sac.py#L490
            # But weights for this custom layer are missing
            # Probably it means that pretrained weights used other way of pooling - probably it's AvgPool2d
            # self.pooler = nn.Sequential(
            #     SpatialLearnedEmbeddings(
            #         height=height,
            #         width=width,
            #         channel=channel,
            #         num_features=self.num_spatial_blocks,
            #     ),
            #     nn.Dropout(0.1, deterministic=not train),
            # )
        else:
            raise ValueError(f"Invalid pooler: {self.config.pooler}")

    def forward(self, x: Tensor, output_hidden_states: Optional[bool] = None) -> BaseModelOutputWithNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        embedding_output = self.embedder(x)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states)

        pooler_output = self.pooler(encoder_outputs.last_hidden_state)

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            pooler_output=pooler_output,
        )

    def print_model_hash(self):
        print("Model parameters hashes:")
        for name, param in self.named_parameters():
            print(name, param.sum())
