# coding=utf-8#
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
"""ResNet model configuration"""


from transformers import PretrainedConfig


class ResNet10Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ResNetModel`]. It is used to instantiate an
    ResNet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ResNet
    [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        layer_type (`str`, *optional*, defaults to `"bottleneck"`):
            The layer to use, it can be either `"basic"` (used for smaller models, like resnet-18 or resnet-34) or
            `"bottleneck"` (used for larger models like resnet-50 and above).
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
            If `True`, the first stage will downsample the inputs using a `stride` of 2.

    Example:
    ```python
    >>> from transformers import AutoConfig, AutoModel

    >>> # Initializing a ResNet resnet-50 style configuration
    >>> configuration = AutoConfig.from_pretrained("helper2424/resnet10")

    >>> # Initializing a model (with random weights) from the resnet-50 style configuration
    >>> model = AutoModel.from_pretrained("helper2424/resnet10")

    >>> # Accessing the model configuration
    >>> model.config = configuration
    ```
    """

    model_type = "resnet10"

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[64, 128, 256, 512],
        depths=[1, 1, 1, 1],
        hidden_act="relu",
        pooler=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.hidden_act = hidden_act
        self.pooler = pooler
