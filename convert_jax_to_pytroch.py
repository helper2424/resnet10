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
import os
import pickle as pkl

import requests
import torch
from huggingface_hub import ModelCard
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from transformers.image_utils import PILImageResampling

from resnet_10.configuration_resnet import ResNet10Config
from resnet_10.modeling_resnet import ResNet10


# The original code is copied from https://github.com/rail-berkeley/hil-serl/blob/7d17d13560d85abffbd45facec17c4f9189c29c0/serl_launcher/serl_launcher/utils/train_utils.py#L103
# It downloads the pretrained ResNet-10 weights for HIL-SERL implementation
# from the github release and loads them into a dictionary.
def load_resnet10_params(image_keys=("image",), public=True):
    """
    Load pretrained resnet10 params from github release to an agent.
    :return: agent with pretrained resnet10 params
    """
    file_name = "resnet10_params.pkl"
    if not public:  # if github repo is not public, load from local file
        with open(file_name, "rb") as f:
            encoder_params = pkl.load(f)
    else:  # when repo is released, download from url
        # Construct the full path to the file
        file_path = os.path.expanduser("~/.serl/")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name)
        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exist at '{file_path}'.")
        else:
            url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            print(f"Downloading file from {url}")

            # Streaming download with progress bar
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                t = tqdm(total=total_size, unit="iB", unit_scale=True)
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()
                if total_size != 0 and t.n != total_size:
                    raise Exception("Error, something went wrong with the download")
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        with open(file_path, "rb") as f:
            encoder_params = pkl.load(f)

    print("Loaded parameters from ResNet-10 pretrained on ImageNet-1K")

    return encoder_params


def apply_block_weights(block, jax_state_dict):
    block.conv1.load_state_dict(convert_jax_conv_state_dict_to_torch_conv_state_dict(jax_state_dict["Conv_0"]))
    block.conv2.load_state_dict(convert_jax_conv_state_dict_to_torch_conv_state_dict(jax_state_dict["Conv_1"]))

    block.norm1.load_state_dict(convert_jax_norm_state_dict_to_torch_norm_state_dict(jax_state_dict["MyGroupNorm_0"]))
    block.norm2.load_state_dict(convert_jax_norm_state_dict_to_torch_norm_state_dict(jax_state_dict["MyGroupNorm_1"]))

    if "conv_proj" in jax_state_dict:
        # Convolution
        block.shortcut[0].load_state_dict(
            convert_jax_conv_state_dict_to_torch_conv_state_dict(jax_state_dict["conv_proj"])
        )
        # Group Norm
        block.shortcut[1].load_state_dict(
            convert_jax_norm_state_dict_to_torch_norm_state_dict(jax_state_dict["norm_proj"])
        )


def convert_jax_conv_state_dict_to_torch_conv_state_dict(jax_state_dict):
    conv = torch.Tensor(list(jax_state_dict["kernel"].tolist())).permute(3, 2, 0, 1)
    return {"conv.weight": conv}


def convert_jax_norm_state_dict_to_torch_norm_state_dict(jax_state_dict):
    return {
        "weight": torch.Tensor(jax_state_dict["scale"].tolist()),
        "bias": torch.Tensor(jax_state_dict["bias"].tolist()),
    }


def apply_pretrained_resnet10_params(model, params):
    model.embedder[0].load_state_dict(
        {"weight": torch.Tensor(list(params["conv_init"]["kernel"].tolist())).permute(3, 2, 0, 1)}
    )

    model.embedder[1].load_state_dict(
        {
            "weight": torch.Tensor(params["norm_init"]["scale"].tolist()),
            "bias": torch.Tensor(params["norm_init"]["bias"].tolist()),
        },
        strict=True,
    )

    for i, block in enumerate(model.encoder.stages):
        apply_block_weights(block, params[f"ResNetBlock_{i}"])


def create_card_model_content(model_name):
    citation = """
@misc{resnet10,
   title = "Resnet10",
   author = "Eugene Mironov and Khalil Meftah and Adil Zouitine and Michel Aractingi and Ke Wang",
   month = jan,
   year = "2025",
   address = "Online",
   publisher = "Hugging Face",
   url = "https://huggingface.co/helper2424/resnet10",
}
"""

    return f"""
---
language: en
license: apache-2.0
tags:
  - pytorch
  - jax-conversion
  - transformers
  - resnet
  - hil-serl
  - Lerobot
  - vision
  - image-classification
library_name: pytorch
---

# JAX to PyTorch Converted Model (ResNet-10)

It's done in context of porting `HIL-SERL` paper code (https://hil-serl.github.io/) to `Lerobot` (https://github.com/Lerobot/lerobot).
The HF doesn't have ResNet-10 model, which could be pretty usefult for robotics tasks because of it's small size.
This model is converted from JAX to PyTorch, and the weights are preserved.
## Model Description

[Brief description of the original model and its purpose]

This model is a PyTorch port of the original JAX implementation. The conversion maintains
the original model's architecture and weights while making it accessible to PyTorch users.
The original model is from https://github.com/rail-berkeley/hil-serl/blob/7d17d13560d85abffbd45facec17c4f9189c29c0/serl_launcher/serl_launcher/utils/train_utils.py#L103.

## Model Details

- **Original Framework:** JAX
- **Target Framework:** PyTorch
- **Model Architecture:** [Specify architecture]
- **Original Model:** [Link to original model]
- **Parameters:** [Number of parameters]

## Conversion Process

This model was converted using an automated JAX to PyTorch conversion pipeline, ensuring:
- Weight preservation
- Architecture matching
- Numerical stability


## Code

https://github.com/helper2424/resnet10


## Usage
```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("{model_name}")
```

## Citation
```bibtex
{citation}
```
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=("The name of the model you wish to convert."),
    )
    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )

    args = parser.parse_args()

    config = ResNet10Config(
        num_channels=3,
        embedding_size=64,
        hidden_act="relu",
        hidden_sizes=[64, 128, 256, 512],  # Smaller hidden sizes for ResNet-10
        depths=[1, 1, 1, 1],  # One block per stage for ResNet-10
    )

    if args.push_to_hub:
        ResNet10Config.register_for_auto_class()
        ResNet10.register_for_auto_class()
        print("Registered for auto class")

        config.push_to_hub(args.model_name)
        print(f"Config uploaded successfully to Hugging Face Hub! {args.model_name}")

        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        print(f"Config loaded successfully from Hugging Face Hub! {args.model_name}")

    model = ResNet10(config)
    params = load_resnet10_params()
    model.train()
    apply_pretrained_resnet10_params(model, params)

    print("Before pushing model to hub:")
    model.print_model_hash()

    dummy_input_1 = torch.zeros(1, 3, 128, 128)
    dummy_input_2 = torch.ones(1, 3, 128, 128)
    dummy_input = torch.cat([dummy_input_1, dummy_input_2], dim=0)

    pred = model(dummy_input)

    print("Test for forward pass:", pred.last_hidden_state.shape)

    # Lets use MS's processor with modifications
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", trust_remote_code=True)
    processor.image_mean = [0.485, 0.456, 0.406]
    processor.image_std = [0.229, 0.224, 0.225]
    processor.do_scale = True
    processor.rescale_factor = 0.00392156862745098
    processor.do_resize = True
    processor.size = {"shortest_edge": 128}
    processor.crop_pct = 1
    processor.resample = PILImageResampling.BILINEAR

    if args.push_to_hub:
        model.push_to_hub(args.model_name)
        print(f"Model uploaded successfully to Hugging Face Hub! {args.model_name}")

        card = ModelCard(content=create_card_model_content(args.model_name))
        card.push_to_hub(args.model_name)
        print(f"Model card uploaded successfully to Hugging Face Hub! {args.model_name}")

        processor.push_to_hub(args.model_name)
        print(f"Processor uploaded successfully to Hugging Face Hub! {args.model_name}")

        loaded_model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
        print("Loaded model parameters hashes:")
        loaded_model.print_model_hash()
