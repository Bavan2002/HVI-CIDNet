import os
import torch
from collections import OrderedDict
from torch import nn as nn
from torchvision.models import vgg as vgg


# Registry pattern for registering modules by name
# Allows dynamic lookup: ARCH_REGISTRY.get('VGGFeatureExtractor')
class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert name not in self._obj_map, (
            f"An object named '{name}' was already registered "
            f"in '{self._name}' registry!"
        )
        self._obj_map[name] = obj

    def register(self, obj=None):
        # Can be used as decorator: @REGISTRY.register() or function: REGISTRY.register(cls)
        if obj is None:

            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
            )
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


# Global registries for different module types
DATASET_REGISTRY = Registry("dataset")
ARCH_REGISTRY = Registry("arch")
MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")

VGG_PRETRAIN_PATH = "experiments/pretrained_models/vgg19-dcbb9e9d.pth"

# VGG layer names for different architectures
# Used to map layer indices to semantic names like 'conv3_4', 'relu5_1'
NAMES = {
    "vgg11": [
        "conv1_1",
        "relu1_1",
        "pool1",
        "conv2_1",
        "relu2_1",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "pool5",
    ],
    "vgg13": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "pool5",
    ],
    "vgg16": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "pool5",
    ],
    "vgg19": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "conv3_4",
        "relu3_4",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "conv4_4",
        "relu4_4",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "conv5_4",
        "relu5_4",
        "pool5",
    ],
}


def insert_bn(names):
    # Insert batch norm layer name after each conv layer (for vgg_bn variants)
    names_bn = []
    for name in names:
        names_bn.append(name)
        if "conv" in name:
            position = name.replace("conv", "")
            names_bn.append("bn" + position)
    return names_bn


@ARCH_REGISTRY.register()
class VGGFeatureExtractor(nn.Module):
    """
    VGG network for extracting intermediate features for perceptual loss.
    Frozen by default (no gradient updates).
    """

    def __init__(
        self,
        layer_name_list,  # which layers to extract, e.g. ['conv3_4', 'conv4_4']
        vgg_type="vgg19",
        use_input_norm=True,  # normalize with ImageNet mean/std
        range_norm=False,  # convert [-1,1] to [0,1]
        requires_grad=False,  # freeze VGG weights
        remove_pooling=False,
        pooling_stride=2,
    ):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace("_bn", "")]
        if "bn" in vgg_type:
            self.names = insert_bn(self.names)

        # Find the last layer we need - only load up to that point
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        # Load pretrained VGG (local or torchvision)
        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(
                VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage
            )
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)

        features = vgg_net.features[: max_idx + 1]  # truncate to needed layers

        # Build named sequential with optional pooling modification
        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if "pool" in k:
                if remove_pooling:
                    continue
                else:
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net).cuda()

        # Freeze or enable training
        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        # ImageNet normalization stats (for images in [0,1] range)
        if self.use_input_norm:
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            )
            self.register_buffer(
                "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            )

    def forward(self, x):
        # Normalize input if needed
        if self.range_norm:
            x = (x + 1) / 2  # [-1,1] -> [0,1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std  # ImageNet normalization

        # Extract features at specified layers
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()  # store feature map

        return output  # dict: layer_name -> feature_tensor
