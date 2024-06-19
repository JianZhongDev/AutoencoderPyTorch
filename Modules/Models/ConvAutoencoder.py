"""
FILENAME: ConvAutoencoder.py
DESCRIPTION: Definition network for convolutional autoencoder network.
@author: Jian Zhong
"""

import torch
from torch import nn

from .Layers import StackedLayers


# simple convolutional encoder = stacked convolutional network + maxpooling 
class SimpleCovEncoder(nn.Module):
    def __init__(
            self,
            convlayer_descriptors = [],
            maxpoollayer_descriptor = {},
    ):
        assert(isinstance(convlayer_descriptors, list))
        assert(isinstance(maxpoollayer_descriptor, dict))

        super().__init__()

        self.network = nn.Identity()

        network_layers = []

        # append stacked convolution layer
        network_layers.append(StackedLayers.VGGStacked2DConv(convlayer_descriptors))

        # append maxpooling layer
        network_layers.append(
            nn.MaxPool2d(
                kernel_size = maxpoollayer_descriptor.get("kernel_size", 2),
                stride = maxpoollayer_descriptor.get("stride", 2),
                padding = maxpoollayer_descriptor.get("padding", 0),
                dilation = maxpoollayer_descriptor.get("dilation", 1),
            )
        )

        # flatten output feature space
        network_layers.append(nn.Flatten(start_dim = 1, end_dim = -1))

        if len(network_layers) > 0:
            self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        y = self.network(x)
        return y
    

# fully connected network with reshape layer in the end
class SimpleFCReshapeNetwork(nn.Module):
    def __init__(
            self,
            layer_descriptors = [],
            output_size = (-1,)
    ):
        assert(isinstance(layer_descriptors, list))

        super().__init__()

        self.network = nn.Identity()

        network_layers = []

        # add stacked fully connected layer
        network_layers.append(StackedLayers.VGGStackedLinear(layer_descriptors))
        
        # reshape output
        network_layers.append(nn.Unflatten(dim = -1, unflattened_size = output_size))

        if len(network_layers) > 0:
            self.network = nn.Sequential(*network_layers)

    def foward(self, x):
        y = self.network(x)
        return y
