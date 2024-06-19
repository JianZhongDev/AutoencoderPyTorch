"""
FILENAME: FCAutoencoder.py
DESCRIPTION: Definition for fully connected autoencoder network.
@author: Jian Zhong
"""

import torch
from torch import nn

from .Layers import (DebugLayers, SparseLayers, StackedLayers)


# fully connected network with only fully connected layers 
class SimpleFCNetwork(nn.Module):
    def __init__(
        self,
        layer_descriptors = [],
    ):
        assert isinstance(layer_descriptors, list)
        super().__init__()
        self.network = StackedLayers.VGGStackedLinear(layer_descriptors)
    
    def forward(self, x):
        y = self.network(x)
        return y
    

# fully connected network with sparsity layer
class SimpleSparseFCNetwork(nn.Module):
    def __init__(
        self,
        layer_descriptors = [],
        feature_sparsity_topk = None,
        lifetime_sparsity_topk = None,
    ):
        assert isinstance(layer_descriptors, list)

        super().__init__()

        self.network = nn.Identity()

        network_layers = []

        # add stacked fully connected layers
        network_layers.append(StackedLayers.VGGStackedLinear(layer_descriptors))

        # add top k sparsity along the feature dimension
        if feature_sparsity_topk is not None :
            network_layers.append(SparseLayers.TopKSparsity(feature_sparsity_topk))

        # add top k sparsity along the sample(batch) dimension
        if lifetime_sparsity_topk is not None:
            network_layers.append(SparseLayers.LifetimeTopkSparsity(lifetime_sparsity_topk))

        if len(network_layers) > 0:
            self.network = nn.Sequential(*network_layers)


    def forward(self, x):
        y = self.network(x)
        return y
    