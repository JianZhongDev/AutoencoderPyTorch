"""
FILENAME: StackedLayers.py
DESCRIPTION: stacked laysers for VGG model
@author: Jian Zhong
"""

import torch
from torch import nn

# stacked 2D convolutional layer
class VGGStacked2DConv(nn.Module):
    def __init__(
            self,
            layer_descriptors = [],
        ):
        assert(isinstance(layer_descriptors, list))
        super().__init__()

        self.network = nn.Identity()

        # create list of stacked layers
        stacked_layers = []

        # iterater through each descriptor for the layers and create corresponding layers
        prev_out_channels = 1
        for i_descrip in range(len(layer_descriptors)):
            cur_descriptor = layer_descriptors[i_descrip]

            # the descriptor needs to be dict
            if not isinstance(cur_descriptor, dict):
                continue
            
            # get input or default values 
            nof_layers = cur_descriptor.get("nof_layers", 1)
            in_channels = cur_descriptor.get("in_channels", prev_out_channels)
            out_channels = cur_descriptor.get("out_channels", 1)
            kernel_size = cur_descriptor.get("kernel_size", 3)
            stride = cur_descriptor.get("stride", 1)
            padding = cur_descriptor.get("padding", 1)
            bias = cur_descriptor.get("bias", True)
            padding_mode = cur_descriptor.get("padding_mode", "zeros")
            activation = cur_descriptor.get("activation", nn.ReLU)
            
            # create layers
            cur_in_channels = in_channels
            for _ in range(nof_layers):
                stacked_layers.append(
                    nn.Conv2d(
                        in_channels = cur_in_channels,
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias = bias,
                        padding_mode = padding_mode,
                    )
                )
                stacked_layers.append(
                    activation()
                )
                cur_in_channels = out_channels
            prev_out_channels = out_channels
            
        # convert list of layers to sequential layers
        if len(stacked_layers) > 0:
            self.network = nn.Sequential(*stacked_layers)

    def forward(self, x):
        y = self.network(x)
        return y
    

# stacked linear layers
class VGGStackedLinear(nn.Module):
    def __init__(
            self,
            layer_descriptors = [],
    ):
        assert(isinstance(layer_descriptors, list))
        super().__init__()

        self.network = nn.Identity()

        # create list of stacked layers
        stacked_layers = []
        
        # iterater through each descriptor for the layers and create corresponding layers
        prev_out_features = 1
        for i_descrip in range(len(layer_descriptors)):
            cur_descriptor = layer_descriptors[i_descrip]

            # the descriptor needs to be dict
            if not isinstance(cur_descriptor, dict):
                continue            
            
            nof_layers = cur_descriptor.get("nof_layers", 1)
            in_features = cur_descriptor.get("in_features", prev_out_features)
            out_features = cur_descriptor.get("out_features", 1)
            bias = cur_descriptor.get("bias", True)
            activation = cur_descriptor.get("activation", nn.ReLU)
            dropout_p = cur_descriptor.get("dropout_p", None)

            # create layers
            cur_in_features = in_features
            for _ in range(nof_layers):
                stacked_layers.append(
                    nn.Linear(
                        in_features = cur_in_features,
                        out_features = out_features,
                        bias = bias,
                    )
                )
                if activation is not None:
                    stacked_layers.append(
                        activation()
                    )
                if dropout_p is not None:
                    stacked_layers.append(
                        nn.Dropout(p = dropout_p)
                    )
                cur_in_features = out_features
            
            prev_out_features = out_features

        # convert list of layers to sequential layers
        if len(stacked_layers) > 0:
            self.network = nn.Sequential(*stacked_layers)
    
    def forward(self, x):
        y = self.network(x)
        return y



         

