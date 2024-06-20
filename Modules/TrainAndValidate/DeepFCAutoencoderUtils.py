"""
FILENAME: DeepFCAutoencoderUtils.py
DESCRIPTION: Utility functions to for deep autoencoder training.
@author: Jian Zhong
"""

import torch
from torch import nn


# get references of all the layer reference of a specific class
def get_layer_refs(
    model: nn.Module,
    layer_class,
):
    # get all the fully connected layers
    layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in model.named_modules() if isinstance(cur_module, layer_class)]

    return layers


# update states of dst layers from src layers
def update_corresponding_layers(
    src_layer_refs,
    dst_layer_refs,
):
    nof_src_layers = len(src_layer_refs)
    nof_dst_layers = len(dst_layer_refs)s

    nof_itr_layers = min(len(nof_src_layers), len(nof_dst_layers))
    for i_layer in nof_itr_layers:
        cur_src_module = src_layer_refs[i_layer]["module"]
        cur_dst_module = dst_layer_refs[i_layer]["module"]
        cur_dst_module.load_state_dict(cur_src_module.state_dict())

    return nof_src_layers, nof_dst_layers


# freeze (disable grad calculation) all the layers in the input layer reference list
def freeze_layers(
    layer_refs,
):
    for cur_layer in layer_refs:
        for param in cur_layer.parameters():
            param.requires_grad = False

    return layer_refs


# unfreeze (enable grad calculation) all the layers in the input layer reference list
def unfreeze_layers(
    layer_refs,    
):
    for cur_layer in layer_refs:
        for param in cur_layer.parameters():
            param.requires_grad = True

    return layer_refs