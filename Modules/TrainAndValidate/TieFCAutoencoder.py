"""
FILENAME: TieFCAutoencoder.py
DESCRIPTION: Utility functions to tie and untie weights for fully connected auto encoder models
@author: Jian Zhong
"""

import torch
from torch import nn


# tie weights for symmetrical fully-connected auto encoder network.
def tie_weight_sym_fc_autoencoder(
        encoder_model: nn.Module,
        decoder_model: nn.Module,
):
    # get all the fully connected layers
    encoder_fc_layers = [cur_layer for cur_layer in encoder_model.modules() if isinstance(cur_layer, nn.Linear)]
    decoder_fc_layers = [cur_layer for cur_layer in decoder_model.modules() if isinstance(cur_layer, nn.Linear)]

    # validate if the autoencoder model are symmetric
    assert len(encoder_fc_layers) == len(decoder_fc_layers)

    # tie weights for corresponding layers
    nof_fc_layers = len(encoder_fc_layers)
    for i_layer in range(nof_fc_layers):
        cur_encoder_layer = encoder_fc_layers[i_layer]
        cur_decoder_layer = decoder_fc_layers[nof_fc_layers - 1 - i_layer]
        assert cur_decoder_layer.weight.size() == cur_encoder_layer.weight.T.size()
        cur_decoder_layer.weight = torch.nn.Parameter(cur_encoder_layer.weight.T)

    return encoder_model, decoder_model


# untie weights for fully-connected network
def untie_weight_fc_models(
        model: nn.Module,
):
    # get all fully connected layers
    fc_layers = [cur_layer for cur_layer in model.modules() if isinstance(cur_layer, nn.Linear)]

    # untie weights
    nof_fc_layers = len(fc_layers)
    for i_layer in range(nof_fc_layers):
        cur_layer = fc_layers[i_layer]
        cur_layer.weight = torch.nn.Parameter(cur_layer.weight.clone())
    
    return model