"""
FILENAME: TieFCAutoencoder.py
DESCRIPTION: Utility functions to tie and untie weights for fully connected auto encoder models
@author: Jian Zhong
"""

import torch
from torch import nn
import torch.nn.functional as F


# create tied linear layer
class WeightTiedLinear(nn.Module):
    def __init__(
            self, 
            src_linear: nn.Linear,
            tie_to_linear: nn.Linear
        ):
        super().__init__()
        assert src_linear.weight.size() == tie_to_linear.weight.t().size()
        self.tie_to_linear = tie_to_linear
        self.bias = nn.Parameter(src_linear.bias.clone())

    # use tie_to_linear layer weigth for foward propagation
    def forward(self, input):
        return F.linear(input, self.tie_to_linear.weight.t(), self.bias)
    
    # return weight of tied linear layer
    @property 
    def weight(self):
        return self.tie_to_linear.weight.t()
    

# tie weights for symmetrical fully-connected auto encoder network.
def tie_weight_sym_fc_autoencoder(
        encoder_model: nn.Module,
        decoder_model: nn.Module,
):
    # get all the fully connected layers
    encoder_fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in encoder_model.named_modules() if isinstance(cur_module, nn.Linear)]
    decoder_fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module}for cur_layerstr, cur_module in decoder_model.named_modules() if isinstance(cur_module, nn.Linear)]

    # validate if the autoencoder model are symmetric
    assert len(encoder_fc_layers) == len(decoder_fc_layers)

    # tie weights for corresponding layers
    nof_fc_layers = len(encoder_fc_layers)
    for i_layer in range(nof_fc_layers):
        cur_encoder_layer = encoder_fc_layers[i_layer]
        cur_decoder_layer = decoder_fc_layers[nof_fc_layers - 1 - i_layer]

        # create tied linear module
        cur_tied_decoder_layermodule = WeightTiedLinear(cur_decoder_layer["module"], cur_encoder_layer["module"])

        # update the corresponding 
        cur_decoder_indexing_substrs = cur_encoder_layer["indexing_str"].split('.')
        decoder_model.get_submodule(".".join(cur_decoder_indexing_substrs[:-1]))[int(cur_decoder_indexing_substrs[-1])] = cur_tied_decoder_layermodule

    return encoder_model, decoder_model


# untie weights for fully-connected network
def untie_weight_fc_models(
        model: nn.Module,
):
    # get all fully connected layers
    # fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in model.named_modules() if isinstance(cur_module, WeightTiedLinear)]
    fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in model.named_modules() if type(cur_module).__name__ == "WeightTiedLinear"]

    # untie weights
    nof_fc_layers = len(fc_layers)
    for i_layer in range(nof_fc_layers):
        cur_layer = fc_layers[i_layer]

        # create linear module for each tied linear layer
        cur_untied_module = nn.Linear(
            in_features = cur_layer["module"].weight.size(1),
            out_features = cur_layer["module"].weight.size(0),
            bias = cur_layer["module"].bias is None,
            device = cur_layer["module"].weight.device,
            dtype = cur_layer["module"].weight.dtype,
        )

        # update linear module weight and bias from tied linear module 
        cur_untied_module.weight = nn.Parameter(cur_layer["module"].weight.clone())
        cur_untied_module.bias = nn.Parameter(cur_layer["module"].bias.clone())

        # update corresponding layers in the model
        cur_indexing_substrs = cur_layer["indexing_str"].split('.')
        model.get_submodule(".".join(cur_indexing_substrs[:-1]))[int(cur_indexing_substrs[-1])] = cur_untied_module
    
    return model