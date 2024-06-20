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
        skip_no_grad_layer = False,
):
    # get all the fully connected layers
    encoder_fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in encoder_model.named_modules() if isinstance(cur_module, nn.Linear)]
    decoder_fc_layers = [{"indexing_str": cur_layerstr, "module": cur_module} for cur_layerstr, cur_module in decoder_model.named_modules() if isinstance(cur_module, nn.Linear)]

    # validate if the autoencoder model are symmetric
    assert len(encoder_fc_layers) == len(decoder_fc_layers)

    # tie weights for corresponding layers
    nof_fc_layers = len(encoder_fc_layers)
    for i_layer in range(nof_fc_layers):
        cur_encoder_layer = encoder_fc_layers[i_layer]
        cur_decoder_layer = decoder_fc_layers[nof_fc_layers - 1 - i_layer]

        # skip freezed (no grad) layers if needed
        if skip_no_grad_layer:
            if not cur_decoder_layer["module"].weight.requires_grad:
                continue
            if not cur_decoder_layer["module"].weight.requires_grad:
                continue

        # create tied linear module
        cur_tied_decoder_layermodule = WeightTiedLinear(cur_decoder_layer["module"], cur_encoder_layer["module"])

        # update the corresponding layers
        cur_decoder_indexing_substrs = cur_decoder_layer["indexing_str"].split('.')
        cur_nof_substrs = len(cur_decoder_indexing_substrs)

        cur_substr_slow_idx = 0
        cur_substr_fast_idx = 0

        # iterative access corresponding layers
        cur_model = decoder_model
        while(cur_substr_fast_idx < cur_nof_substrs):
            if cur_decoder_indexing_substrs[cur_substr_fast_idx].isdigit():
                if cur_substr_fast_idx == cur_nof_substrs - 1:
                    cur_model.get_submodule(".".join(cur_decoder_indexing_substrs[cur_substr_slow_idx:cur_substr_fast_idx]))[int(cur_decoder_indexing_substrs[cur_substr_fast_idx])] = cur_tied_decoder_layermodule
                else:
                    cur_model = cur_model.get_submodule(".".join(cur_decoder_indexing_substrs[cur_substr_slow_idx:cur_substr_fast_idx]))[int(cur_decoder_indexing_substrs[cur_substr_fast_idx])]
                cur_substr_slow_idx = cur_substr_fast_idx + 1
            cur_substr_fast_idx += 1                             

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

        # update the corresponding layers
        cur_indexing_substrs = cur_layer["indexing_str"].split('.')
        cur_nof_substrs = len(cur_indexing_substrs)

        cur_substr_slow_idx = 0
        cur_substr_fast_idx = 0

        # iterative access corresponding layers
        cur_model = model
        while(cur_substr_fast_idx < cur_nof_substrs):
            if cur_indexing_substrs[cur_substr_fast_idx].isdigit():
                if cur_substr_fast_idx == cur_nof_substrs - 1:
                    cur_model.get_submodule(".".join(cur_indexing_substrs[cur_substr_slow_idx:cur_substr_fast_idx]))[int(cur_indexing_substrs[cur_substr_fast_idx])] = cur_untied_module
                else:
                    cur_model = cur_model.get_submodule(".".join(cur_indexing_substrs[cur_substr_slow_idx:cur_substr_fast_idx]))[int(cur_indexing_substrs[cur_substr_fast_idx])]
                cur_substr_slow_idx = cur_substr_fast_idx + 1
            cur_substr_fast_idx += 1 

    return model