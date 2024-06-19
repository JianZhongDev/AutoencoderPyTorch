"""
FILENAME: SparseLayers.py
DESCRIPTION: Layer definitions for employing sparsity in models
@author: Jian Zhong
"""

import torch
from torch import nn


# hard sparsity function to select the largest k features for each sample in the batch input data
# NOTE: this function works on 1d feature space
class FeatureTopKFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k):
        assert len(x.size()) == 2

        src_data_detach = x.detach() 

        # create mask indicating the top k features for each sample within the feature space
        topk_mask = torch.zeros_like(x, dtype = bool, requires_grad = False)
        _, indices = src_data_detach.topk(k, dim = -1)
        for i_batch in range(x.size(0)):
            topk_mask[i_batch, indices[i_batch,:]] = True

        # save mask for backward propagation
        ctx.save_for_backward(topk_mask)

        # only propagate largest k features of each sample 
        y = torch.zeros_like(x)
        y[topk_mask] = x[topk_mask]

        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        topk_mask = ctx.saved_tensors[0]
        
        # only propagate gradient for largest k features of each sample
        grad_input = torch.zeros_like(grad_output, requires_grad = True)
        grad_input[topk_mask] = grad_output[topk_mask]

        return grad_input, None


# lifetime sparsity functon to select the largest k samples for each feature
# NOTE: this function works on 1d feature space 
class LifetimeTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        assert len(x.size()) == 2

        k = min(k, x.size(0))

        src_data_detach = x.detach()

        # create mask indicating the top k samples for each feature along the batch dimension
        topk_mask = torch.zeros_like(x, dtype = bool, requires_grad = False)
        _, indices = src_data_detach.topk(k, dim = 0) 
        for i_feature in range(x.size(-1)):
            topk_mask[indices[:,i_feature],i_feature] = True

        # save mask indicationg the top k samples for each feature for back propagation
        ctx.save_for_backward(topk_mask)

        # only propagate largest k samples for each feature
        y = torch.zeros_like(x)
        y[topk_mask] = x[topk_mask]

        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        topk_mask = ctx.saved_tensors[0]
        
        # only propagate gradient for largest k samples for each feature
        grad_input = torch.zeros_like(grad_output, requires_grad = True)
        grad_input[topk_mask] = grad_output[topk_mask]

        return grad_input, None


# hard sparsity layer 
class TopKSparsity(nn.Module):
    def __init__(self, topk = 1):
        super().__init__()
        self.topk = topk

    def __repr__(self):
        return self.__class__.__name__ + f"(topk = {self.topk})"

    def forward(self, x):
        y = FeatureTopKFunction.apply(x, self.topk)
        return y


# lifetime sparsity layer
class LifetimeTopkSparsity(nn.Module):
    def __init__(self, topk = 5):
        super().__init__()
        self.topk = topk

    def __repr__(self):
        return self.__class__.__name__ + f"(topk = {self.topk})"
    
    def forward(self, x):
        y = None
        if self.training:
            # only apply lifetime sparsity during training
            y = LifetimeTopKFunction.apply(x, self.topk)
        else:
            y = x
        return y


    