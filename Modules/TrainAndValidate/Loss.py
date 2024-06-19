"""
FILENAME: Loss.py
DESCRIPTION: Loss function definitions.
@author: Jian Zhong
"""

import torch
from torch import nn


# Kullback-Leibler divergence formula
def kullback_leibler_divergence(
        rho, rho_hat    
):
    return rho * torch.log(rho/rho_hat) + (1 - rho)*torch.log((1 - rho)/(1 - rho_hat))


# nn.Module of sparsity loss function 
class KullbackLeiblerDivergenceLoss(nn.Module):
    def __init__(self, rho = 0.05):
        assert rho > 0 and rho < 1
        super().__init__()
        self.rho = rho 

    def forward(self, x):
        rho_hat = torch.mean(x, dim = 0)
        kl = torch.mean(kullback_leibler_divergence(self.rho, rho_hat))
        return kl
    


