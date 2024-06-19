"""
FILENAME: DebugLayers.py
DESCRIPTION: customized layers used for debug
@author: Jian Zhong
"""

import torch
from torch import nn


# debug layer: print the shape of input 
class PrintShapeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(f"DEBUG:\t x.size = {x.size()}")
        return x


# debug layer: print value of input 
class PrintValueLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(f"DEBUG:\t x = {x}")
        return x


# debug layer: send out message 
class PrintMessageLayer(nn.Module):
    def __init__(self,message = ""):
        super().__init__()
        self.message = message
    
    def forward(self, x):
        print("DEBUG:\t MSG: " + self.message)
        return x