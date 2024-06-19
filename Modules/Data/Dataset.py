"""
FILENAME: Dataset.py
DESCRIPTION: Helper function for dataset conversion and generation.
@author: Jian Zhong
"""

import torch

# convert supervised data to autoencoder data
def supdata_to_autoencoderdata(
        supdata,
        feature_transform = None,
        target_transform = None,
):
    src_feature = supdata[0] #extract feature
    
    # NOTE: the usuer of this function is responsible for necessary data duplication
    feature = src_feature
    if feature_transform: 
        feature = feature_transform(feature)
    
    target = src_feature
    if target_transform:
        target = target_transform(target)

    return feature, target

# dataset class of autoencoder using existing supervised learning dataset
class AutoencoderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            src_supdataset,
            feature_transform = None,
            target_transform = None,
    ):
        self.dataset = src_supdataset
        self.feature_transform = feature_transform
        self.target_transform = target_transform 

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        src_data = self.dataset[idx]
        feature, target = supdata_to_autoencoderdata(
            src_data,
            self.feature_transform,
            self.target_transform,
        )
        return feature, target