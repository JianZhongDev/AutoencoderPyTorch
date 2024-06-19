"""
FILENAME: Transforms.py
DESCRIPTION: Data transform definitions.
@author: Jian Zhong
"""

import torch


# Rondomly choose pixels and set them to a constant value
class RandomSetConstPxls(object):
    """
    Rondomly choose pixels and set them to a constant value
    """
    def __init__(
            self, 
            rand_rate = 0.5,
            const_val = 0,
            ):
        self.rand_rate = rand_rate
        self.const_val = const_val

    def __call__(self, src_image):
        src_image_size = src_image.size()
        tot_nof_pxls = src_image.nelement()

        # calculate number of randomly choosed pixel 
        nof_mod_pxls = tot_nof_pxls * self.rand_rate
        nof_mod_pxls = int(nof_mod_pxls)

        # generate mask for chosen pixels
        mod_pxl_mask = torch.full((tot_nof_pxls,), False)
        mod_pxl_mask[:nof_mod_pxls] = True
        mod_pxl_mask = mod_pxl_mask[torch.randperm(tot_nof_pxls)]

        # clone image and set the chosen pixels to corresponding contant value
        dst_image = src_image.clone()
        dst_image = dst_image.view(-1)
        dst_image[mod_pxl_mask] = self.const_val
        dst_image = dst_image.view(src_image_size)

        return dst_image
    
    def __repr__(self):
        return self.__class__.__name__ + f"(rand_rate = {self.rand_rate}, const_val = {self.const_val})"
    

# Add gaussian noise to image pixel values
class AddGaussianNoise(object):
    """
        Add gaussian noise to image pixel values
    """
    def __init__(
            self,
            mean = 0.0,
            variance = 1.0,
            generator = None,
    ):
        self.mean = mean
        self.variance = variance
        self.generator = generator # random number generator

    def __call__(self, src_image):
        src_image_shape = src_image.size()

        # generate random gaussian noise
        gauss_noise = torch.randn(
            size = src_image_shape,
            generator = self.generator,
            )
        gauss_noise = self.mean + (self.variance ** 0.5) * gauss_noise
        
        # add guassian noise to image 
        return src_image + gauss_noise

    def __repr__(self):
        return self.__class__.__name__ + f"(mean = {self.mean}, variance = {self.variance}, generator = {self.generator})"


# Clip image channel ranges to be within [min, max]
class ClipChannelValues(object):
    """
        Clip image channel ranges to be within [min, max]
    """
    def __init__(
            self,
            min = 0.0,
            max = 1.0,
    ):
        self.min = min
        self.max = max

    def __call__(self, src_image):
        
        # make a copy of source image
        dst_image = src_image.clone()

        # clip pixel values
        dst_image[dst_image < self.min] = self.min
        dst_image[dst_image > self.max] = self.max

        return dst_image
    
    def __repr__(self):
        return self.__class__.__name__ + f"(min = {self.min}, max = {self.max})"


# Reshape input data
class Reshape(object):
    """
        Reshape data
    """
    def __init__(
            self,
            shape,
    ):
        self.shape = shape

    def __call__(self, src_data):
        return src_data.view(self.shape)
    
    def __repr__(self):
        return self.__class__.__name__ + f"(shape = {self.shape})"