# -*- coding: utf-8 -*-
# File   : antialiased_cnns_converter.py
# Author : tattaka
# Email  : tattaka666@gmail.com
# Date   : 27/11/2019
#
# This file is part of Antialiased-CNNs-Converter-PyTorch
# https://github.com/tattaka/Antialiased-CNNs-Converter-PyTorch
# Distributed under MIT License.

import torch
from torch import nn

from .functions import Downsample, Downsample1D

__all__ = [
     'convert_model'
]

class BatchNorm2D_ReLU(nn.Module):
    def __init__(self, num_features, inplace=True):
        super(BatchNorm2D_ReLU, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, input):
        return self.relu(self.bn(input))
    
    
class BatchNorm1D_ReLU(nn.Module):
    def __init__(self, num_features, inplace=True):
        super(BatchNorm1D_ReLU, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, input):
        return self.relu(self.bn(input))
    
class MaxPool2D_Downsample(nn.Module):
    def __init__(self, filt_size, kernel_size=2, stride=2):
        super(MaxPool2D_Downsample, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1)
        self.downsample = Downsample(filt_size=filt_size, stride=stride)
    def forward(self, input):
        return self.downsample(self.max_pool(input))

class Conv2D_ReLU_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, filt_size, kernel_size, stride, padding, inplace=True):
        super(Conv2D_ReLU_Downsample, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=1,padding=padding)
        self.relu = nn.ReLU(inplace=inplace)
        self.downsample = Downsample(filt_size=filt_size, stride=stride[0])
    def forward(self, input):
        return self.downsample(self.relu(self.conv2d(input)))

    
def convert_model(module, filt_size=3):
    """Traverse the input module and its child recursively
       and replace.
    Args:
        module: the input module needs to be convert to Antialiased-CNN model
        filt_size: the blur filter size using Antialiased-CNN model
    Examples:
        >>> import torch.nn as nn
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = nn.DataParallel(m)
        >>> # after convert, m is using Antialiased-CNNs
        >>> m = convert_model(m)
    """

    mod = module
    if isinstance(module, nn.ReLU):
        mod = torch.nn.Identity()
    elif isinstance(module, nn.AvgPool2d):
        mod = Downsample(filt_size=filt_size, stride=module.stride)
    elif isinstance(module, nn.Conv2d):
        mod = Conv2D_ReLU_Downsample(in_channels=module.in_channels, out_channels=module.out_channels, filt_size=filt_size, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding)
    elif isinstance(module, nn.MaxPool2d):
        mod = MaxPool2D_Downsample(filt_size=filt_size, kernel_size=module.kernel_size, stride=module.stride)
    elif isinstance(module, nn.modules.batchnorm.BatchNorm2d):
        mod = BatchNorm2D_ReLU(module.num_features)
    for name, child in module.named_children():
        mod.add_module(name, convert_model(child, filt_size=filt_size))
    return mod