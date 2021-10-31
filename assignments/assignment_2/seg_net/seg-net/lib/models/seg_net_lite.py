from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = []
        layers_bn_down = []
        layers_pooling = []
        # raise NotImplementedError('Downsampling layers are not implemented!')
        _in_ch = input_size
        for idx, value_set in enumerate(
                zip(down_filter_sizes, kernel_sizes, conv_paddings,
                    pooling_kernel_sizes, pooling_strides)):
            _out_ch, _k, _pad, _pool_k, _pool_s = value_set
            layers_conv_down.append(
                nn.Conv2d(in_channels=_in_ch, out_channels=_out_ch,
                          kernel_size=_k, padding=_pad))
            layers_bn_down.append(
                nn.BatchNorm2d(num_features=_out_ch)
            )
            layers_pooling.append(
                nn.MaxPool2d(kernel_size=_pool_k, stride=_pool_s,
                             return_indices=True)
            )
            _in_ch = _out_ch  # for next round
        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = []
        layers_bn_up = []
        layers_unpooling = []

        _in_ch = down_filter_sizes[-1]  # last down sampling layer out channels
        # NOTE: reversing kernel size/strides - Should be done for paddings
        # as well
        for idx, value_set in enumerate(
                zip(up_filter_sizes, kernel_sizes, conv_paddings.reverse(),
                    pooling_kernel_sizes.reverse(),
                    pooling_strides.reverse())):
            _out_ch, _k, _pad, _unpool_k, _unpool_s = value_set
            layers_unpooling.append(
                nn.MaxUnpool2d(kernel_size=_unpool_k, stride=_unpool_s)
            )
            layers_conv_up.append(
                nn.Conv2d(in_channels=_in_ch, out_channels=_out_ch,
                          kernel_size=_k, padding=_pad))
            layers_bn_up.append(
                nn.BatchNorm2d(num_features=_out_ch)
            )
            _in_ch = _out_ch  # for next round

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        final_layer_input_ch = up_filter_sizes[-1]  # last upsampler output dim
        self.logit_layer = nn.Conv2d(in_channels=final_layer_input_ch, out_channels=11,
                                     kernel_size=1, padding=0)

    def forward(self, x):
        #  TODO: may need to assert size of module lists
        pooling_indices = []
        _x = x
        for conv_layer, bn_layer, pool_layer in \
                zip(self.layers_conv_down,
                    self.layers_bn_down, self.layers_pooling):
            _x = conv_layer(x)
            _x = bn_layer(x)
            _x = self.relu(x)
            _x, _indices = pool_layer(_x)
            pooling_indices.append(_indices)

        pooling_indices = pooling_indices.reverse()

        for unpool_layer, pool_ind, conv_layer, bn_layer,  in \
                zip(self.layers_unpooling, pooling_indices,
                    self.layers_conv_up, self.layers_bn_up):
            _x = unpool_layer(_x, pool_ind)
            _x = conv_layer(_x)
            _x = bn_layer(_x)
            _x = self.relu(_x)

        _x = self.logit_layer(_x)

        return _x

def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
