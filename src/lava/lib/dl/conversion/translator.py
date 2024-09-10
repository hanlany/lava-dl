# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import numpy as np

import torch
import torch.nn.functional as F

from .block import DenseBlock, ConvBlock

def map_dense_block(dense, bn=None, activation=None,
                    num_wgt_bits=8, input_exp=6, da_scale=1.0):
    bn_bias = 0
    bn_weight_factor = 1
    if bn is not None:
        num = 1
        bias = 0
        var = bn.running_var.data.clone().detach()
        mean = bn.running_mean.data.clone().detach()
        if bn.affine:
            num = bn.weight.data.clone().detach()
            bias = bn.bias.data.clone().detach()
        den = torch.sqrt(var + bn.eps)
        bn_weight_factor = (num / den).reshape([-1, 1])
        bn_bias = (bias - mean * num / den).reshape([1, -1])

    weight = dense.weight.data.clone().detach() * bn_weight_factor
    if dense.bias is not None:
        bias = dense.bias.data.clone().detach() + bn_bias
    else:
        bias = bn_bias

    wgt_int_exp = int(
        np.ceil(np.log2(torch.abs(weight).max().cpu().data.numpy())))
    wgt_exp = num_wgt_bits - wgt_int_exp

    bias_in_block = True
    if hasattr(activation, 'set_bias'):
        bias_in_block = False

    block = DenseBlock(in_neurons=dense.in_features,
                       out_neurons=dense.out_features,
                       bias=bias_in_block,
                       activation=activation,
                       num_wgt_bits=num_wgt_bits,
                       wgt_exp=wgt_exp,
                       msg_exp=input_exp,
                       scale=da_scale)

    block.synapse.weight.data = weight.reshape(block.synapse.weight.shape)
    # block.bias.data = bias.reshape(block.bias.shape)
    if bias_in_block:
        block.bias.data = bias.reshape(block.bias.shape)
    else:
        bias = bias.reshape([1, -1, 1])
        block.activation.set_bias(bias)
        block.activation.wgt_exp = wgt_exp

    return block


def map_conv_block(conv, bn=None, activation=None,
                   num_wgt_bits=8, input_exp=6, da_scale=1.0):
    bn_bias = 0
    bn_weight_factor = 1
    if bn is not None:
        num = 1
        bias = 0
        var = bn.running_var.data.clone().detach()
        mean = bn.running_mean.data.clone().detach()
        if bn.affine:
            num = bn.weight.data.clone().detach()
            bias = bn.bias.data.clone().detach()
        den = torch.sqrt(var + bn.eps)
        bn_weight_factor = (num / den).reshape([-1, 1, 1, 1])
        bn_bias = (bias - mean * num / den).reshape([1, -1, 1, 1])

    weight = conv.weight.data.clone().detach() * bn_weight_factor
    if conv.bias is not None:
        bias = conv.bias.data.clone().detach().reshape([1, -1, 1, 1]) + bn_bias
    else:
        bias = bn_bias

    wgt_int_exp = int(
        np.ceil(np.log2(torch.abs(weight).max().cpu().data.numpy())))
    wgt_exp = num_wgt_bits - wgt_int_exp

    bias_in_block = True
    if hasattr(activation, 'set_bias'):
        bias_in_block = False

    block = ConvBlock(in_channels=conv.in_channels,
                      out_channels=conv.out_channels,
                      kernel_size=conv.kernel_size,
                      stride=conv.stride,
                      padding=conv.padding,
                      dilation=conv.dilation,
                      groups=conv.groups,
                      bias=bias_in_block,
                      activation=activation,
                      num_wgt_bits=num_wgt_bits,
                      wgt_exp=wgt_exp,
                      msg_exp=input_exp,
                      scale=da_scale)

    block.synapse.weight.data = weight.reshape(block.synapse.weight.shape)
    if bias_in_block:
        block.bias.data = bias.reshape(block.bias.shape)
    else:
        bias = bias.reshape([1, -1, 1, 1, 1])
        block.activation.set_bias(bias)
        block.activation.wgt_exp = wgt_exp

    return block
