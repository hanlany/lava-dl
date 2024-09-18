# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import torch
import torch.nn.functional as F

from lava.lib.dl.slayer.synapse import Dense, Conv
from lava.lib.dl.slayer.utils.quantize import QuantizeAndClamp, MODE


class AbstractBlock(torch.nn.Module):
    def __init__(self, activation=None, bias=False,
                 num_msg_bits=16, msg_exp=6,
                 num_wgt_bits=8, wgt_exp=6,
                 num_scale_bits=16, scale_exp=12, scale=1.0) -> None:
        # Generic implementation of
        # y = f(scale * W * x + bias)
        super().__init__()

        self._quantized = False
        self._validate = False
        self.wgt_exp = wgt_exp
        self.num_da_bits = 24
        self.num_var_bits = 24

        self.activation = activation

        if hasattr(self.activation, 'quantizer'):
            self.input_quantizer = self.activation.quantizer
        else:
            self.input_quantizer = QuantizeAndClamp(num_bits=num_msg_bits,
                                                    step=1 / (1 << msg_exp))
        self.wgt_quantizer = QuantizeAndClamp(num_bits=num_wgt_bits,
                                              step=1 / (1 << wgt_exp))
        self.da_quantizer = QuantizeAndClamp(num_bits=self.num_da_bits,
                                             step=1 / (1 << msg_exp + wgt_exp))
        self.bias_quantizer = QuantizeAndClamp(num_bits=self.num_var_bits,
                                               step=1 / (1 << msg_exp + wgt_exp))

        post_scale_exp = msg_exp + wgt_exp + num_scale_bits - scale_exp
        self.scale_quantizer = QuantizeAndClamp(num_bits=num_scale_bits + 1,
                                                step=1 / (1 << scale_exp))
        self.post_scale_quantizer = QuantizeAndClamp(num_bits=self.num_da_bits,
                                                     step=1 / (1 << post_scale_exp))
        self.scale = self.scale_quantizer(torch.tensor([scale]))
        self.scale_exp = scale_exp
        if self.scale == 1:
            self.scale = None  # No need to apply scale

        self.synapse = None
        self.bias = None


    def synapse_quant(self,
                    x_int: torch.tensor,
                    weight_int: torch.tensor) -> torch.tensor:
        raise NotImplemented

    def fixed_precision(self, validate=False):
        self._quantized = True
        self._validate = validate
        if hasattr(self.activation, 'fixed_precision'):
            self.activation.fixed_precision(validate=validate)

    def full_precision(self):
        self._quantized = False
        self._validate = False
        if hasattr(self.activation, 'full_precision'):
            self.activation.full_precision()

    def quantize(self, x: torch.tensor) -> torch.tensor:
        if torch.is_floating_point(x):
            x_int = (self.input_quantizer(x) /
                     self.input_quantizer.step).to(torch.int16)
        else:
            x_int = x.to(torch.int16)
            x = x.to(torch.float) * self.input_quantizer.step
        return x_int, x

    def dequantize(self, x: torch.tensor) -> torch.tensor:
        if torch.is_floating_point(x):
            return x
        else:
            return x.to(torch.float) * self.input_quantizer.step

    def forward(self, x):
        if self._quantized:
            return self.forward_quant(x)
        x = self.input_quantizer(x)
        z = self.synapse(x)
        z = self.da_quantizer(z)
        if self.bias is not None:
            z = z + self.bias_quantizer(self.bias)
        if self.activation is not None:
            z = self.activation(z)
            if hasattr(self.activation, 'quantizer'):
                return self.activation.quantizer(z)
        return self.input_quantizer(z, mode=MODE.FLOOR)

    def forward_quant(self, x):
        with torch.no_grad():
            x_int, x = self.quantize(x)
            if self._validate:
                x_gt = self.input_quantizer(x) / self.input_quantizer.step
                x_diff = x_int - x_gt
                if torch.abs(x_diff).max() > 0:
                    print('Validation WARNING: x error[int] = '
                          f'{torch.abs(x_diff).max()}')

            wgt = self.synapse.weight.data
            weight_int = self.wgt_quantizer.quantize(wgt).to(torch.int8)
            if self._validate:
                weight_gt = self.wgt_quantizer(wgt) / self.wgt_quantizer.step
                wgt_diff = weight_int - weight_gt
                if torch.abs(wgt_diff).max() > 0:
                    print('Validation WARNING: wgt error[int] = '
                          f'{torch.abs(wgt_diff).max()}')

            z_int = self.synapse_quant(x_int.to(torch.float64),
                                       weight_int.to(torch.float64)).to(torch.int32)

            num = 32 - self.num_var_bits
            z_int = (z_int << num) >> num
            if self._validate:
                z = self.synapse(self.input_quantizer.dequantize(x_int))
                z_diff = z_int - z / self.da_quantizer.step
                if torch.abs(z_diff).max() > 0:
                    print(f'{z.shape = }')
                    print(f'{(z / self.da_quantizer.step)[z_diff != 0].flatten() = }')
                    print(f'{z_int[z_diff != 0].flatten() = }')
                    print('Validation WARNING: w*x error[int] = '
                          f'{torch.abs(z_diff).max()}')

            if self.bias is not None:
                bias_int = self.da_quantizer.quantize(self.bias).to(torch.int32)
                z_int = z_int + bias_int
                num = 32 - self.num_var_bits
                if self._validate:
                    z = z + self.bias_quantizer(self.bias)
                    z_diff = z_int - z / self.da_quantizer.step
                    if torch.abs(z_diff).max() > 0:
                        print('Validation WARNING: w*x + bias error[int] = '
                              f'{torch.abs(z_diff).max()}')

            if self.activation is not None:
                validate = self._validate and hasattr(self.activation, 'quantizer')
                if validate:
                    z = self.activation.forward_full(self.da_quantizer.dequantize(z_int))
                elif self._validate:
                    z = self.activation(z)
                z_int = self.activation(z_int)

                if validate:
                    z_diff = z_int - z / self.activation.quantizer.step
                    if torch.abs(z_diff).max() > 0:
                        print('Validation WARNING: f(w*x + bias) error[int] = '
                              f'{torch.abs(z_diff).max()}')
                        # assert False
                if hasattr(self.activation, 'quantizer'):
                    return z_int

            z_int = (z_int >> self.wgt_exp).to(torch.int16)
            if self._validate:
                z = self.input_quantizer(z, mode=MODE.FLOOR)
                z_diff = z_int - z / self.input_quantizer.step
                if torch.abs(z_diff).max() > 0:
                    print('Validation WARNING: y error[int] = '
                          f'{torch.abs(z_diff).max()}')
            return z_int


class DenseBlock(AbstractBlock):
    def __init__(self, in_neurons, out_neurons,
                 activation=None, bias=False,
                 num_msg_bits=16, msg_exp=6,
                 num_wgt_bits=8, wgt_exp=6,
                 num_scale_bits=16, scale_exp=12, scale=1.0) -> None:
        super().__init__(activation=activation, bias=bias,
                         num_msg_bits=num_msg_bits, msg_exp=msg_exp,
                         num_wgt_bits=num_wgt_bits, wgt_exp=wgt_exp,
                         num_scale_bits=num_scale_bits, scale_exp=scale_exp,
                         scale=scale)

        self.synapse = Dense(in_neurons, out_neurons,
                             pre_hook_fx=self.wgt_quantizer)
        if bias:
            bias_shape = [1, out_neurons, 1]
            self.bias = torch.nn.Parameter(torch.zeros(bias_shape))

    def synapse_quant(self,
                    x_int: torch.tensor,
                    weight_int: torch.tensor) -> torch.tensor:
        if len(x_int.shape) == 3:
            old_shape = x_int.shape
            return F.conv3d(
                x_int.reshape(old_shape[0], -1, 1, 1, old_shape[-1]),
                weight_int, None,
                self.synapse.stride, self.synapse.padding,
                self.synapse.dilation, self.synapse.groups,
            ).reshape(old_shape[0], -1, old_shape[-1])
        else:
            return F.conv3d(
                x_int, weight_int, None,
                self.synapse.stride, self.synapse.padding,
                self.synapse.dilation, self.synapse.groups,
            )


class ConvBlock(AbstractBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=False, activation=None,
                 num_msg_bits=16, msg_exp=6,
                 num_wgt_bits=8, wgt_exp=6,
                 num_scale_bits=16, scale_exp=12, scale=1.0) -> None:
        super().__init__(activation=activation, bias=bias,
                         num_msg_bits=num_msg_bits, msg_exp=msg_exp,
                         num_wgt_bits=num_wgt_bits, wgt_exp=wgt_exp,
                         num_scale_bits=num_scale_bits, scale_exp=scale_exp,
                         scale=scale)

        self.synapse = Conv(in_channels, out_channels,
                            kernel_size, stride, padding, dilation,
                            groups, pre_hook_fx=self.wgt_quantizer)
        if bias:
            bias_shape = [1, out_channels, 1, 1, 1]
            self.bias = torch.nn.Parameter(torch.zeros(bias_shape))

    def synapse_quant(self,
                    x_int: torch.tensor,
                    weight_int: torch.tensor) -> torch.tensor:
        return F.conv3d(
            x_int.to(torch.float64), weight_int.to(torch.float64), None,
            self.synapse.stride, self.synapse.padding,
            self.synapse.dilation, self.synapse.groups,
        ).to(torch.int32)


class Flatten(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.reshape((x.shape[0], -1, x.shape[-1]))
