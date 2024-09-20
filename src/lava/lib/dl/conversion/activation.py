# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import torch
import torch.nn.functional as F

from lava.lib.dl.slayer.dendrite import Sigma as SigmaDendrite
from lava.lib.dl.slayer.utils.quantize import QuantizeAndClamp, MODE


class AbstractActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._quantized = False
        self._validate = False

    def fixed_precision(self, validate=False):
        self._quantized = True
        self._validate = validate

    def full_precision(self):
        self._quantized = False
        self._validate = False

    def forward(self, x):
        if self._quantized:
            return self.forward_quant(x)
        return self.forward_full(x)

class Delta(AbstractActivation):
    def __init__(self, threshold, num_msg_bits=16, msg_exp=6) -> None:
        super().__init__()
        self.quantizer = QuantizeAndClamp(num_bits=num_msg_bits,
                                                step=1 / (1 << msg_exp))
        self.threshold = self.quantizer(torch.tensor([threshold]))


    def forward_dynamics(self, x, threshold):
        y = torch.zeros_like(x)
        act_last = torch.zeros_like(x[..., 0])
        with torch.no_grad():
            for t in range(x.shape[-1]):
                delta = x[..., t] - act_last
                y[..., t] = torch.where(torch.abs(delta) >= threshold.to(x.device),
                                        delta, 0 * delta)
                act_last = act_last + y[..., t]
        return y

    def forward_full(self, x):
        x = self.quantizer(x)
        return self.forward_dynamics(x, self.threshold)

    def forward_quant(self, x_int):
        if torch.is_floating_point(x_int):
            x_int = (self.quantizer(x_int) /
                     self.quantizer.step).to(torch.int16)
        threshold_int = self.quantizer.quantize(self.threshold).to(torch.int16)
        y_int = self.forward_dynamics(x_int, threshold_int)
        return y_int.to(torch.int16)



class SigmaDeltaReLU(AbstractActivation):
    def __init__(self, threshold, num_msg_bits=16, msg_exp=6, wgt_exp=6) -> None:
        super().__init__()
        self.sigma = SigmaDendrite()
        self.activation = F.relu
        self.delta = Delta(threshold=threshold,
                           num_msg_bits=num_msg_bits,
                           msg_exp=msg_exp)
        self.bias = torch.zeros([1, 1, 1])
        self.quantizer = self.delta.quantizer
        self.wgt_exp = wgt_exp

    def set_bias(self, bias):
        self.bias = self.quantizer(bias)

    def forward(self, x):
        if self._quantized:
            return self.forward_quant(x)
        return self.forward_full(x)

    def forward_full(self, x):
        x = self.sigma(x)
        x = self.quantizer(x, mode=MODE.FLOOR)
        x += self.bias.to(x.device)
        x = self.activation(x)
        x = self.delta.forward_full(x)
        return x

    def forward_quant(self, x_int):
        x_int = self.sigma(x_int).to(torch.int32)
        x_int = (x_int >> self.wgt_exp).to(torch.int16)
        x_int += self.quantizer.quantize(self.bias).to(x_int.device).to(torch.int16)
        x_int = self.activation(x_int)
        x_int = self.delta.forward_quant(x_int)
        return x_int.to(torch.int16)



class Sigma(AbstractActivation):
    def __init__(self, num_msg_bits=16, msg_exp=6, wgt_exp=6) -> None:
        super().__init__()
        self.sigma = SigmaDendrite()
        self.bias = torch.zeros([1, 1, 1])
        self.quantizer = QuantizeAndClamp(num_bits=num_msg_bits,
                                                step=1 / (1 << msg_exp))
        self.wgt_exp = wgt_exp

    def set_bias(self, bias):
        self.bias = self.quantizer(bias)

    def forward_full(self, x):
        x = self.sigma(x)
        x = self.quantizer(x, mode=MODE.FLOOR)
        x += self.bias.to(x.device)
        return x

    def forward_quant(self, x):
        x = self.sigma(x)
        x = (x >> self.wgt_exp).to(torch.int16)
        x += self.quantizer.quantize(self.bias).to(x.device).to(torch.int16)
        return x.to(torch.int16)
