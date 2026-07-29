# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lava.lib.dl import slayer
from lava.lib.dl.slayer.utils import (
    DecomposedWeightQuantizer,
    SignMode,
    WeightDecomposer,
)


@pytest.mark.parametrize(
    "integer, expected",
    [
        (8388607, [255, 255, 127]),
        (-8388608, [0, 0, -128]),
        (-1, [255, 255, -1]),
        (0, [0, 0, 0]),
    ],
)
def test_mixed_golden_vectors(integer, expected):
    decomposer = WeightDecomposer(24, 8)
    values = np.asarray([integer], dtype=np.int64)
    chunks = decomposer.decompose(values, SignMode.MIXED)
    assert [int(chunk.tensor[0]) for chunk in chunks] == expected
    np.testing.assert_array_equal(decomposer.reconstruct(chunks), values)
    assert decomposer.last_diagnostics["saturation_count"] == 0
    assert decomposer.last_diagnostics["reconstruction_error_count"] == 0


def test_random_mixed_integer_reconstruction():
    rng = np.random.default_rng(17)
    values = rng.integers(
        -(1 << 23), 1 << 23, size=10000, dtype=np.int64
    )
    decomposer = WeightDecomposer()
    chunks = decomposer.decompose(values)
    np.testing.assert_array_equal(decomposer.reconstruct(chunks), values)


def test_excitatory_and_inhibitory_golden_vectors():
    decomposer = WeightDecomposer()
    excitatory = np.asarray([0, 1, 255, 256, (1 << 24) - 1])
    chunks = decomposer.decompose(excitatory, SignMode.EXCITATORY)
    np.testing.assert_array_equal(decomposer.reconstruct(chunks), excitatory)
    assert [int(chunk.tensor[-1]) for chunk in chunks] == [255, 255, 255]

    inhibitory = -excitatory
    chunks = decomposer.decompose(inhibitory, SignMode.INHIBITORY)
    np.testing.assert_array_equal(decomposer.reconstruct(chunks), inhibitory)
    assert [int(chunk.tensor[-1]) for chunk in chunks] == [255, 255, 255]
    assert all(chunk.contribution_scale < 0 for chunk in chunks)


@pytest.mark.parametrize(
    "mode, values, expected",
    [
        (SignMode.MIXED, [-8388609, 8388608], [-8388608, 8388607]),
        (SignMode.EXCITATORY, [-1, 16777216], [0, 16777215]),
        (SignMode.INHIBITORY, [-16777216, 1], [-16777215, 0]),
    ],
)
def test_saturation_clamps_and_counts(mode, values, expected):
    decomposer = WeightDecomposer()
    chunks = decomposer.decompose(np.asarray(values), mode)
    np.testing.assert_array_equal(
        decomposer.reconstruct(chunks), np.asarray(expected)
    )
    assert decomposer.last_diagnostics["saturation_count"] == 2


def test_saturation_raise_policy():
    decomposer = WeightDecomposer(saturation_policy="raise")
    with pytest.raises(OverflowError, match="outside"):
        decomposer.decompose(np.asarray([1 << 23]))


@pytest.mark.parametrize(
    "args, match",
    [
        ({"target_bits": 0}, "positive"),
        ({"chunk_bits": 0}, "positive"),
        ({"target_bits": 24, "chunk_bits": 7}, "divisible"),
        ({"target_bits": 64}, "<= 63"),
        ({"saturation_policy": "ignore"}, "clamp.*raise"),
    ],
)
def test_invalid_decomposer_config(args, match):
    with pytest.raises(ValueError, match=match):
        WeightDecomposer(**args)


def test_invalid_sign_mode_and_input_type():
    decomposer = WeightDecomposer()
    with pytest.raises(ValueError, match="Unsupported sign mode"):
        decomposer.decompose(np.asarray([0]), "unsigned")
    with pytest.raises(TypeError, match="NumPy.*PyTorch"):
        decomposer.decompose([0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quantizer_preserves_torch_shape_dtype_device_and_ste(dtype):
    weights = torch.tensor(
        [-1.001, -0.1, 0.0, 0.1, 1.001],
        dtype=dtype,
        requires_grad=True,
    )
    quantizer = DecomposedWeightQuantizer()
    output = quantizer(weights)
    assert output.shape == weights.shape
    assert output.dtype == weights.dtype
    assert output.device == weights.device
    assert torch.max(torch.abs(output - weights)).item() <= 1 / 128 + 1e-7
    output.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))
    assert quantizer.last_diagnostics["saturation_count"] == 0


def test_numpy_torch_quantization_parity():
    values = np.asarray(
        [-2.125, -0.007, 0.0, 0.007, 2.125], dtype=np.float32
    )
    numpy_quantizer = DecomposedWeightQuantizer()
    torch_quantizer = DecomposedWeightQuantizer()
    numpy_output = numpy_quantizer(values)
    torch_output = torch_quantizer(torch.from_numpy(values)).numpy()
    np.testing.assert_array_equal(numpy_output, torch_output)
    assert numpy_output.shape == values.shape
    assert numpy_output.dtype == values.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_quantizer_preserves_cuda_device():
    weights = torch.randn(32, device="cuda", dtype=torch.float32)
    output = DecomposedWeightQuantizer()(weights)
    assert output.device == weights.device
    assert output.dtype == weights.dtype
    assert output.shape == weights.shape


def test_dense_forward_matches_reference_reconstructed_weight():
    quantizer = DecomposedWeightQuantizer()
    layer = slayer.synapse.Dense(3, 2, pre_hook_fx=quantizer)
    inputs = torch.randn(4, 3, 5)
    output = layer(inputs)

    weight = quantizer(layer.weight)
    expected = F.conv3d(
        inputs.reshape(4, 3, 1, 1, 5),
        weight,
        bias=None,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    ).reshape(4, 2, 5)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_public_api_and_metadata():
    assert SignMode.MIXED.value == "mixed"
    quantizer = DecomposedWeightQuantizer(24, 8, "mixed")
    chunks = quantizer.decompose(np.asarray([8388607 / 64], dtype=np.float64))
    assert [chunk.exponent for chunk in chunks] == [0, 8, 16]
    assert [chunk.is_msb for chunk in chunks] == [False, False, True]
    assert all(chunk.sign_mode is SignMode.MIXED for chunk in chunks)

def test_affine_forward_matches_reference_reconstructed_weight():
    quantizer = DecomposedWeightQuantizer()
    layer = slayer.block.cuba.Affine(
        neuron_params={
            "threshold": 0.2,
            "current_decay": 0.3,
            "voltage_decay": 0.02,
        },
        in_neurons=3,
        out_neurons=2,
        dynamics=False,
        pre_hook_fx=quantizer,
    )
    inputs = torch.randn(4, 3, 5)
    output = layer(inputs)

    weight = quantizer(layer.synapse.weight)
    expected = F.conv3d(
        inputs.reshape(4, 3, 1, 1, 5),
        weight,
        bias=None,
        stride=layer.synapse.stride,
        padding=layer.synapse.padding,
        dilation=layer.synapse.dilation,
        groups=layer.synapse.groups,
    ).reshape(4, 2, 5)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_decomposed_hdf5_export_is_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    quantizer = DecomposedWeightQuantizer()
    layer = slayer.block.cuba.Dense(
        neuron_params={
            "threshold": 0.2,
            "current_decay": 0.3,
            "voltage_decay": 0.02,
        },
        in_neurons=3,
        out_neurons=2,
        pre_hook_fx=quantizer,
    )
    layer(torch.zeros(1, 3, 1))
    with h5py.File(tmp_path / "decomposed.net", "w") as handle:
        with pytest.raises(RuntimeError, match="legacy_8bit"):
            layer.export_hdf5(handle.create_group("layer/0"))

