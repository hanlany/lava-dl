# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-chunk fixed-point weight decomposition utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch


class SignMode(str, Enum):
    """Sign interpretation for decomposed integer weights."""

    MIXED = "mixed"
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"


@dataclass(frozen=True)
class WeightChunk:
    """One least-significant-first chunk of a decomposed weight."""

    tensor: Any
    exponent: int
    sign_mode: SignMode
    is_msb: bool

    @property
    def contribution_scale(self) -> int:
        """Integer multiplier used when reconstructing this chunk."""
        scale = 1 << self.exponent
        return -scale if self.sign_mode is SignMode.INHIBITORY else scale


Array = Union[np.ndarray, torch.Tensor]


class WeightDecomposer:
    """Decompose integer weights into fixed-width chunks.

    Mixed mode uses unsigned lower chunks and a signed most-significant chunk.
    Excitatory mode uses unsigned chunks. Inhibitory mode decomposes magnitudes
    into unsigned chunks and negates each contribution during reconstruction.

    Out-of-range values are clamped and reported by default. The optional
    raise policy rejects any saturation.
    """

    def __init__(
        self,
        target_bits: int = 24,
        chunk_bits: int = 8,
        saturation_policy: str = "clamp",
    ) -> None:
        if not isinstance(target_bits, int) or target_bits <= 0:
            raise ValueError("target_bits must be a positive integer")
        if not isinstance(chunk_bits, int) or chunk_bits <= 0:
            raise ValueError("chunk_bits must be a positive integer")
        if target_bits % chunk_bits:
            raise ValueError("target_bits must be divisible by chunk_bits")
        if target_bits > 63:
            raise ValueError("target_bits must be <= 63 for int64 arithmetic")
        if saturation_policy not in ("clamp", "raise"):
            raise ValueError(
                "saturation_policy must be either 'clamp' or 'raise'"
            )
        self.target_bits = target_bits
        self.chunk_bits = chunk_bits
        self.num_chunks = target_bits // chunk_bits
        self.saturation_policy = saturation_policy
        self.last_diagnostics: Dict[str, Any] = {}

    @staticmethod
    def normalize_sign_mode(
        sign_mode: Union[SignMode, str],
    ) -> SignMode:
        """Validate and normalize a sign-mode value."""
        try:
            return SignMode(sign_mode)
        except ValueError as error:
            supported = ", ".join(mode.value for mode in SignMode)
            raise ValueError(
                f"Unsupported sign mode {sign_mode!r}; expected {supported}"
            ) from error

    @staticmethod
    def _is_torch(value: Array) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, np.ndarray):
            return False
        raise TypeError("weights must be a NumPy array or PyTorch tensor")

    def integer_range(
        self, sign_mode: Union[SignMode, str] = SignMode.MIXED
    ) -> tuple:
        """Return the inclusive representable integer range."""
        mode = self.normalize_sign_mode(sign_mode)
        if mode is SignMode.MIXED:
            return -(1 << (self.target_bits - 1)), (
                1 << (self.target_bits - 1)
            ) - 1
        if mode is SignMode.EXCITATORY:
            return 0, (1 << self.target_bits) - 1
        return -((1 << self.target_bits) - 1), 0

    def _prepare(self, values: Array, mode: SignMode) -> tuple:
        is_torch = self._is_torch(values)
        integers = (
            values.to(dtype=torch.int64)
            if is_torch
            else np.asarray(values, dtype=np.int64)
        )
        minimum, maximum = self.integer_range(mode)
        outside = (integers < minimum) | (integers > maximum)
        saturation_count = (
            int(torch.count_nonzero(outside).item())
            if is_torch
            else int(np.count_nonzero(outside))
        )
        if saturation_count and self.saturation_policy == "raise":
            raise OverflowError(
                f"{saturation_count} value(s) outside {mode.value} "
                f"{self.target_bits}-bit range [{minimum}, {maximum}]"
            )
        clamped = (
            torch.clamp(integers, minimum, maximum)
            if is_torch
            else np.clip(integers, minimum, maximum).astype(
                np.int64, copy=False
            )
        )
        self.last_diagnostics = {
            "saturation_count": saturation_count,
            "minimum": minimum,
            "maximum": maximum,
            "sign_mode": mode.value,
        }
        return clamped, is_torch

    def decompose(
        self,
        values: Array,
        sign_mode: Union[SignMode, str] = SignMode.MIXED,
    ) -> List[WeightChunk]:
        """Return least-significant-first chunks for integer values."""
        mode = self.normalize_sign_mode(sign_mode)
        clamped, is_torch = self._prepare(values, mode)
        working = -clamped if mode is SignMode.INHIBITORY else clamped
        mask = (1 << self.chunk_bits) - 1
        chunks = []
        for index in range(self.num_chunks):
            exponent = index * self.chunk_bits
            is_msb = index == self.num_chunks - 1
            value = (working >> exponent) & mask
            if mode is SignMode.MIXED and is_msb:
                boundary = 1 << (self.chunk_bits - 1)
                modulus = 1 << self.chunk_bits
                value = (
                    torch.where(value >= boundary, value - modulus, value)
                    if is_torch
                    else np.where(
                        value >= boundary, value - modulus, value
                    ).astype(np.int64, copy=False)
                )
            chunks.append(WeightChunk(
                tensor=value,
                exponent=exponent,
                sign_mode=mode,
                is_msb=is_msb,
            ))

        reconstructed = self.reconstruct(chunks)
        error = reconstructed - clamped
        error_count = (
            int(torch.count_nonzero(error).item())
            if is_torch
            else int(np.count_nonzero(error))
        )
        max_error = (
            int(torch.max(torch.abs(error)).item())
            if is_torch and error.numel()
            else int(np.max(np.abs(error)))
            if not is_torch and error.size
            else 0
        )
        self.last_diagnostics.update({
            "reconstruction_error_count": error_count,
            "max_reconstruction_error": max_error,
        })
        return chunks

    def reconstruct(self, chunks: Sequence[WeightChunk]) -> Array:
        """Reconstruct an integer array exactly from its chunks."""
        if len(chunks) != self.num_chunks:
            raise ValueError(
                f"Expected {self.num_chunks} chunks, received {len(chunks)}"
            )
        first = chunks[0].tensor
        is_torch = self._is_torch(first)
        result = (
            torch.zeros_like(first, dtype=torch.int64)
            if is_torch
            else np.zeros_like(first, dtype=np.int64)
        )
        for index, chunk in enumerate(chunks):
            if chunk.exponent != index * self.chunk_bits:
                raise ValueError("Chunk exponents are inconsistent")
            if chunk.is_msb != (index == self.num_chunks - 1):
                raise ValueError("Chunk MSB metadata is inconsistent")
            if self._is_torch(chunk.tensor) != is_torch:
                raise TypeError("All chunks must use the same array backend")
            value = (
                chunk.tensor.to(dtype=torch.int64)
                if is_torch
                else np.asarray(chunk.tensor, dtype=np.int64)
            )
            result = result + value * chunk.contribution_scale
        return result


class DecomposedWeightQuantizer:
    """Fixed-point STE quantizer suitable for a synapse pre-hook."""

    is_decomposed_weight_quantizer = True

    def __init__(
        self,
        target_bits: int = 24,
        chunk_bits: int = 8,
        sign_mode: Union[SignMode, str] = SignMode.MIXED,
        scale: int = 64,
        saturation_policy: str = "clamp",
    ) -> None:
        if not isinstance(scale, int) or scale <= 0:
            raise ValueError("scale must be a positive integer")
        self.decomposer = WeightDecomposer(
            target_bits, chunk_bits, saturation_policy
        )
        self.target_bits = target_bits
        self.chunk_bits = chunk_bits
        self.sign_mode = self.decomposer.normalize_sign_mode(sign_mode)
        self.scale = scale
        self.saturation_policy = saturation_policy
        self.last_diagnostics: Dict[str, Any] = {}

    def _integer_weights(self, weights: Array) -> Array:
        is_torch = self.decomposer._is_torch(weights)
        if is_torch:
            if not weights.is_floating_point():
                raise TypeError("PyTorch weights must have a floating dtype")
            return torch.round(weights.detach() * self.scale).to(
                dtype=torch.int64
            )
        if not np.issubdtype(weights.dtype, np.floating):
            raise TypeError("NumPy weights must have a floating dtype")
        return np.rint(weights * self.scale).astype(np.int64)

    def decompose(self, weights: Array) -> List[WeightChunk]:
        """Quantize floating weights and return integer chunks."""
        return self.decomposer.decompose(
            self._integer_weights(weights), self.sign_mode
        )

    def __call__(self, weights: Array, descale: bool = False) -> Array:
        if descale:
            raise RuntimeError(
                "HDF5 export is unsupported for decomposed weights because "
                "the legacy format stores only one weight tensor. Disable "
                "HDF5 export or select legacy_8bit quantization."
            )
        is_torch = self.decomposer._is_torch(weights)
        chunks = self.decompose(weights)
        reconstructed = self.decomposer.reconstruct(chunks)
        quantized = reconstructed / self.scale
        diagnostics = dict(self.decomposer.last_diagnostics)

        if is_torch:
            discrete = quantized.to(
                device=weights.device, dtype=weights.dtype
            )
            output = weights + (discrete - weights).detach()
            error = torch.abs(discrete.detach() - weights.detach())
            max_quantization_error = (
                float(torch.max(error).item()) if error.numel() else 0.0
            )
        else:
            output = quantized.astype(weights.dtype, copy=False)
            error = np.abs(output - weights)
            max_quantization_error = (
                float(np.max(error)) if error.size else 0.0
            )

        diagnostics.update({
            "target_bits": self.target_bits,
            "chunk_bits": self.chunk_bits,
            "num_chunks": self.decomposer.num_chunks,
            "scale": self.scale,
            "max_quantization_error": max_quantization_error,
        })
        self.last_diagnostics = diagnostics
        return output

