# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2022, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" FP8 quantization and supporting range setting functions """

import torch

def max_float_value(mantissa_bits, total_bits=8):
    exponent_bits = total_bits - mantissa_bits - 1  # Subtract 1 for the sign bit
    # The mantissa has an implicit leading 1, so the fraction added to it can be computed from the mantissa_bits
    mantissa_val = 1 + sum([(1/2)**(i+1) for i in range(mantissa_bits)])
    
    # The bias is 2^(exponent_bits-1) - 1
    bias = 2**(exponent_bits - 1) - 1
    
    # Maximum exponent value
    max_exponent_val = (2**(exponent_bits - 1)) - 1
    #print("Max exponent value is", max_exponent_val)
    # Calculate maximum value
    max_val = mantissa_val * (2**max_exponent_val)
    #max_val = torch.tensor(max_val, dtype=torch.float32)
    return max_val
    



def fp8_quantizer(tensor, NUM_MANTISSA_BITS):
    """
    FP8 quantization entry function.
    """
    fp8_maxval = max_float_value(NUM_MANTISSA_BITS)
    # fp8_maxval = torch.tensor(fp8_maxval, dtype=torch.float32)
    
    if isinstance(fp8_maxval, torch.Tensor):
        fp8_maxval = fp8_maxval.clone().detach().requires_grad_(False)
    else:
        fp8_maxval = torch.tensor(fp8_maxval, dtype=torch.float32)
        
    return quantize_to_fp8(
        tensor, fp8_maxval, NUM_MANTISSA_BITS)


def quantize_to_fp8(x_float: torch.Tensor,
                    maxval: torch.Tensor,
                    mantissa_bits: torch.Tensor
                    ) -> torch.Tensor:
    """
    FP8 quantizer that exploits the fact that FP quantization is just INT quantization with
    scales that depend on the input.

    """
    # For learning: ensure that the number of mantissa bits is rounded and clamped to
    # allowed values. NB: torch.round should be replaced by ste_round_func (not included
    # here yet)
    # TODO for learning we need this as well:
    # mantissa_bits = torch.clamp(torch.round(mantissa_bits), 1, 7)

    # Compute exponent bits from the (learned) number of exponent bits. NB: assumes FP8
    exponent_bits = 7 - mantissa_bits

    # Math explanation of what happens here:
    # Bias is computed from maxval: $B=2^E - \log_2(M) + \log_2(2 - 2^{-M}) - 1$
    # This follows from maxval $M=(2 - 2^{-M}) \cdot 2^{2^E-1-B}$.
    
    #bias = 2 ** exponent_bits - torch.log2(maxval) + torch.log2(2 - 2 ** (-mantissa_bits)) - 1
    
    bias = 2**(exponent_bits - 1) - 1

    # Ensure no values are greater than the maximum value represented by an 8 bit float system
    # with M mantissa and E exponent bits. torch.min/torch.max are used to allow gradients to
    # flow to maxval
    x_clipped = torch.min(torch.max(x_float, -maxval), maxval)

    # FP quantization scale is determined per-element, and is computed as
    # \log_2 s = \left\lfloor \log_2 |x_c| + B \right\rfloor - M - B
    # the addition of bias inside the floor and subtraction outside ensures that a
    # tensor scaling $\alpha \neq 1$ is correctly incorporated
    log_scales = torch.floor(torch.log2(torch.abs(x_clipped)) + bias).detach()

    # This ensures scales are never smaller than the subnormal scale
    log_scales = torch.clamp(log_scales, 1.)

    # Second step of computing scale $s$
    scales = 2. ** (log_scales - mantissa_bits - bias)

    # Using the per-element scale we can quantize the clipped input tensor to the FP grid
    return torch.round(x_clipped / scales) * scales