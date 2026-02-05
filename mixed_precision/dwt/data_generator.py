#!/bin/python3
import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, TextIO

import math
import numpy as np
import torch
import pywt
from dataclasses import dataclass
from typing import Union, Literal

# repo root: translib_jr (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# project helpers (same style as FIR)
from utils.helper_functions import (
    DKind,
    _to_dkind,
    dkind_name,
    _cast_apply,
    finalize_out_dtype,
    _acc_dtype_main,
    acc_add,
    check_pulp_warnings,
    select_dtypes,
    check_vec_flag,  # not strictly used here, but imported for parity/debug if you want
    check_cast,
    matrix_init_like as matrix_init,
    str2bool,
    error_metric,
)

# dataset (expects `input_signal` in there, same as your original)
from data import input_signal


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class DWTConfig:
    # flags
    mac: bool
    vec: bool
    cast: bool
    hw_mixed: bool
    mixed_vec: bool
    mantissa_bits: int

    # dtypes
    dt_x: DKind     # signal dtype
    dt_f: DKind     # filter dtype
    dt_out: DKind   # output / accumulation policy dtype

    # optional cast target
    cast_to: Optional[DKind] = None

# ----------------------------
# Small utilities
# ----------------------------

def _work_dtype(kind: DKind) -> torch.dtype:
    """Torch dtype to use in buffers; FP8_CUSTOM uses float32 storage with rounding via finalize_out_dtype."""
    return torch.float32 if (kind is DKind.FP8_CUSTOM) else kind.to_torch()

def _maybe_cast(x: torch.Tensor, cfg: DWTConfig) -> torch.Tensor:
    """Apply front-end cast (FIR-style) if requested."""
    if cfg.cast and (cfg.cast_to is not None) and not cfg.hw_mixed and not cfg.mixed_vec:
        return _cast_apply(x, cfg.cast_to, cfg.mantissa_bits)
    return x

def write_matrix(
    matrix_to_write: torch.Tensor,
    name: str,
    length: int,
    file_pointer: TextIO,
    float_type: torch.dtype,
) -> None:
    """
    Writes a matrix to a file in the specified format.

    Args:
        matrix_to_write (torch.Tensor): The matrix to write. Shape: (N,) or (N, M).
        name (str): The name of the matrix (used as a variable name in the file).
        length (int): The length of the matrix (used in array declarations).
        file_pointer (TextIO): The file pointer where the matrix will be written.
        float_type (torch.dtype): The data type of the matrix elements.

    Returns:
        None
    """
    # Ensure matrix_to_write is a 1D tensor
    if matrix_to_write.dim() > 1:
        raise ValueError(
            f"Error: Expected a 1D tensor, but got shape {matrix_to_write.shape}."
        )

    # Determine the declaration style based on the name
    if "ref" in name:
        file_pointer.write(f"PI_L2 OUT_TYPE {name}[{length}] = {{")
    elif "Input" in name:
        file_pointer.write(f"DATA_LOCATION INP_TYPE {name}[{length}] = {{")
    else:
        file_pointer.write(f"DATA_LOCATION FIL_TYPE {name}[{length}] = {{")

    # Determine the replacement string based on float type
    float_removals = {
        torch.float32: ")",
        torch.float16: ", dtype=torch.float16)",
        torch.bfloat16: ", dtype=torch.bfloat16)",
    }
    rem_part = float_removals.get(
        float_type, ")"
    )  # Default to `)` if dtype is unexpected

    # Convert tensor elements to string and format them
    matrix_string = ", ".join(
        str(matrix_to_write[i].item()).replace("tensor(", "").replace(rem_part, "")
        for i in range(matrix_to_write.shape[0])
    )

    # Write the formatted matrix values to the file
    file_pointer.write(matrix_string)
    file_pointer.write("};\n")

# ----------------------------
# Core DWT kernels
# ----------------------------



@dataclass
class _VecBufs:
    a: torch.Tensor
    b: torch.Tensor
    inp: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    sum_a: torch.Tensor
    sum_b: torch.Tensor

    @classmethod
    def alloc(cls, vec_step: int, in_t: torch.dtype, fil_t: torch.dtype,
            acc_t: torch.dtype, device: torch.device) -> "_VecBufs":
        return cls(
            a=torch.zeros(vec_step, dtype=acc_t, device=device),
            b=torch.zeros(vec_step, dtype=acc_t, device=device),
            inp=torch.empty(vec_step, dtype=in_t,  device=device),
            lo=torch.empty(vec_step,  dtype=fil_t, device=device),
            hi=torch.empty(vec_step,  dtype=fil_t, device=device),
            sum_a=torch.zeros(1, dtype=acc_t, device=device),
            sum_b=torch.zeros(1, dtype=acc_t, device=device),
        )

    def reset_pair(self):
        self.a.zero_()
        self.b.zero_()
        self.sum_a.zero_()
        self.sum_b.zero_()

def _lane_fma_reduce(
    a: torch.Tensor,
    b: torch.Tensor,
    inp: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    cfg: DWTConfig,
    zero_ten: torch.Tensor,
    vec_step: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform lane-wise multiply-accumulate for 1 vector chunk and return partial reductions.
    - a, b: lane accumulators (shape [vec_step], acc dtype)
    - inp, lo, hi: lane inputs (shape [vec_step], already loaded and typed)
    - returns: (updated a, b, part_a, part_b) where part_* are the lane-reduced sums (for mixed_vec)
    """
    assert vec_step in (2, 4)
    for k in range(vec_step):
        inpk = inp[k]
        lok  = lo[k]
        hik  = hi[k]
        if cfg.mixed_vec:
            a[k] = acc_add(zero_ten, inpk, lok, cfg, leftover=False, in_main=True)
            b[k] = acc_add(zero_ten, inpk, hik, cfg, leftover=False, in_main=True)
        else:
            a[k] = acc_add(a[k], inpk, lok, cfg, leftover=False, in_main=True)
            b[k] = acc_add(b[k], inpk, hik, cfg, leftover=False, in_main=True)
            a[k] = finalize_out_dtype(a[k], cfg)
            b[k] = finalize_out_dtype(b[k], cfg)

    if vec_step == 2:
        part_a = a[0] + a[1]
        part_b = b[0] + b[1]
    else:
        part_a = (a[0] + a[1]) + (a[2] + a[3])
        part_b = (b[0] + b[1]) + (b[2] + b[3])

    return a, b, part_a, part_b
def dwt_step(
    input_sig: torch.Tensor,
    output_sig: torch.Tensor,
    n: int,
    idx_level: int,
    nc: int,
    Lo: torch.Tensor,
    Hi: torch.Tensor,
    cfg: DWTConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One DWT stage. Same loop structure as your original (prologue/main/tails + next-input copy).
    Numerical policy via:
    - optional front-end cast (`_cast_apply`) like FIR
    - MAC promotion via `_acc_dtype_main(cfg)` when cfg.mac or cfg.hw_mixed
    - FP8/BF16/FP16 rounding via `finalize_out_dtype`
    - multiply-accumulate via `acc_add(acc, a, b, cfg, ...)`
    """
    ii = 0
    core_id = 0
    num_cores = 1

    in_t  = _work_dtype(cfg.dt_x)
    fil_t = _work_dtype(cfg.dt_f)
    out_t = _work_dtype(cfg.dt_out)
    
    acc_t = _acc_dtype_main(cfg) if (cfg.mac or cfg.hw_mixed) else out_t

    # ------------------------
    # Non-vectorized branch
    # ------------------------
    if not cfg.vec:
        # prologue
        for i in range((nc // 2) - 1):

            a = torch.zeros(1, dtype=acc_t, device=input_sig.device)
            b = torch.zeros(1, dtype=acc_t, device=input_sig.device)
            j = 0
            for j in range(2 * (i + 1)):
                inp = input_sig[2 * (i + 1) - j - 1].to(in_t)
                lo  = Lo[j].to(fil_t)
                hi  = Hi[j].to(fil_t)
                
                inp = _maybe_cast(inp, cfg)
                lo  = _maybe_cast(lo,  cfg)
                hi  = _maybe_cast(hi,  cfg)
                # a += inp * lo; b += inp * hi
                a  = acc_add(a,  inp, lo, cfg, leftover=False, in_main=True)
                b  = acc_add(b,  inp, hi, cfg, leftover=False, in_main=True)
                if not cfg.hw_mixed:
                    a = finalize_out_dtype(a, cfg)
                    b = finalize_out_dtype(b, cfg)

            if cfg.mac or cfg.hw_mixed:
                a = finalize_out_dtype(a, cfg)
                b = finalize_out_dtype(b, cfg)
            output_sig[ii + core_id]               = a.to(out_t)
            output_sig[ii + core_id + idx_level]   = b.to(out_t)
            ii += num_cores
        # main body
        for i in range((nc - 1), n, 2):
            a = torch.zeros(1, dtype=acc_t, device=input_sig.device)
            b = torch.zeros(1, dtype=acc_t, device=input_sig.device)

            for j in range(nc):
                inp = input_sig[i - j].to(in_t)
                lo  = Lo[j].to(fil_t)
                hi  = Hi[j].to(fil_t)

                inp = _maybe_cast(inp, cfg)
                lo  = _maybe_cast(lo,  cfg)
                hi  = _maybe_cast(hi,  cfg)

                a  = acc_add(a, inp, lo, cfg, leftover=False, in_main=True)
                b  = acc_add(b, inp, hi, cfg, leftover=False, in_main=True)

                if not cfg.hw_mixed:
                    a = finalize_out_dtype(a, cfg)
                    b = finalize_out_dtype(b, cfg)

            if cfg.mac or cfg.hw_mixed:
                a = finalize_out_dtype(a, cfg)
                b = finalize_out_dtype(b, cfg)

            output_sig[ii + core_id]             = a.to(out_t)
            output_sig[ii + core_id + idx_level] = b.to(out_t)
            ii += num_cores

        # even tail
        if n % 2 == 0:  # even
            for i in range((nc // 2) - 1):
                a = torch.zeros(1, dtype=acc_t, device=input_sig.device)
                b = torch.zeros(1, dtype=acc_t, device=input_sig.device)
                for j in range(nc - 2 * (i + 1)):
                    inp = input_sig[n - j - 1].to(in_t)
                    lo  = Lo[2 * (i + 1) + j].to(fil_t)
                    hi  = Hi[2 * (i + 1) + j].to(fil_t)

                    inp = _maybe_cast(inp, cfg)
                    lo  = _maybe_cast(lo,  cfg)
                    hi  = _maybe_cast(hi,  cfg)

                    a  = acc_add(a, inp, lo, cfg, leftover=False, in_main=True)
                    b  = acc_add(b, inp, hi, cfg, leftover=False, in_main=True)

                    if not cfg.hw_mixed:
                        a = finalize_out_dtype(a, cfg)
                        b = finalize_out_dtype(b, cfg)

                if cfg.mac or cfg.hw_mixed:
                    a = finalize_out_dtype(a, cfg)
                    b = finalize_out_dtype(b, cfg)

                output_sig[ii + core_id]             = a.to(out_t)
                output_sig[ii + core_id + idx_level] = b.to(out_t)
                ii += num_cores

        # odd tail
        if n % 2 == 1:  # odd
            for i in range(nc // 2):
                a = torch.zeros(1, dtype=acc_t, device=input_sig.device)
                b = torch.zeros(1, dtype=acc_t, device=input_sig.device)

                for j in range(nc - 2 * (i + 1) + 1):
                    inp = input_sig[n - j - 1].to(in_t)
                    lo  = Lo[2 * (i + 1) + j - 1].to(fil_t)
                    hi  = Hi[2 * (i + 1) + j - 1].to(fil_t)

                    inp = _maybe_cast(inp, cfg)
                    lo  = _maybe_cast(lo,  cfg)
                    hi  = _maybe_cast(hi,  cfg)

                    a  = acc_add(a, inp, lo, cfg, leftover=False, in_main=True)
                    b  = acc_add(b, inp, hi, cfg, leftover=False, in_main=True)

                    if not cfg.hw_mixed:
                        a = finalize_out_dtype(a, cfg)
                        b = finalize_out_dtype(b, cfg)

                if cfg.mac or cfg.hw_mixed:
                    a = finalize_out_dtype(a, cfg)
                    b = finalize_out_dtype(b, cfg)

                output_sig[ii + core_id]             = a.to(out_t)
                output_sig[ii + core_id + idx_level] = b.to(out_t)
                ii += num_cores

        # copy next inputs (low/high back to input)
        next_inputs = (n + nc - 1) // 2
        
        # finalize-to-X policy
        cfg_x = DWTConfig(
            mac=cfg.mac, vec=cfg.vec, cast=False, hw_mixed=cfg.hw_mixed, mixed_vec=False,
            mantissa_bits=cfg.mantissa_bits,
            dt_x=cfg.dt_x, dt_f=cfg.dt_f, dt_out=cfg.dt_x, cast_to=None
        )
        for i in range(0, (next_inputs // 2) * 2, 2):
            a = finalize_out_dtype(output_sig[i],   cfg_x).to(in_t)
            b = finalize_out_dtype(output_sig[i+1], cfg_x).to(in_t)
            input_sig[i]   = a
            input_sig[i+1] = b
        if next_inputs & 0x1:
            a = finalize_out_dtype(output_sig[next_inputs - 1], cfg_x).to(in_t)
            input_sig[next_inputs - 1] = a
        return input_sig, output_sig
    
    # ------------------------
    # Vectorized branch (vec_step=2),
    # ------------------------
    
    vec_step = 2 # Vectorized step size
    zero_ten = torch.zeros((), dtype=acc_t, device=input_sig.device)   # scalar 0 on the right device/dtype
    buf = _VecBufs.alloc(vec_step, in_t, fil_t, acc_t, input_sig.device)
    acc_t = _acc_dtype_main(cfg) if (cfg.mac or cfg.mixed_vec) else out_t
    # prologue
    # prologue
    for i in range((nc // 2) - 1):
        
        a = torch.zeros(vec_step, dtype=acc_t, device=input_sig.device)
        b = torch.zeros(vec_step, dtype=acc_t, device=input_sig.device)
        inp = torch.zeros(vec_step, dtype=in_t,  device=input_sig.device)
        lo  = torch.zeros(vec_step, dtype=fil_t, device=input_sig.device)
        hi  = torch.zeros(vec_step, dtype=fil_t, device=input_sig.device)
        sum_val_a = torch.zeros(1, dtype=acc_t, device=input_sig.device)
        sum_val_b = torch.zeros(1, dtype=acc_t, device=input_sig.device)

        for j in range(2 * (i + 1) - 1, 0, -vec_step):
            for k in range(vec_step):
                inp[k] = input_sig[2 * (i + 1) - j - 1 + k].to(in_t)
                lo[k]  = Lo[(nc - 1) - j + k].to(fil_t)
                hi[k]  = Hi[(nc - 1) - j + k].to(fil_t)

            a, b, pa, pb = _lane_fma_reduce(a, b, inp, lo, hi, cfg, zero_ten, vec_step)
            if cfg.mixed_vec:
                sum_val_a += pa
                sum_val_b += pb

        if cfg.mixed_vec:
            out_a = finalize_out_dtype(sum_val_a, cfg)
            out_b = finalize_out_dtype(sum_val_b, cfg)
        else:
            out_a = finalize_out_dtype(a[0] + a[1], cfg)
            out_b = finalize_out_dtype(b[0] + b[1], cfg)
        output_sig[ii + core_id]             = out_a.to(out_t)
        output_sig[ii + core_id + idx_level] = out_b.to(out_t)
        ii += num_cores

    # main body
    for i in range((nc - 1), n, 2):
        a = torch.zeros(vec_step, dtype=acc_t, device=input_sig.device)
        b = torch.zeros(vec_step, dtype=acc_t, device=input_sig.device)
        inp = torch.zeros(vec_step, dtype=in_t,  device=input_sig.device)
        lo  = torch.zeros(vec_step, dtype=fil_t, device=input_sig.device)
        hi  = torch.zeros(vec_step, dtype=fil_t, device=input_sig.device)
        sum_val_a = torch.zeros(1, dtype=acc_t, device=input_sig.device)
        sum_val_b = torch.zeros(1, dtype=acc_t, device=input_sig.device)

        for j in range(nc - 1, 0, -2):
            for k in range(vec_step):
                inp[k] = input_sig[i - j + k].to(in_t)
                lo[k]  = Lo[(nc - 1) - j + k].to(fil_t)
                hi[k]  = Hi[(nc - 1) - j + k].to(fil_t)

            a, b, pa, pb = _lane_fma_reduce(a, b, inp, lo, hi, cfg, zero_ten, vec_step)
            if cfg.mixed_vec:
                sum_val_a += pa
                sum_val_b += pb

        if nc & 0x1:
            inpk = input_sig[i - 0].to(in_t)
            lok  = Lo[nc - 1].to(fil_t)
            hik  = Hi[nc - 1].to(fil_t)
            if cfg.mixed_vec:
                sum_val_a = acc_add(sum_val_a, inpk, lok, cfg, leftover=True, in_main=False)
                sum_val_b = acc_add(sum_val_b, inpk, hik, cfg, leftover=True, in_main=False)
            else:
                a[0] = acc_add(a[0], inpk, lok, cfg, leftover=False, in_main=True)
                b[0] = acc_add(b[0], inpk, hik, cfg, leftover=False, in_main=True)
                a[0] = finalize_out_dtype(a[0], cfg)
                b[0] = finalize_out_dtype(b[0], cfg)

        if cfg.mixed_vec:
            out_a = finalize_out_dtype(sum_val_a, cfg)
            out_b = finalize_out_dtype(sum_val_b, cfg)
        else:
            out_a = finalize_out_dtype(a[0] + a[1], cfg)
            out_b = finalize_out_dtype(b[0] + b[1], cfg)

        output_sig[ii + core_id]             = out_a.to(out_t)
        output_sig[ii + core_id + idx_level] = out_b.to(out_t)
        ii += num_cores

    # even tail
    if n % 2 == 0:
        for i in range((nc // 2) - 1):
            a = torch.zeros(vec_step, dtype=acc_t, device=input_sig.device)
            b = torch.zeros(vec_step, dtype=acc_t, device=input_sig.device)
            inp = torch.zeros(vec_step, dtype=in_t,  device=input_sig.device)
            lo  = torch.zeros(vec_step, dtype=fil_t, device=input_sig.device)
            hi  = torch.zeros(vec_step, dtype=fil_t, device=input_sig.device)
            sum_val_a = torch.zeros(1, dtype=acc_t, device=input_sig.device)
            sum_val_b = torch.zeros(1, dtype=acc_t, device=input_sig.device)

            for j in range((nc - 2 * (i + 1)) - 1, 0, -vec_step):
                for k in range(vec_step):
                    inp[k] = input_sig[n - j - 1 + k].to(in_t)
                    lo[k]  = Lo[(nc - 1) - (2 * (i + 1) + j) + k].to(fil_t)
                    hi[k]  = Hi[(nc - 1) - (2 * (i + 1) + j) + k].to(fil_t)

                a, b, pa, pb = _lane_fma_reduce(a, b, inp, lo, hi, cfg, zero_ten, vec_step)
                if cfg.mixed_vec:
                    sum_val_a += pa
                    sum_val_b += pb

            if nc & 0x1:
                inpk = input_sig[i - 0].to(in_t)
                lok  = Lo[(nc - 1) - (2 * (i + 1))].to(fil_t)
                hik  = Hi[(nc - 1) - (2 * (i + 1))].to(fil_t)
                if cfg.mixed_vec:
                    sum_val_a = acc_add(sum_val_a, inpk, lok, cfg, leftover=True, in_main=False)
                    sum_val_b = acc_add(sum_val_b, inpk, hik, cfg, leftover=True, in_main=False)
                else:
                    a[0] = acc_add(a[0], inpk, lok, cfg, leftover=False, in_main=True)
                    b[0] = acc_add(b[0], inpk, hik, cfg, leftover=False, in_main=True)
                    a[0] = finalize_out_dtype(a[0], cfg)
                    b[0] = finalize_out_dtype(b[0], cfg)

            if cfg.mixed_vec:
                out_a = finalize_out_dtype(sum_val_a, cfg)
                out_b = finalize_out_dtype(sum_val_b, cfg)
            else:
                out_a = finalize_out_dtype(a[0] + a[1], cfg)
                out_b = finalize_out_dtype(b[0] + b[1], cfg)

            output_sig[ii + core_id]             = out_a.to(out_t)
            output_sig[ii + core_id + idx_level] = out_b.to(out_t)
            ii += num_cores

    # odd tail
    if n % 2 == 1:
        for i in range(nc // 2):
            a = torch.zeros(vec_step, dtype=acc_t, device=input_sig.device)
            b = torch.zeros(vec_step, dtype=acc_t, device=input_sig.device)
            inp = torch.zeros(vec_step, dtype=in_t,  device=input_sig.device)
            lo  = torch.zeros(vec_step, dtype=fil_t, device=input_sig.device)
            hi  = torch.zeros(vec_step, dtype=fil_t, device=input_sig.device)
            sum_val_a = torch.zeros(1, dtype=acc_t, device=input_sig.device)
            sum_val_b = torch.zeros(1, dtype=acc_t, device=input_sig.device)

            for j in range(nc - 2 * (i + 1), 0, -2):
                for k in range(vec_step):
                    inp[k] = input_sig[n - j - 1 + k].to(in_t)
                    lo[k]  = Lo[(nc - 1) - (2 * (i + 1) + j - 1) + k].to(fil_t)
                    hi[k]  = Hi[(nc - 1) - (2 * (i + 1) + j - 1) + k].to(fil_t)

                a, b, pa, pb = _lane_fma_reduce(a, b, inp, lo, hi, cfg, zero_ten, vec_step)
                if cfg.mixed_vec:
                    sum_val_a += pa
                    sum_val_b += pb

            if not (nc & 0x1):
                inpk = input_sig[n - 1].to(in_t)
                lok  = Lo[(nc - 1) - (2 * (i + 1) + 0 - 1)].to(fil_t)
                hik  = Hi[(nc - 1) - (2 * (i + 1) + 0 - 1)].to(fil_t)
                if cfg.mixed_vec:
                    sum_val_a = acc_add(sum_val_a, inpk, lok, cfg, leftover=True, in_main=False)
                    sum_val_b = acc_add(sum_val_b, inpk, hik, cfg, leftover=True, in_main=False)
                else:
                    a[0] = acc_add(a[0], inpk, lok, cfg, leftover=False, in_main=True)
                    b[0] = acc_add(b[0], inpk, hik, cfg, leftover=False, in_main=True)
                    a[0] = finalize_out_dtype(a[0], cfg)
                    b[0] = finalize_out_dtype(b[0], cfg)

            if cfg.mixed_vec:
                out_a = finalize_out_dtype(sum_val_a, cfg)
                out_b = finalize_out_dtype(sum_val_b, cfg)
            else:
                out_a = finalize_out_dtype(a[0] + a[1], cfg)
                out_b = finalize_out_dtype(b[0] + b[1], cfg)

            output_sig[ii + core_id]             = out_a.to(out_t)
            output_sig[ii + core_id + idx_level] = out_b.to(out_t)
            ii += num_cores

    # prepare next inputs (finalize into X policy)
    next_inputs = (n + nc - 1) // 2
    cfg_x = DWTConfig(
        mac=cfg.mac, vec=cfg.vec, cast=False, hw_mixed=cfg.hw_mixed, mixed_vec=cfg.mixed_vec,
        mantissa_bits=cfg.mantissa_bits, dt_x=cfg.dt_x, dt_f=cfg.dt_f, dt_out=cfg.dt_x, cast_to=None
    )
    for i in range(0, (next_inputs // 2) * 2, 2):
        a = finalize_out_dtype(output_sig[i],   cfg_x).to(in_t)
        b = finalize_out_dtype(output_sig[i+1], cfg_x).to(in_t)
        input_sig[i]   = a
        input_sig[i+1] = b
    if next_inputs & 0x1:
        input_sig[next_inputs - 1] = finalize_out_dtype(output_sig[next_inputs - 1], cfg_x).to(in_t)

    return input_sig, output_sig
def dwt(
    input_size: int,
    coeff: List[np.ndarray],
    levels: int,
    x: torch.Tensor,
    nc: int,
    lo: torch.Tensor,
    hi: torch.Tensor,
    cfg: DWTConfig,
) -> torch.Tensor:
    """
    Multi-level DWT driver (flow preserved). Flips filters for vector mode like before.
    """
    level_dim  = input_size
    output_dim = get_outdim(coeff)

    x_work = x.clone().detach()
    Lo = torch.flip(lo, dims=[0]) if cfg.vec else lo
    Hi = torch.flip(hi, dims=[0]) if cfg.vec else hi

    out_t = _work_dtype(cfg.dt_out)
    output_temp = torch.zeros(output_dim, dtype=out_t, device=x_work.device)

    for _ in range(levels):
        input_dim  = level_dim
        level_dim  = (level_dim + nc - 1) / 2
        level_dim  = int(level_dim)
        output_dim -= level_dim
        

        x_work, output_temp = dwt_step(
            x_work, output_temp,
            input_dim, output_dim, nc,
            Lo, Hi, cfg
        )

    # final policy round
    output_temp = finalize_out_dtype(output_temp, cfg)
    return output_temp

def load_data(
    input_size: int,
    input_signal: Union[list, np.ndarray],
    scale: Union[float, None] = None,
    method: Literal["multiplicative", "normalize", "standardize"] = "normalize",
    target_range: tuple = (0.0, 1.0),
) -> torch.Tensor:
    """
    Loads and preprocesses the input signal data.

    Args:
        input_size (int): Number of samples to load.
        input_signal (list or np.ndarray): Raw signal values.
        scale (float or None): Optional manual scaling factor.
        method (str): One of ["multiplicative", "normalize", "standardize"]. Determines scaling method.
        target_range (tuple): If `method` is "normalize", defines the (min, max) range.

    Returns:
        torch.Tensor: Preprocessed signal.
    """

    if input_size > len(input_signal):
        raise ValueError(
            f"input_size ({input_size}) exceeds input_signal length ({len(input_signal)})."
        )

    # Convert and truncate
    data = np.asarray(input_signal[:input_size]).astype(np.float32)

    # Manual scaling (e.g., scale=0.0025)
    if method == "multiplicative":
        data *= scale

    # Normalize to range [a, b]
    elif method == "normalize":
        a, b = target_range
        min_val, max_val = np.min(data), np.max(data)
        if max_val != min_val:
            data = a + ((data - min_val) * (b - a)) / (max_val - min_val)
        else:
            data[:] = (a + b) / 2  # if constant input

    # Standardize to mean=0, std=1
    elif method == "standardize":
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            data = (data - mean) / std
        else:
            data[:] = 0.0  # constant input

    return torch.from_numpy(data)


def get_filters(
    x_test: torch.Tensor, levels: int, family: str
) -> Tuple[torch.Tensor, torch.Tensor, int, List[np.ndarray]]:
    """
    Computes the Discrete Wavelet Transform (DWT) filters and coefficients.

    Args:
        x_test (torch.Tensor): Input signal as a PyTorch tensor.
        levels (int): Number of decomposition levels for the wavelet transform.
        family (str): Wavelet family/type to be used.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int, List[np.ndarray]]:
            - hi (torch.Tensor): High-pass decomposition filter.
            - lo (torch.Tensor): Low-pass decomposition filter.
            - nc (int): Length of the filters.
            - coeff (List[np.ndarray]): Wavelet decomposition coefficients.
    """
    # Initialize the specified wavelet
    wavelet = pywt.Wavelet(family)

    # Perform Discrete Wavelet Transform (DWT) using 'zero' padding mode
    coeff = pywt.wavedec(x_test.numpy(), wavelet, mode="zero", level=levels)
    
    # Print the maximum decomposition level for this wavelet and input length
    max_levels = pywt.dwt_max_level(len(x_test), wavelet.dec_len)
    print(f"Maximum levels for wavelet '{family}' and input length {len(x_test)}: {max_levels}")

    # Extract low-pass and high-pass decomposition filters
    lo_filter = wavelet.dec_lo  # Low-pass filter coefficients
    hi_filter = wavelet.dec_hi  # High-pass filter coefficients
    nc = len(lo_filter)  # Number of coefficients (filter length)

    # Convert filters to PyTorch tensors in float32 format
    lo = torch.tensor(lo_filter, dtype=torch.float32)
    hi = torch.tensor(hi_filter, dtype=torch.float32)

    return hi, lo, nc, coeff


def get_outdim(coeff: List[np.ndarray]) -> int:
    """
    Computes the output dimension by concatenating all wavelet decomposition coefficients.

    Args:
        coeff (List[np.ndarray]): List of wavelet decomposition coefficient arrays.

    Returns:
        int: The total length of the concatenated coefficient arrays.
    """
    # Concatenate all coefficient arrays into a single array
    ref = np.concatenate(coeff) if coeff else np.array([])

    # Return the total length of the concatenated coefficients
    return len(ref) 


def save_data_into_hfile(
    x_test: torch.Tensor,
    output_dim: int,
    levels: int,
    nc: int,
    output_sig: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> None:
    """
    Saves configuration and data into header files for use in a C-based project.

    Args:
        x_test (torch.Tensor): Input signal tensor.
        output_dim (int): Output signal length.
        levels (int): Number of decomposition levels.
        nc (int): Length of wavelet filter.
        output_sig (torch.Tensor): Output signal tensor.
        lo (torch.Tensor): Low-pass decomposition filter.
        hi (torch.Tensor): High-pass decomposition filter.

    Returns:
        None
    """

    # Write the configuration file (config.h)
    with open("config.h", "w") as g:
        g.write(
            f"""\
#ifndef _CONF_
#define _CONF_
#include "config.h"
#define WINDOW_LEN {x_test.shape[0]}
#define DWT_LEN {x_test.shape[0]}  // input
#define DWT_LEN_OUT {output_dim}   // output
#define LEVELS {levels}     // ((int)ceil(log2(DWT_LEN))) // 2^LEVELS = DWT_LEN if executed fully until max levels = log2(N)
#define NC {nc}

#endif
"""
        )

    # Write the input signal and reference output file (input_ch2_off.h)
    with open("input_ch2_off.h", "w") as f:
        f.write(
            """\
#ifndef INPUT_CH2_OFF_H
#define INPUT_CH2_OFF_H
#include "config.h"
"""
        )
        write_matrix(x_test, "Input_Signal", str(x_test.numel()), f, x_test.dtype)
        write_matrix(output_sig, "ref", str(output_sig.numel()), f, output_sig.dtype)
        f.write("#endif\n")

    # Flip filters for reconstruction
    lo_rec = torch.flip(lo, [0])
    hi_rec = torch.flip(hi, [0])

    # Write the filter coefficients into kernels.def
    with open("kernels.def", "w") as f:
        if nc != 2:
            f.write("#ifdef VECTORIAL\n")
            write_matrix(lo_rec, "Lo", str(nc), f, lo_rec.dtype)
            write_matrix(hi_rec, "Hi", str(nc), f, hi_rec.dtype)
            f.write("#else\n")
            write_matrix(lo, "Lo", str(nc), f, lo.dtype)
            write_matrix(hi, "Hi", str(nc), f, hi.dtype)
            f.write("#endif\n")
        else:
            f.write(
                """\
DATA_LOCATION FIL_TYPE R2_2 = 0.70710678118654752440f; // wfilters in Matlab
"""
            )

# ----------------------------
# CLI
# ----------------------------

def get_initial_config():
    parser = argparse.ArgumentParser(description="DWT kernel (cfg-based)")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--mode", type=str, default="sym4")
    parser.add_argument("--mac_flag", default=True)
    parser.add_argument("--vec_flag", default=False)
    parser.add_argument("--hwmixed_flag", default=False)
    parser.add_argument("--float_type", type=str, default="FP32")
    parser.add_argument("--mantissa_bits", type=int, default=2)
    parser.add_argument("--exploration_flag", default=False)

    parser.add_argument("--scaling_method", type=str, default="normalize",
                        choices=["multiplicative", "normalize", "standardize"])
    parser.add_argument("--scale_value", type=float, default=0.0025)
    parser.add_argument("--target_range", type=str, default="0,1")
    args = parser.parse_args()

    input_size = int(args.input_size)
    levels = int(args.levels)
    family = str(args.mode)

    mac_flag = str2bool(args.mac_flag)
    vec_flag = str2bool(args.vec_flag)
    hwmixed_flag = str2bool(args.hwmixed_flag)
    exploration_flag = str2bool(args.exploration_flag)

    bits = [s.strip() for s in args.float_type.split(",")]
    mantissa_bits = int(args.mantissa_bits)

    scaling_method = args.scaling_method
    scale_value = float(args.scale_value)
    target_range = tuple(map(float, args.target_range.split(",")))

    return (
        input_size,
        levels,
        family,
        bits,
        mac_flag,
        vec_flag,
        exploration_flag,
        mantissa_bits,
        scaling_method,
        scale_value,
        target_range,
        hwmixed_flag,
    )

def main():
    """
    Main function to perform Discrete Wavelet Transform (DWT) processing.

    Steps:
    1. Reads configuration settings from the command-line.
    2. Loads input signal data.
    3. Retrieves wavelet filters and decomposition coefficients.
    4. Performs DWT in FP32 as a reference.
    5. Sets data types according to the configuration.
    6. Initializes matrices and executes DWT in the selected precision.
    7. Computes the error between FP32 and the selected precision.
    8. Saves processed data into header files for embedded system integration.

    Returns:
        None
    """
    # ----------------------------
# Main
# ----------------------------

def main():
    (
        input_size,
        levels,
        family,
        bits,
        mac_flag,
        vec_flag,
        exploration_flag,
        mantissa_bits,
        scaling_method,
        scale_value,
        target_range,
        hwmixed_flag,
    ) = get_initial_config()

    # sanity/info messages (like FIR)
    if not exploration_flag:
        check_pulp_warnings(bits, mac_flag, hwmixed_flag, vec_flag)

    # data
    x_test = load_data(
        input_size,
        input_signal,
        scale=scale_value,
        method=scaling_method,
        target_range=target_range,
    )
    # filters + reference coeffs
    hi, lo, nc, coeff = get_filters(x_test, levels, family)
    if not exploration_flag:
        print(f"Running FP32 reference: N={input_size}, L={levels}, wavelet={family}")
    # FP32 reference (flags off)
    cfg_ref = DWTConfig(
        mac=False,
        vec=False,
        cast=False,
        hw_mixed=False,
        mixed_vec=False,
        mantissa_bits=0,
        dt_x=DKind.FP32,
        dt_f=DKind.FP32,
        dt_out=DKind.FP32,
        cast_to=None,
    )
    # Perform DWT in FP32 as a reference
    out_ref = dwt(
        input_size, coeff, levels,
        x_test, nc, lo, hi,
        cfg_ref
    )
# resolved dtypes
    datatypes: List[DKind] = select_dtypes(bits, 3)   # [X, F, OUT]
    cast_flag = check_cast(datatypes)
    cast_to = _to_dkind(bits[-1]) if cast_flag else None
    mixed_vec_flag = check_vec_flag(datatypes, vec_flag)

    if not exploration_flag:
        print(f"Mixed Vectorization Flag: {mixed_vec_flag}")

    if not exploration_flag:
        if DKind.FP8_CUSTOM in datatypes:
            print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
            print(f"and mantissa = {mantissa_bits} bits")
        else:
            print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
        if cast_flag:
            print(f"Casting enabled -> {dkind_name(cast_to) if cast_to else 'None'}")

    # exploration folder tag
    if scaling_method == "multiplicative":
        scale_tag = f"scale_{scale_value}"
    elif scaling_method == "normalize":
        scale_tag = f"{target_range[0]}_{target_range[1]}"
    else:
        scale_tag = "minmax_0_mean0_std1"

    out_dir = os.path.join(
        os.getcwd(), "exploration", scaling_method, scale_tag, f"mode_{family}", f"levels_{levels}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # init tensors in declared input/filter kinds (via helper)
    x_typed  = matrix_init(x_test, datatypes[0], mantissa_bits)
    lo_typed = matrix_init(lo,     datatypes[1], mantissa_bits)
    hi_typed = matrix_init(hi,     datatypes[1], mantissa_bits)

    # cfg for run
    cfg_run = DWTConfig(
        mac=mac_flag,
        vec=vec_flag,
        cast=cast_flag,
        hw_mixed=hwmixed_flag,
        mixed_vec=mixed_vec_flag,
        mantissa_bits=mantissa_bits,
        dt_x=datatypes[0],
        dt_f=datatypes[1],
        dt_out=datatypes[2],
        cast_to=cast_to,
    )

    # run DWT
    out_sig = dwt(
        input_size, coeff, levels,
        x_typed, nc, lo_typed, hi_typed,
        cfg_run
    )

    # metrics
    dtype_tag = f"{dkind_name(datatypes[0])}_{dkind_name(datatypes[1])}_{dkind_name(datatypes[2])}"
    output_file = (
        os.path.join(
            out_dir,
            f"error_metric__{input_size}_{dtype_tag}_{mantissa_bits}.txt",
        )
        if exploration_flag
        else None
    )
    error_metric(out_ref, out_sig, output_file)

    # headers
    if not exploration_flag:
        save_data_into_hfile(x_typed, get_outdim(coeff), levels, nc, out_sig, lo_typed, hi_typed)


if __name__ == "__main__":
    main()