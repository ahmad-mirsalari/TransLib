#!/bin/python3
import sys
import os
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple, TextIO

import numpy as np
import torch

# repo root: translib_jr (three levels up from this file)
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Project helpers (same as FIR/DWT)
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
    check_vec_flag,
    check_cast,
    matrix_init_like as matrix_init,
    str2bool,
    error_metric,
)


# -----------------------------------------------------------------------------
# Config & small utilities
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FFTConfig:
    # flags
    mac: bool
    vec: bool
    cast: bool
    hw_mixed: bool
    mixed_vec: bool
    mantissa_bits: int

    # dtypes
    dt_x: DKind    # signal dtype
    dt_t: DKind    # twiddle dtype
    dt_out: DKind  # working/output dtype policy

    # optional cast target
    cast_to: Optional[DKind] = None


def _work_dtype(kind: DKind) -> torch.dtype:
    """Torch dtype used in buffers; FP8_CUSTOM keeps float32 storage with policy rounding."""
    return torch.float32 if (kind is DKind.FP8_CUSTOM) else kind.to_torch()


def _maybe_cast(x: torch.Tensor, cfg: FFTConfig) -> torch.Tensor:
    """Front-end cast (FIR-style) if requested and not using hw_mixed path."""
    if cfg.cast and (cfg.cast_to is not None) and not cfg.hw_mixed and not cfg.mixed_vec:
        return _cast_apply(x, cfg.cast_to, cfg.mantissa_bits)
    return x

def is_power_of_two(x: int) -> bool:
    """
    Checks if a given integer x is a power of 2.

    Args:
        x (int): The input integer.

    Returns:
        bool: True if x is a power of 2, False otherwise.
    """
    return x > 0 and (x & (x - 1)) == 0

# -----------------------------------------------------------------------------
# I/O helpers (complex writer)
# -----------------------------------------------------------------------------


def write_matrix(
    matrix_to_write_r: torch.Tensor,
    matrix_to_write_i: torch.Tensor,
    name: str,
    length: str,
    file_pointer: TextIO,
    float_type: torch.dtype,
) -> None:
    """
    Write complex array as { {r,i}, ... } into C header.
    """
    if "ref" in name:
        file_pointer.write(f"PI_L2 Complex_type {name}[{length}] = {{\n")
    else:
        file_pointer.write(f"DATA_LOCATION Complex_type {name}[{length}] = {{\n")

    float_removals = {
        torch.float32: ")",
        torch.float16: ", dtype=torch.float16)",
        torch.bfloat16: ", dtype=torch.bfloat16)",
    }
    rem_part = float_removals.get(float_type, ")")

    assert matrix_to_write_r.shape[0] == matrix_to_write_i.shape[0], "Real/Imag lengths must match"

    for real, imag in zip(matrix_to_write_r, matrix_to_write_i):
        rs = str(real.item()).replace("tensor(", "").replace(rem_part, "")
        is_ = str(imag.item()).replace("tensor(", "").replace(rem_part, "")
        file_pointer.write(f"    {{{rs}, {is_}}},\n")

    file_pointer.write("};\n")
    
# -----------------------------------------------------------------------------
# FFT core (radix-2 DIF) + reordering
# -----------------------------------------------------------------------------

def bracewell_buneman(xarray: torch.Tensor, length: int, log2length: int) -> torch.Tensor:
    """
    Bracewell-Buneman bit reversal in-place.
    """
    muplus = int((log2length + 1) // 2)
    mvar = 1
    reverse = torch.zeros(length, dtype=torch.int64, device=xarray.device)
    upper_range = muplus + 1
    for _ in torch.arange(1, upper_range, device=xarray.device):
        for kvar in torch.arange(0, mvar, device=xarray.device):
            tvar = 2 * reverse[kvar]
            reverse[kvar] = tvar
            reverse[kvar + mvar] = tvar + 1
        mvar = mvar + mvar
    if log2length & 0x01:
        mvar = mvar // 2
    for qvar in torch.arange(1, mvar, device=xarray.device):
        nprime = qvar - mvar
        rprimeprime = reverse[qvar] * mvar
        for pvar in torch.arange(0, reverse[qvar], device=xarray.device):
            nprime = nprime + mvar
            rprime = rprimeprime + reverse[pvar]
            tmp = xarray[nprime].clone()
            xarray[nprime] = xarray[rprime]
            xarray[rprime] = tmp
    return xarray



def dif_fft0_cfg(
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
    t_real: torch.Tensor,
    t_imag: torch.Tensor,
    log2length: int,
    cfg: FFTConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Radix-2 DIF FFT (flow preserved). Numerical policy via cfg:
    - Inputs read in dt_x space; twiddles in dt_t.
    - Accumulator promotion via _acc_dtype_main(cfg).
    - Rounding/quant via finalize_out_dtype.
    - Front-end cast via _cast_apply if cfg.cast.
    """
    # working dtypes
    in_t   = _work_dtype(cfg.dt_x)
    twid_t = _work_dtype(cfg.dt_t)
    out_t  = _work_dtype(cfg.dt_out)
    acc_t  = _acc_dtype_main(cfg) if (cfg.mac or cfg.hw_mixed) else out_t

    # clone to working buffers in OUT space (keep same device)
    xarray_real = x_real.to(out_t).clone()
    xarray_imag = x_imag.to(out_t).clone()

    twiddle_real = t_real.to(twid_t)
    twiddle_imag = t_imag.to(twid_t)

    b_p = 1
    nvar_p = xarray_real.shape[0]
    twiddle_step_size = 1

    for _ in range(log2length):  # pass loop
        nvar_pp = nvar_p // 2
        base_e = 0

        for _ in range(b_p):  # block loop
            base_o = base_e + nvar_pp
            for nvar in range(nvar_pp):  # butterfly loop
                # e = x_e + x_o (sum path)
                er = xarray_real[base_e + nvar] + xarray_real[base_o + nvar]
                ei = xarray_imag[base_e + nvar] + xarray_imag[base_o + nvar]
                if not cfg.hw_mixed:
                    er = finalize_out_dtype(er, cfg)
                    ei = finalize_out_dtype(ei, cfg)

                # o = (x_e - x_o) * W_k for nvar>0; for nvar==0, W=1
                dr = xarray_real[base_e + nvar] - xarray_real[base_o + nvar]
                di = xarray_imag[base_e + nvar] - xarray_imag[base_o + nvar]
                if not cfg.hw_mixed:
                    dr = finalize_out_dtype(dr, cfg)
                    di = finalize_out_dtype(di, cfg)

                if nvar == 0:
                    # W = 1 + j0
                    or_ = dr
                    oi  = di
                else:
                    # twiddle index
                    tf = nvar * twiddle_step_size

                    tr = twiddle_real[tf]
                    ti = twiddle_imag[tf]

                    # front-end cast if enabled
                    dr = _maybe_cast(dr.to(in_t), cfg)
                    di = _maybe_cast(di.to(in_t), cfg)
                    tr = _maybe_cast(tr.to(twid_t), cfg)
                    ti = _maybe_cast(ti.to(twid_t), cfg)

                    # complex multiply in accumulator space
                    # or = dr*tr - di*ti
                    # oi = di*tr + dr*ti
                    or_acc = torch.zeros(1, dtype=acc_t, device=x_real.device)
                    oi_acc = torch.zeros(1, dtype=acc_t, device=x_real.device)

                    # or_acc += dr*tr
                    or_acc = acc_add(or_acc, dr, tr, cfg, leftover=False, in_main=True)
                    if not cfg.hw_mixed:
                        or_acc = finalize_out_dtype(or_acc, cfg)

                    # or_acc += (-di)*ti
                    m_di = -di
                    or_acc = acc_add(or_acc, m_di, ti, cfg, leftover=False, in_main=True)
                    if cfg.mac and cfg.hw_mixed:
                        or_acc = finalize_out_dtype(or_acc, cfg)
                    elif not cfg.hw_mixed:
                        or_acc = finalize_out_dtype(or_acc, cfg)

                    # oi_acc += di*tr
                    oi_acc = acc_add(oi_acc, di, tr, cfg, leftover=False, in_main=True)
                    if not cfg.hw_mixed:
                        oi_acc = finalize_out_dtype(oi_acc, cfg)

                    # oi_acc += dr*ti
                    oi_acc = acc_add(oi_acc, dr, ti, cfg, leftover=False, in_main=True)
                    if cfg.mac and cfg.hw_mixed:
                        oi_acc = finalize_out_dtype(oi_acc, cfg)
                    elif not cfg.hw_mixed:
                        oi_acc = finalize_out_dtype(oi_acc, cfg)

                    or_ = or_acc.squeeze(0)
                    oi  = oi_acc.squeeze(0)

                # store back
                xarray_real[base_e + nvar] = er.to(out_t)
                xarray_imag[base_e + nvar] = ei.to(out_t)
                xarray_real[base_o + nvar] = or_.to(out_t)
                xarray_imag[base_o + nvar] = oi.to(out_t)

            base_e = base_e + nvar_p

        b_p = b_p * 2
        nvar_p = nvar_p // 2
        twiddle_step_size = 2 * twiddle_step_size

    # final bit-reversal (Bracewell-Buneman)
    xarray_real = bracewell_buneman(xarray_real, xarray_real.shape[0], log2length)
    xarray_imag = bracewell_buneman(xarray_imag, xarray_imag.shape[0], log2length)

    # final policy round
    if not cfg.hw_mixed:
        xarray_real = finalize_out_dtype(xarray_real, cfg)
        xarray_imag = finalize_out_dtype(xarray_imag, cfg)

    return xarray_real, xarray_imag

# -----------------------------------------------------------------------------
# Twiddles & data
# -----------------------------------------------------------------------------

def get_twiddle(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the twiddle factors used in FFT computations.

    Args:
        n (int): Length of the signal (must be a power of 2 for FFT).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Twiddle_real (torch.Tensor): Real part of the twiddle factors.
            - Twiddle_imag (torch.Tensor): Imaginary part of the twiddle factors.
    """
    # Generate twiddle factors using the exponential form of the FFT basis
    twiddle = torch.exp(-2j * torch.pi * torch.arange(0, 0.5, 1.0 / n))

    # Extract real and imaginary parts
    twiddle_real = twiddle.real.to(torch.float32)
    twiddle_imag = twiddle.imag.to(torch.float32)

    return twiddle_real, twiddle_imag


def save_data_into_hfile(
    xtest_real: torch.Tensor,
    xtest_imag: torch.Tensor,
    output_real: torch.Tensor,
    output_imag: torch.Tensor,
) -> None:
    """
    Saves FFT input and output data into header files for embedded system integration.

    Args:
        xtest_real (torch.Tensor): Real part of the input signal.
        xtest_imag (torch.Tensor): Imaginary part of the input signal.
        output_real (torch.Tensor): Real part of the FFT output.
        output_imag (torch.Tensor): Imaginary part of the FFT output.

    Returns:
        None
    """
    # Write FFT input data to 'data_signal.h'
    with open("data_signal.h", "w") as g:
        g.write(
            """\
#ifndef FFT_DATA_H
#define FFT_DATA_H
#ifdef FABRIC
#define DATA_LOCATION
#else
#define DATA_LOCATION __attribute__((section(".data_l1")))
#endif

"""
        )
        # Save input matrix
        write_matrix(
            xtest_real,
            xtest_imag,
            "Input_Signal",
            str(xtest_real.numel()),
            g,
            xtest_real.dtype,
        )

        # Add output buffer definition
        g.write(
            """
#ifndef SORT_OUTPUT
DATA_LOCATION Complex_type Buffer_Signal_Out[FFT_LEN_RADIX2];
#endif
#endif
"""
        )

    # Write FFT output reference data to 'data_out.h'
    with open("data_out.h", "w") as f:
        write_matrix(
            output_real,
            output_imag,
            "ref",
            str(output_real.numel()),
            f,
            output_real.dtype,
        )

    # Write FFT configuration to 'config.h'
    fft_log2_len = int(np.log2(len(xtest_real)))
    with open("config.h", "w") as h:
        h.write(f"#define LOG2_FFT_LEN {fft_log2_len}\n")

    # Create directory if it doesn't exist and save the same config in '../fft_radix8/config.h'
    os.makedirs("../fft_radix8", exist_ok=True)
    with open("../fft_radix8/config.h", "w") as h:
        h.write(f"#define LOG2_FFT_LEN {fft_log2_len}\n")


def load_data(
    n: int,
    scaling_method: str = "normalize",
    scale_value: float = 0.0025,
    target_range: tuple = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same waveform/scaling as your original (sine, optional scaling/normalize/standardize).
    """
    time = np.linspace(0.0, (n / 20) * np.pi, n)
    amplitude = np.sin(time).astype(np.float32)

    if scaling_method == "multiplicative":
        amplitude *= scale_value
    elif scaling_method == "normalize":
        a, b = target_range
        mn, mx = float(np.min(amplitude)), float(np.max(amplitude))
        if mx != mn:
            amplitude = a + (amplitude - mn) * (b - a) / (mx - mn)
        else:
            amplitude[:] = (a + b) / 2.0
    elif scaling_method == "standardize":
        mean = float(np.mean(amplitude))
        std = float(np.std(amplitude))
        amplitude = (amplitude - mean) / std if std > 0 else amplitude * 0.0

    xr = torch.tensor(amplitude, dtype=torch.float32)
    xi = torch.zeros(n, dtype=torch.float32)
    return xr, xi

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def get_initial_config():
    parser = argparse.ArgumentParser(description="FFT (radix-2 DIF) cfg-based")
    parser.add_argument("--input_size", type=int, default=2048)
    parser.add_argument("--mac_flag", default=False)
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
    mac_flag = str2bool(args.mac_flag)
    vec_flag = str2bool(args.vec_flag)    # accepted but not used for FFT lanes in this version
    hwmixed_flag = str2bool(args.hwmixed_flag)
    exploration_flag = str2bool(args.exploration_flag)

    bits = [s.strip() for s in args.float_type.split(",")]
    mantissa_bits = int(args.mantissa_bits)

    scaling_method = args.scaling_method
    scale_value = float(args.scale_value)
    target_range = tuple(map(float, args.target_range.split(",")))

    if not is_power_of_two(input_size):
        raise SystemExit("Error: input_size must be a power of 2.")

    return (
        input_size,
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    (
        input_size,
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

    if not exploration_flag:
        check_pulp_warnings(bits, mac_flag, hwmixed_flag, vec_flag)

    # data + twiddles
    x_test_real, x_test_imag = load_data(input_size, scaling_method, scale_value, target_range)
    twiddle_real, twiddle_imag = get_twiddle(input_size)

    # reference (FP32, flags off)
    cfg_ref = FFTConfig(
        mac=False, vec=False, cast=False, hw_mixed=False, mixed_vec=False,
        mantissa_bits=0,
        dt_x=DKind.FP32, dt_t=DKind.FP32, dt_out=DKind.FP32, cast_to=None
    )
    ref_real, ref_imag = dif_fft0_cfg(
        x_test_real, x_test_imag, twiddle_real, twiddle_imag,
        int(np.log2(input_size)), cfg_ref
    )

    # resolve dtypes & flags (same as FIR/DWT)
    datatypes: List[DKind] = select_dtypes(bits, 3)  # [X, T, OUT]
    cast_flag = check_cast(datatypes)
    cast_to = _to_dkind(bits[-1]) if cast_flag else None
    mixed_vec_flag = check_vec_flag(datatypes, vec_flag)  # not used by this FFT kernel, but we keep parity

    if not exploration_flag:
        if DKind.FP8_CUSTOM in datatypes:
            print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
            print(f"and mantissa = {mantissa_bits} bits")
        else:
            print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
        if cast_flag:
            if mac_flag:
                print(f"Casting enabled -> {dkind_name(cast_to)}")
            else:
                print(f"Note: casting to {dkind_name(cast_to)} but MAC is off (on PULP you may need MAC for cast paths).")

    # init typed inputs
    xtest_real = matrix_init(x_test_real, datatypes[0], mantissa_bits)
    xtest_imag = torch.zeros_like(xtest_real)

    twid_real = matrix_init(twiddle_real, datatypes[1], mantissa_bits)
    twid_imag = matrix_init(twiddle_imag, datatypes[1], mantissa_bits)

    # run cfg
    cfg_run = FFTConfig(
        mac=mac_flag, vec=vec_flag, cast=cast_flag, hw_mixed=hwmixed_flag, mixed_vec=mixed_vec_flag,
        mantissa_bits=mantissa_bits,
        dt_x=datatypes[0], dt_t=datatypes[1], dt_out=datatypes[2], cast_to=cast_to
    )
    out_real, out_imag = dif_fft0_cfg(
        xtest_real, xtest_imag, twid_real, twid_imag,
        int(np.log2(input_size)), cfg_run
    )

    # exploration path (dump metrics)
    if scaling_method == "multiplicative":
        scale_tag = f"scale_{scale_value}"
    elif scaling_method == "normalize":
        scale_tag = f"{target_range[0]}_{target_range[1]}"
    else:
        scale_tag = "minmax_0_mean0_std1"

    out_dir = os.path.join(os.getcwd(), "exploration", scaling_method, scale_tag)
    os.makedirs(out_dir, exist_ok=True)

    if exploration_flag:
        dtype_tag = f"{dkind_name(datatypes[0])}_{dkind_name(datatypes[1])}_{dkind_name(datatypes[2])}"
        f_real = os.path.join(out_dir, f"error_metric__{input_size}_{dtype_tag}_{mantissa_bits}_real.txt")
        f_imag = os.path.join(out_dir, f"error_metric__{input_size}_{dtype_tag}_{mantissa_bits}_imag.txt")
        error_metric(ref_real, out_real, f_real)
        error_metric(ref_imag, out_imag, f_imag)
    else:
        print("############################## Error Metrics #############################")
        print("Real part:")
        error_metric(ref_real, out_real, None)
        print("Imag part:")
        error_metric(ref_imag, out_imag, None)

        # headers
        save_data_into_hfile(xtest_real, xtest_imag, out_real, out_imag)
        print("############################## Done! ###################################")


if __name__ == "__main__":
    main()