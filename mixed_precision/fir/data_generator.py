#!/bin/python3
import sys
import os
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, TextIO
import torch

# repo root: translib_jr (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.helper_functions import (
    DKind,
    _to_dkind,
    dkind_name,
    _cast_apply,
    promote_for_mac,
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


# ----------------------------
# Type system & configuration
# ----------------------------
@dataclass(frozen=True)
class ConvConfig:
    """Configuration for convolution operations."""

    # flags
    mac: bool
    vec: bool
    cast: bool
    hw_mixed: bool
    mixed_vec: bool
    mantissa_bits: int
    reversed_filter: bool
    streaming: bool

    # types
    dt_x: DKind
    dt_f: DKind
    dt_out: DKind
    cast_to: Optional[DKind]  # None if no casting destination


# ----------------------------
# Core kernel
# ----------------------------


def convolve(
    xs: torch.Tensor,
    fs: torch.Tensor,
    dt: List[DKind],
    outlen: int,
    mac_flag: bool = False,
    vec_flag: bool = False,
    cast_flag: bool = False,
    cast_to: Optional[str] = "FP32",
    mantissa_bits: int = 2,
    hwmixed_flag: bool = False,
    mixed_vec_flag: bool = False,
    streaming: bool = False,
) -> torch.Tensor:
    """
    Convolution y[i] = sum_j xs[i+j] * fs[M-1-j]
    Supports FP8_CUSTOM/BF16/FP16/FP32 casting & MAC promotion similar to matmul.
    """

    if xs.dim() != 1 or fs.dim() != 1:
        raise ValueError("xs and fs must be 1D tensors")

    cast_to_kind = (
        None
        if (cast_to is None or str(cast_to).lower() in ("false", "none", ""))
        else _to_dkind(cast_to)
    )
    cfg = ConvConfig(
        mac=mac_flag,
        vec=vec_flag,
        cast=cast_flag,
        hw_mixed=hwmixed_flag,
        mixed_vec=mixed_vec_flag,
        mantissa_bits=mantissa_bits,
        reversed_filter=False,
        streaming=streaming,
        dt_x=dt[0],
        dt_f=dt[1],
        dt_out=dt[2],
        cast_to=cast_to_kind,
    )

    # output/init
    out_dtype = (
        torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()
    )
    rs = torch.zeros(outlen, dtype=out_dtype, device=xs.device)

    # scalar path
    if not cfg.vec:
        # main-loop accumulator dtype policy
        acc_main_dtype = _acc_dtype_main(cfg)
        limit = 1 if cfg.streaming else (len(xs) - len(fs))
        for i in range(limit):
            temp = torch.zeros(
                1,
                dtype=(
                    torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else acc_main_dtype
                ),
                device=xs.device,
            )

            for j in range(len(fs)):
                a = xs[i + j]
                b = fs[len(fs) - 1 - j]

                if not ((cfg.mac and not cfg.cast) or cfg.hw_mixed):
                    if cfg.cast and cfg.cast_to is not None:
                        a = _cast_apply(a, cfg.cast_to, cfg.mantissa_bits)
                        b = _cast_apply(b, cfg.cast_to, cfg.mantissa_bits)

                # Perform multiplication
                temp = acc_add(temp, a, b, cfg, leftover=False, in_main=True)

                if not cfg.hw_mixed:
                    temp = finalize_out_dtype(temp, cfg)

            if cfg.hw_mixed:
                temp = finalize_out_dtype(temp, cfg)

            rs[i] = temp
        return rs

    # vectorized path
    vec_step = 4 if cfg.dt_x is DKind.FP8_CUSTOM else 2
    remainder = len(fs) % vec_step
    processed = len(fs) - remainder

    a = torch.zeros(
        vec_step,
        dtype=torch.float32 if cfg.dt_x is DKind.FP8_CUSTOM else cfg.dt_x.to_torch(),
        device=xs.device,
    )
    b = torch.zeros(
        vec_step,
        dtype=torch.float32 if cfg.dt_f is DKind.FP8_CUSTOM else cfg.dt_f.to_torch(),
        device=xs.device,
    )

    # main-loop accumulator dtype policy
    acc_main_dtype = _acc_dtype_main(cfg)
    acc_dtype = _acc_dtype_main(cfg)              # your accumulator dtype chooser
    zero_ten = torch.zeros((), dtype=acc_dtype, device=xs.device)   # scalar 0 on the right device/dtype

    limit = 1 if cfg.streaming else (len(xs) - len(fs))
    for i in range(limit):

        temp = torch.zeros(
            vec_step,
            dtype=torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else acc_main_dtype,
            device=xs.device,
        )
        sum_val = torch.zeros(1, dtype=acc_main_dtype, device=xs.device)
        acc = torch.zeros(1, dtype=acc_main_dtype, device=xs.device)

        for j in range(0, processed, vec_step):
            for k in range(vec_step):
                a[k] = xs[i + j + k]
                b[k] = fs[len(fs) - 1 - j - k]

            for k in range(vec_step):  # Different summation in the C code
                if cfg.mixed_vec:
                    temp[k] = acc_add(zero_ten, a[k], b[k], cfg, leftover=False, in_main=True)
                else:
                    tk = temp[k].unsqueeze(0)
                    tk = acc_add(tk, a[k], b[k], cfg, leftover=False, in_main=True)
                    tk = finalize_out_dtype(tk, cfg)
                    temp[k] = tk.squeeze(0)

            if cfg.mixed_vec:
                if vec_step == 2:
                    temp[0] += temp[1]
                else:
                    temp[0] += temp[1]
                    temp[2] += temp[3]
                    temp[0] += temp[2]
                sum_val += temp[0]
            # else:
            #     temp = finalize_out_dtype(temp, cfg)

        # leftovers
        if remainder > 0:
            for j in range(remainder):
                idx = processed + j
                a0 = xs[i + idx]
                b0 = fs[len(fs) - 1 - idx]

                if cfg.mixed_vec:
                    # (use acc_add so product-side quantize/FMA rules apply)
                    sum_val = acc_add(
                        sum_val, a0, b0, cfg, leftover=True, in_main=False
                    )
                else:
                    t0 = temp[0].unsqueeze(0)
                    t0 = acc_add(t0, a0, b0, cfg, leftover=True, in_main=False)
                    # t0 = finalize_out_dtype(t0, cfg)
                    temp[0] = t0.squeeze(0)
                    temp[0] = finalize_out_dtype(temp[0], cfg)

        if cfg.mixed_vec:
            rs[i] = finalize_out_dtype(sum_val, cfg)
        else:
            for k in range(vec_step):
                acc += temp[k]
                acc = finalize_out_dtype(acc, cfg)
            rs[i] = acc

    return rs


# ----------------------------
# I/O utilities
# ----------------------------


def write_matrix(
    matrix_to_write: torch.Tensor,
    name: str,
    length: int,
    file_pointer: TextIO,
) -> None:
    if "Buffer0" in name:
        file_pointer.write(f"DATA_LOCATION OUT_TYPE {name}[{length}];\n")
        return
    if "check" in name:
        file_pointer.write(f"PI_L2 OUT_TYPE {name}[] = {{")
    elif "UnitImpulse" in name:
        file_pointer.write(f"DATA_LOCATION INP_TYPE {name}[{length}] = {{")
    elif "input_data" in name:
        file_pointer.write(f"DATA_LOCATION dataType {name}[{length}] = {{")
    elif "reference" in name:
        file_pointer.write(f"DATA_LOCATION float {name}[{length}] = {{")
    else:
        file_pointer.write(f"DATA_LOCATION FIL_TYPE {name}[{length}] = {{")

    vals = []
    for i in range(matrix_to_write.size(0)):
        vals.append(str(matrix_to_write[i].item()))
    file_pointer.write(", ".join(vals))
    file_pointer.write("};\n")


def save_data_into_hfile(
    length: int,
    order: int,
    res: torch.Tensor,
    filter_conv: torch.Tensor,
    input_conv: torch.Tensor,
    reversed_flag: bool = False,
) -> None:
    with open("data.h", "w", encoding="utf-8") as f:
        f.write('#include "config.h"\n\n')
        f.write(f"#define LENGTH {length}\n")
        f.write(f"#define ORDER {order}\n\n")
        write_matrix(res, "Buffer0", "LENGTH-ORDER", f)
        write_matrix(filter_conv, "Filter0", "ORDER", f)
        write_matrix(input_conv, "UnitImpulse", "LENGTH", f)
        write_matrix(res, "check", "", f)

    with open("flags.h", "w", encoding="utf-8") as f:
        if reversed_flag:
            f.write("\n#define REVERSED\n")


# ----------------------------
# CLI parsing & policies
# ----------------------------


def get_initial_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", default=512)
    parser.add_argument("--order", default=100)
    parser.add_argument("--std", default=1)
    parser.add_argument("--mac_flag", default=True)
    parser.add_argument("--vec_flag", default=False)
    parser.add_argument("--float_type", default="FP32")
    parser.add_argument("--reversed", default=False)
    parser.add_argument("--hwmixed_flag", default=False)
    parser.add_argument("--mantissa_bits", default=2)
    parser.add_argument("--exploration_flag", default=False)
    parser.add_argument("--streaming", default=False)
    args = parser.parse_args()

    length = int(args.length)
    order = int(args.order)
    std = float(args.std)
    outlen = length - order

    mac_flag = str2bool(args.mac_flag)
    vec_flag = str2bool(args.vec_flag)
    exploration_flag = str2bool(args.exploration_flag)
    reversed_flag = str2bool(args.reversed)
    hwmixed_flag = str2bool(args.hwmixed_flag)
    streaming = str2bool(args.streaming)

    bits = [s.strip() for s in args.float_type.split(",")]
    mantissa_bits = int(args.mantissa_bits)

    return (
        length,
        order,
        std,
        outlen,
        bits,
        mac_flag,
        vec_flag,
        exploration_flag,
        mantissa_bits,
        reversed_flag,
        hwmixed_flag,
        streaming,
    )


# ----------------------------
# Main
# ----------------------------


def main():
    (
        length,
        order,
        std,
        outlen,
        bits,
        mac_flag,
        vec_flag,
        exploration_flag,
        mantissa_bits,
        reversed_flag,
        hwmixed_flag,
        streaming,
    ) = get_initial_config()

    # only warn in non-exploration (mirrors your matmul policy)
    if not exploration_flag:
        check_pulp_warnings(bits, mac_flag, hwmixed_flag, vec_flag)

    # dtypes & derived flags
    datatypes = select_dtypes(bits, 3)  # [X, F, OUT]
    mixed_vec_flag = check_vec_flag(datatypes, vec_flag)

    if not exploration_flag:
        print(f"Mixed Vectorization Flag: {mixed_vec_flag}")

    # inputs (keep 1D consistently)
    if exploration_flag:
        mean = 0.0
        input_ref = torch.normal(mean, std, (length,))
        filter_ref = torch.normal(mean, std, (order,))
    else:
        input_ref = torch.randn(length, dtype=torch.float32)
        filter_ref = torch.randn(order, dtype=torch.float32)

    # reference FP32 conv (flags off except MAC same as user to mirror policy in matmul)
    ref = convolve(
        xs=input_ref,
        fs=filter_ref,
        dt=[DKind.FP32, DKind.FP32, DKind.FP32],
        outlen=outlen,
        mac_flag=False,
        vec_flag=False,
        cast_flag=False,
        cast_to=None,
        mantissa_bits=0,
        hwmixed_flag=False,
        mixed_vec_flag=False,
        streaming=False,
    )

    cast_flag = check_cast(datatypes)
    if not exploration_flag:
        if DKind.FP8_CUSTOM in datatypes:
            print(
                f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}"
            )
            print(f"and mantissa = {mantissa_bits} bits")
        else:
            print(
                f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}"
            )

    # choose cast_to for prints (same heuristic as matmul-era conv)
    cast_to = bits[-1] if len(bits) else "FP32"
    if not exploration_flag and cast_flag:
        if mac_flag:
            print(f"Running with casting to {cast_to}")
        else:
            warnings.warn(
                f"Running with casting to {cast_to}. On PULP, you may need to true the MAC flag."
            )

    # exploration out folder
    output_folder = os.path.join(os.getcwd(), "exploration", str(std))
    os.makedirs(output_folder, exist_ok=True)

    # init to declared input kinds
    input_res = matrix_init(input_ref, datatypes[0], mantissa_bits)
    filter_res = matrix_init(filter_ref, datatypes[1], mantissa_bits)

    # run conv
    res = convolve(
        xs=input_res,
        fs=filter_res,
        dt=datatypes,
        outlen=outlen,
        mac_flag=mac_flag,
        vec_flag=vec_flag,
        cast_flag=cast_flag,
        cast_to=cast_to,
        mantissa_bits=mantissa_bits,
        hwmixed_flag=hwmixed_flag,
        mixed_vec_flag=mixed_vec_flag,
        streaming=streaming,
    )

    # metrics dump (use resolved dtype names)
    dtype_tag = f"{dkind_name(datatypes[0])}_{dkind_name(datatypes[1])}_{dkind_name(datatypes[2])}"
    output_file = (
        os.path.join(
            output_folder,
            f"error_metric__{length}__{order}_{dtype_tag}_{mantissa_bits}_{std}.txt",
        )
        if exploration_flag
        else None
    )
    error_metric(ref, res, output_file)

    # header emission
    if not exploration_flag:
        if reversed_flag:
            if vec_flag:
                print("Reversing the filter")
                filter_res = torch.flip(filter_res, dims=[0])
            else:
                warnings.warn(
                    "Reversing the filter is not supported in non-vectorized mode. The reversed_flag will be set to false."
                )
                reversed_flag = False
        save_data_into_hfile(length, order, res, filter_res, input_res, reversed_flag)


if __name__ == "__main__":
    main()
