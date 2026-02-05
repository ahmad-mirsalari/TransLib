import sys
import os
import warnings
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path
import argparse

import torch

from pyparsing import TextIO

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


@dataclass(frozen=True)
class MMConfig:
    """Configuration for matrix multiplication.
    """
    # flags
    mac: bool
    cast: bool
    vec: bool
    transpose_second: bool
    hw_mixed: bool
    mixed_vec: bool
    mantissa_bits: int

    # types
    dt_a: DKind
    dt_b: DKind
    dt_out: DKind
    cast_to: Optional[DKind]  # None if no casting destination


# ----------------------------
# Core kernel (loops preserved)
# ----------------------------


def matrix_mult(
    first_matrix: torch.Tensor,
    second_matrix: torch.Tensor,
    dt: List[DKind],
    mac_flag: bool = False,
    cast_flag: bool = False,
    cast_to: Optional[str] = "FP32",  # <- Optional (you pass None)
    vec_flag: bool = False,
    transpose: bool = False,
    mantissa_bits: int = 2,
    hwmixed_flag: bool = False,
    mixed_vec_flag: bool = False,
) -> torch.Tensor:
    """
    Performs matrix multiply with optional casting, MAC promotion and quantization.
    Loop order and accumulation semantics intentionally preserved.
    """
    if first_matrix.dim() != 2 or second_matrix.dim() != 2:
        raise ValueError("Both matrices must be 2D.")
    if first_matrix.shape[1] != second_matrix.shape[0]:
        raise ValueError(
            f"Incompatible shapes: {first_matrix.shape} x {second_matrix.shape}"
        )

    # Build config (once), then use helpers in the loops
    cast_to_kind = (
        None
        if (cast_to is None or str(cast_to).lower() in ("false", "none", ""))
        else _to_dkind(cast_to)
    )
    cfg = MMConfig(
        mac=mac_flag,
        cast=cast_flag,
        vec=vec_flag,
        transpose_second=transpose,
        hw_mixed=hwmixed_flag,
        mixed_vec=mixed_vec_flag,
        mantissa_bits=mantissa_bits,
        dt_a=dt[0],
        dt_b=dt[1],
        dt_out=dt[2],
        cast_to=cast_to_kind,
    )

    # Pre-cast inputs per policy
    A_eff, B_eff = first_matrix, second_matrix
    device = A_eff.device
    out_dtype = (
        torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()
    )
    rs = torch.zeros((A_eff.shape[0], B_eff.shape[1]), dtype=out_dtype, device=device)

    # before the loops
    acc_dtype = _acc_dtype_main(cfg)              # your accumulator dtype chooser
    zero_ten = torch.zeros((), dtype=acc_dtype, device=device)   # scalar 0 on the right device/dtype
    # ----------------------------
    # Transposed Vectorized branch
    # ----------------------------
    if cfg.transpose_second and cfg.vec:
        vec_step = 4 if cfg.dt_a is DKind.FP8_CUSTOM else 2
        full_chunks = int(B_eff.shape[0] / vec_step)
        processed = vec_step * full_chunks

        # main-loop accumulator dtype policy
        acc_main_dtype = _acc_dtype_main(cfg)

        for i in range(A_eff.shape[0]):
            for j in range(B_eff.shape[1]):
                # temp holds lane accumulators when not mixed_vec; products when mixed_vec
                temp = torch.zeros(
                    vec_step,
                    dtype=(
                        torch.float32
                        if cfg.dt_out is DKind.FP8_CUSTOM
                        else acc_main_dtype
                    ),
                    device=device,
                )
                a = torch.zeros(
                    vec_step,
                    dtype=(
                        torch.float32
                        if cfg.dt_a is DKind.FP8_CUSTOM
                        else cfg.dt_a.to_torch()
                    ),
                    device=device,
                )
                b = torch.zeros(
                    vec_step,
                    dtype=(
                        torch.float32
                        if cfg.dt_b is DKind.FP8_CUSTOM
                        else cfg.dt_b.to_torch()
                    ),
                    device=device,
                )
                # sum_ is the scalar accumulator across lanes/chunks
                sum_ = torch.zeros(
                    1,
                    dtype=(
                        torch.float32
                        if (cfg.mixed_vec or cfg.dt_out is DKind.FP8_CUSTOM)
                        else acc_main_dtype
                    ),
                    device=device,
                )

                # ---------- main processed chunks ----------
                for k in range(0, processed, vec_step):
                    for l in range(vec_step):
                        a[l] = A_eff[i][k + l]
                        b[l] = B_eff[k + l][j]

                    for l in range(vec_step):
                        if cfg.mixed_vec:
                            # compute products; reduction is below
                            temp[l] = acc_add(zero_ten, a[l], b[l], cfg, leftover=False, in_main=True)
                        else:
                            # lane-wise accumulate with policy (FMA/quantize-before-add if needed)
                            tl = temp[l].unsqueeze(0)  # make scalar tensor
                            tl = acc_add(
                                tl, a[l], b[l], cfg, leftover=False, in_main=True
                            )
                            temp[l] = tl.squeeze(0)

                    if cfg.mixed_vec:
                        # reduce the lane products into sum_ (no a*b here, just adds)
                        if vec_step == 2:
                            temp[0] += temp[1]
                        else:
                            temp[0] += temp[1]
                            temp[2] += temp[3]
                            temp[0] += temp[2]
                        sum_ += temp[0]
                    else:
                        # per-chunk accumulator-side finalize when not mixed_vec
                        temp = finalize_out_dtype(temp, cfg)

                # ---------- leftovers (k from processed .. N-1) ----------
                for k in range(processed, B_eff.shape[0]):
                    a0 = A_eff[i][k]
                    b0 = B_eff[k][j]

                    if cfg.mixed_vec:
                        # sum_ += a0 * b0  (use acc_add so product-side quantize/FMA rules apply)
                        sum_ = acc_add(sum_, a0, b0, cfg, leftover=True, in_main=False)
                    else:
                        # lane 0 accumulate (vector tail)
                        t0 = temp[0].unsqueeze(0)
                        t0 = acc_add(t0, a0, b0, cfg, leftover=True, in_main=False)
                        temp[0] = t0.squeeze(0)
                        # per-iter finalize when not mixed_vec
                        temp = finalize_out_dtype(temp, cfg)

                # ---------- finalize and write out ----------
                if not cfg.mixed_vec:
                    # fold lane accumulators into sum_
                    for l in range(vec_step):
                        sum_ += temp[l]
                        sum_ = finalize_out_dtype(sum_, cfg)
                else:
                    sum_ = finalize_out_dtype(sum_, cfg)

                rs[i][j] = sum_

        return rs
        # ----------------- End of Transposed Vectorized branch --------------------

    # ----------------------------
    # Non-transposed Vectorized branch
    # ----------------------------
    if not cfg.transpose_second and cfg.vec:
        # Determine unroll factor
        if cfg.mixed_vec and cfg.dt_a is DKind.FP8_CUSTOM:
            unroll = 4
        elif cfg.mixed_vec and (cfg.dt_a in (DKind.FP16, DKind.BF16)):
            unroll = 2
        else:
            unroll = 1

        # Leftover elements on N
        full_chunks = int(B_eff.shape[0] / unroll)
        processed = unroll * full_chunks

        # Leftover elements on P
        full_chunks_outer = int(B_eff.shape[1] / unroll)
        processed_outer = unroll * full_chunks_outer

        acc_main_dtype = _acc_dtype_main(cfg)

        for i in range(A_eff.shape[0]):  # M
            for j in range(0, processed_outer, 1):  # P
                # Scalar accumulator across k for this (i,j)
                temp = torch.zeros(
                    1,
                    dtype=(
                        torch.float32
                        if cfg.dt_out is DKind.FP8_CUSTOM
                        else acc_main_dtype
                    ),
                    device=device,
                )
                # Lane buffers (used for mixed_vec multiplies and/or loading)
                a = torch.zeros(
                    unroll,
                    dtype=(
                        torch.float32
                        if cfg.dt_a is DKind.FP8_CUSTOM
                        else cfg.dt_a.to_torch()
                    ),
                    device=device,
                )
                b = torch.zeros(
                    unroll,
                    dtype=(
                        torch.float32
                        if cfg.dt_b is DKind.FP8_CUSTOM
                        else cfg.dt_b.to_torch()
                    ),
                    device=device,
                )
                reg = torch.zeros(
                    unroll,
                    dtype=(
                        torch.float32
                        if cfg.dt_out is DKind.FP8_CUSTOM
                        else acc_main_dtype
                    ),
                    device=device,
                )

                # -------- Main vectorized body over k --------
                for k in range(0, processed, unroll):  # N
                    for l in range(unroll):
                        a[l] = A_eff[i][k + l]
                        b[l] = B_eff[k + l][j]

                    # Casting path for main body (only when not in the "mac and not cast" fast path, and not mixed_vec)
                    if not ((cfg.mac and not cfg.cast) or cfg.mixed_vec):
                        if cfg.cast and cfg.cast_to is not None:
                            a = _cast_apply(a, cfg.cast_to, cfg.mantissa_bits)
                            b = _cast_apply(b, cfg.cast_to, cfg.mantissa_bits)

                    if cfg.mixed_vec:
                        # Compute lane products; reduction into temp below
                        for l in range(unroll):
                            reg[l] = acc_add(zero_ten, a[l], b[l], cfg, leftover=False, in_main=True)

                        # Optional product-side finalize only applies in non-mixed_vec; we keep products wide here.
                        if unroll == 2:
                            reg[0] += reg[1]
                        else:
                            reg[0] += reg[1]
                            reg[2] += reg[3]
                            reg[0] += reg[2]
                        temp += reg[0]
                    else:
                        # Non-mixed-vec: accumulate via acc_add so FMA/quantize-product rules apply
                        for l in range(unroll):
                            temp = acc_add(
                                temp, a[l], b[l], cfg, leftover=False, in_main=True
                            )
                            if not cfg.mixed_vec:
                                # per-step accumulator-side finalize when not HW mixed
                                temp = finalize_out_dtype(temp, cfg)

                # -------- Leftovers on N --------
                for k in range(processed, B_eff.shape[0]):
                    a0 = A_eff[i][k]
                    b0 = B_eff[k][j]

                    # Casting path (leftovers)
                    if not ((cfg.mac and not cfg.cast) or cfg.mixed_vec):
                        if cfg.cast and cfg.cast_to is not None:
                            a0 = _cast_apply(
                                a0.unsqueeze(0), cfg.cast_to, cfg.mantissa_bits
                            )[0]
                            b0 = _cast_apply(
                                b0.unsqueeze(0), cfg.cast_to, cfg.mantissa_bits
                            )[0]

                    if cfg.mixed_vec:
                        # Use acc_add so product-side quantize/FMA rules apply for the scalar tail
                        temp = acc_add(temp, a0, b0, cfg, leftover=True, in_main=False)
                    else:
                        temp = acc_add(temp, a0, b0, cfg, leftover=True, in_main=False)
                        temp = finalize_out_dtype(temp, cfg)

                    # Finalize once at end for mixed_vec (kept wide during loop)
                    # if cfg.mixed_vec:
                    #     temp = finalize_out_dtype(temp, cfg)
                # Finalize once at end for mixed_vec (kept wide during loop)
                if cfg.mixed_vec:
                    temp = finalize_out_dtype(temp, cfg)

                rs[i][j] = temp

        # -------- Leftover on P (outer dimension) --------
        for j in range(processed_outer, B_eff.shape[1]):
            for i in range(A_eff.shape[0]):
                temp = torch.zeros(
                    1,
                    dtype=(
                        torch.float32
                        if cfg.dt_out is DKind.FP8_CUSTOM
                        else acc_main_dtype
                    ),
                    device=device,
                )
                for k in range(B_eff.shape[0]):
                    a0 = A_eff[i][k]
                    b0 = B_eff[k][j]

                    # Casting path
                    if not ((cfg.mac and not cfg.cast) or cfg.mixed_vec):
                        if cfg.cast and cfg.cast_to is not None:
                            a0 = _cast_apply(
                                a0.unsqueeze(0), cfg.cast_to, cfg.mantissa_bits
                            )[0]
                            b0 = _cast_apply(
                                b0.unsqueeze(0), cfg.cast_to, cfg.mantissa_bits
                            )[0]

                    # Accumulate via acc_add in all cases to honor product-side rules
                    temp = acc_add(temp, a0, b0, cfg, leftover=True, in_main=False)
                    if not cfg.mixed_vec:
                        temp = finalize_out_dtype(temp, cfg)

                if cfg.mixed_vec:
                    temp = finalize_out_dtype(temp, cfg)

                rs[i][j] = temp

        return rs
        # ------------------ End of Non-Transposed Vectorized branch --------------------

    # ----------------------------
    # Non-Vectorized branch
    # ----------------------------

    # Determine unroll factor
    unroll = 2 if cfg.hw_mixed else 1

    full_chunks = int(B_eff.shape[0] / unroll)
    processed = unroll * full_chunks

    for i in range(A_eff.shape[0]):
        for j in range(B_eff.shape[1]):

            # MAIN LOOP accumulator dtype per policy
            acc_main_dtype = _acc_dtype_main(cfg)
            temp = torch.zeros(
                1,
                dtype=(
                    torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else acc_main_dtype
                ),
                device=device,
            )
            a = torch.zeros(
                unroll,
                dtype=(
                    torch.float32
                    if cfg.dt_a is DKind.FP8_CUSTOM
                    else cfg.dt_a.to_torch()
                ),
                device=device,
            )
            b = torch.zeros(
                unroll,
                dtype=(
                    torch.float32
                    if cfg.dt_b is DKind.FP8_CUSTOM
                    else cfg.dt_b.to_torch()
                ),
                device=device,
            )
            # Main vectorized body
            for k in range(0, processed, unroll):

                for l in range(unroll):
                    a[l] = A_eff[i][k + l]
                    b[l] = B_eff[k + l][j]
                # casting path
                if not ((cfg.mac and not cfg.cast) or cfg.hw_mixed):
                    if cfg.cast and cfg.cast_to is not None:
                        # cast to cfg.cast_to for the multiply, then (optionally) down-quantize when mac disabled
                        a = _cast_apply(a, cfg.cast_to, cfg.mantissa_bits)
                        b = _cast_apply(b, cfg.cast_to, cfg.mantissa_bits)

                for l in range(unroll):
                    temp = acc_add(temp, a[l], b[l], cfg, leftover=False, in_main=True)

                if not cfg.hw_mixed:
                    temp = finalize_out_dtype(temp, cfg)

            # If HW mixed, quantize after loop
            if cfg.hw_mixed:
                temp = finalize_out_dtype(temp, cfg)

            # Leftovers
            for k in range(processed, B_eff.shape[0]):
                a0 = A_eff[i][k]
                b0 = B_eff[k][j]

                # casting path for leftovers (unchanged)
                if not ((cfg.mac and not cfg.cast) or cfg.hw_mixed):
                    if cfg.cast and cfg.cast_to is not None:
                        a0 = _cast_apply(
                            a0.unsqueeze(0), cfg.cast_to, cfg.mantissa_bits
                        )[0]
                        b0 = _cast_apply(
                            b0.unsqueeze(0), cfg.cast_to, cfg.mantissa_bits
                        )[0]

                temp = acc_add(temp, a0, b0, cfg, leftover=True, in_main=False)

                if not cfg.hw_mixed:
                    temp = finalize_out_dtype(temp, cfg)

            rs[i][j] = temp
    return rs


# ----------------------------
# I/O utilities
# ----------------------------


def write_matrix(
    matrix_to_write: torch.Tensor, name: str, file_pointer: TextIO
) -> None:
    """
    Writes a matrix to a file in a specific format.
    """
    sz0, sz1 = matrix_to_write.size()
    if "ref" in name:
        file_pointer.write(f"PI_L2 OUT_TYPE {name}[] = {{")
    elif "matA" in name:
        file_pointer.write(f"DATA_LOCATION MA_TYPE {name}[] = {{")
    else:
        file_pointer.write(f"DATA_LOCATION MB_TYPE {name}[] = {{")

    vals = []
    for i in range(sz0):
        for j in range(sz1):
            vals.append(str(matrix_to_write[i][j].item()))
    file_pointer.write(", ".join(vals))
    file_pointer.write("};\n")


def get_initial_config():
    """
    Parses command-line arguments to get the initial configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default=10)
    parser.add_argument("--n", default=10)
    parser.add_argument("--p", default=10)
    parser.add_argument("--std", default=1)
    parser.add_argument("--mac_flag", default=False)
    parser.add_argument("--float_type", default="FP32")  # e.g. "FP16,FP16,FP32"
    parser.add_argument("--vec_flag", default=False)
    parser.add_argument("--transpose", default=False)
    parser.add_argument("--mantissa_bits", default=2)
    parser.add_argument("--exploration_flag", default=False)
    parser.add_argument("--hwmixed_flag", default=False)
    args = parser.parse_args()

    args.mac_flag = str2bool(args.mac_flag)
    args.exploration_flag = str2bool(args.exploration_flag)
    args.transpose = str2bool(args.transpose)
    args.vec_flag = str2bool(args.vec_flag)
    args.hwmixed_flag = str2bool(args.hwmixed_flag)

    return (
        int(args.m),
        int(args.n),
        int(args.p),
        float(args.std),
        [s.strip() for s in args.float_type.split(",")],
        args.mac_flag,
        args.exploration_flag,
        args.transpose,
        args.vec_flag,
        int(args.mantissa_bits),
        args.hwmixed_flag,
    )


def transpose_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    Transposes a 2D matrix.
    """
    rows, cols = matrix.shape
    transposed = torch.zeros((cols, rows), dtype=matrix.dtype, device=matrix.device)
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed


def save_data_into_hfile(
    m: int, n: int, p: int, a_mat: torch.Tensor, b_mat: torch.Tensor, res: torch.Tensor
) -> None:
    """
    Saves matrix data into a header (.h) file.

    Args:
        m (int): Number of rows in matrix A.
        n (int): Number of columns in matrix A.
        p (int): Number of columns in matrix B.
        a_mat (torch.Tensor): Matrix A.
        b_mat (torch.Tensor): Matrix B.
        res (torch.Tensor): Result matrix.

    Returns:
        None
    """
    with open("data.h", "w") as f:
        f.write("#define M %s\n#define N %s\n#define P %s\n\n" % (m, n, p))
        write_matrix(a_mat, "matA", f)
        write_matrix(b_mat, "matB", f)
        write_matrix(res, "ref", f)


# ----------------------------
# Main
# ----------------------------


def main() -> None:

    """
    The main function that sets up the matrix multiplication, computes results, and
    saves data to a file.
    """
    (
        m,
        n,
        p,
        std,
        bits,
        mac_flag,
        exploration_flag,
        transpose,
        vec_flag,
        mantissa_bits,
        hwmixed_flag,
    ) = get_initial_config()

    if not exploration_flag:
        check_pulp_warnings(bits, mac_flag, hwmixed_flag, vec_flag)

    # Determine mixed vector flag from types + external vec_flag (your original rule)
    datatypes = select_dtypes(bits, 3)  # -> List[DKind]
    mixed_vec_flag = check_vec_flag(datatypes, vec_flag)

    if not exploration_flag:
        print(f"Mixed Vectorization Flag: {mixed_vec_flag}")

    # Reference inputs
    if exploration_flag:
        mean = 0.0
        a_ref = torch.normal(mean, std, (m, n))
        b_ref = torch.normal(mean, std, (n, p))
    else:
        a_ref = torch.randn((m, n), dtype=torch.float32)
        b_ref = torch.randn((n, p), dtype=torch.float32)

    # Reference FP32 result (note: we keep your flag usage; only dt is FP32 triplet)
    ref = matrix_mult(
        first_matrix=a_ref,
        second_matrix=b_ref,
        dt=[DKind.FP32, DKind.FP32, DKind.FP32],
        mac_flag=mac_flag,
        cast_flag=False,
        transpose=False,
        vec_flag=False,
        cast_to=None,
        mantissa_bits=0,
        hwmixed_flag=False,
        mixed_vec_flag=False,
    )

    cast_flag = check_cast(datatypes)
    if not exploration_flag:
        if DKind.FP8_CUSTOM in datatypes:
            print(
                f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])} and mantissa= {mantissa_bits} bits"
            )
        else:
            print(
                f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}"
            )

    # Choose cast_to
    cast_to = bits[-1] if len(bits) else "FP32"

    if not exploration_flag and cast_flag:
        if mac_flag:
            print(f"Running with casting to {cast_to}")
        else:
            warnings.warn(
                f"Running with casting to {cast_to}. On PULP, you may need to true the MAC flag."
            )

    output_folder = os.path.join(os.getcwd(), "exploration", str(std))
    os.makedirs(output_folder, exist_ok=True)

    # Init (C-like copy) in declared input kinds
    a_mat = matrix_init(a_ref, datatypes[0], mantissa_bits)
    b_mat = matrix_init(b_ref, datatypes[1], mantissa_bits)

    # Actual run
    res = matrix_mult(
        first_matrix=a_mat,
        second_matrix=b_mat,
        dt=datatypes,
        mac_flag=mac_flag,
        cast_flag=cast_flag,
        vec_flag=vec_flag,
        transpose=transpose,
        cast_to=cast_to,
        mantissa_bits=mantissa_bits,
        hwmixed_flag=hwmixed_flag,
        mixed_vec_flag=mixed_vec_flag,
    )

    # Metrics
    # Use resolved dtype names (works even if bits had fewer than 3 entries)
    dtype_tag = f"{dkind_name(datatypes[0])}_{dkind_name(datatypes[1])}_{dkind_name(datatypes[2])}"
    output_file = (
        os.path.join(
            output_folder,
            f"error_metric__{m}__{n}__{p}_{dtype_tag}_{mantissa_bits}_{std}.txt",
        )
        if exploration_flag
        else None
    )
    error_metric(ref, res, output_file)

    # Emit header
    if not exploration_flag:
        if transpose:
            b_mat = transpose_matrix(b_mat)
        save_data_into_hfile(m, n, p, a_mat, b_mat, res)


if __name__ == "__main__":
    main()
