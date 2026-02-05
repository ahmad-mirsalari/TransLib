#!/bin/python3
import sys
import os
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, TextIO
import torch
from traitlets import Any

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
    # flags
    mac: bool
    vec: bool
    cast: bool
    hw_mixed: bool
    mixed_vec: bool
    mantissa_bits: int

    # types
    dt_img: DKind
    dt_filt: DKind
    dt_out: DKind
    cast_to: Optional[DKind]  # None if no casting destination


# ----------------------------
# Image padding & sizing
# ----------------------------


def add_padding_to_image(img: torch.Tensor, padding_width: int) -> torch.Tensor:
    """
    Add padding to the input image tensor.

    Args:
        img (torch.Tensor): The input image tensor.
        padding_width (int): The width of the padding to add.

    Returns:
        torch.Tensor: The padded image tensor.
    """
    out = torch.zeros(
        (img.shape[0] + 2 * padding_width, img.shape[1] + 2 * padding_width),
        dtype=img.dtype,
        device=img.device,
    )
    out[padding_width:-padding_width, padding_width:-padding_width] = img
    return out


def get_padding_width_per_side(kernel_size: int) -> int:
    """
    Get the padding width to apply to each side of the image.

    Args:
        kernel_size (int): The size of the convolutional kernel.

    Returns:
        int: The padding width for each side.
    """
    if kernel_size <= 0:
        raise ValueError("Kernel size must be a positive integer.")
    return kernel_size // 2  # p = floor(K / 2)


def calculate_target_size(
    img_width: int, kernel_width: int, stride: int, padding: int
) -> int:
    """
    Calculate the target size of the image after applying convolution.

    Args:
        img_width (int): The width of the input image.
        kernel_width (int): The width of the convolutional kernel.
        stride (int): The stride of the convolution.
        padding (int): The amount of padding applied to the image.

    Returns:
        int: The target size of the image after convolution.
    """
    if stride <= 0:
        raise ValueError("Stride must be a positive integer.")
    if kernel_width <= 0 or img_width <= 0:
        raise ValueError("Kernel width and image width must be positive integers.")
    target_size = ((img_width - kernel_width + 2 * padding) // stride) + 1
    if target_size <= 0:
        raise ValueError(
            f"Invalid target size: {target_size}. Check stride/kernel/padding."
        )
    return target_size


# ----------------------------
# I/O helpers
# ----------------------------


def write_matrix(
    matrix_to_write: torch.Tensor,
    name: str,
    length: int,
    file_pointer: Any,
    float_type: torch.dtype,
) -> None:
    """
    Write a matrix to a file in a specific format.

    Args:
        matrix_to_write (torch.Tensor): The matrix to write.
        name (str): The name of the matrix.
        length (int): The length of the matrix.
        file_pointer (Any): The file pointer to write to.
        float_type (torch.dtype): The data type of the matrix.
    """
    sz0 = matrix_to_write.size(0)
    sz1 = matrix_to_write.size(1)

    if "Filter_Kern" in name:
        file_pointer.write(f"DATA_LOCATION FIL_TYPE {name}[{length}] = {{")
    elif "ref" in name:
        file_pointer.write(f"PI_L2 OUT_TYPE {name}[{length}] = {{")
    else:
        file_pointer.write(f"DATA_LOCATION INP_TYPE {name}[{length}] = {{")

    if float_type == torch.float32:
        suffix = ")"
    elif float_type == torch.float16:
        suffix = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        suffix = ", dtype=torch.bfloat16)"
    else:
        suffix = ")"

    vals = []
    for i in range(sz0):
        for j in range(sz1):
            vals.append(
                str(matrix_to_write[i][j].item())
                .replace("tensor(", "")
                .replace(suffix, "")
            )
    file_pointer.write(", ".join(vals))
    file_pointer.write("};\n")


def save_data_into_hfile(
    out_width: int,
    img_width: int,
    filt_win: int,
    stride: int,
    res: Any,
    filter_conv: Any,
    input_conv: Any,
) -> None:
    with open("data.h", "w", encoding="utf-8") as f:
        f.write(
            '#ifndef _INPUT_IMAGE_ \n#define _INPUT_IMAGE_\n#pragma GCC diagnostic ignored "-Woverflow"\n\n'
        )
        f.write(
            "#define OUT_DIM %s\n#define OUT_ROW %s\n#define OUT_COL %s\n#define INP_COL %s\n#define STRIDE %s\n#define FILT_WIN %s\n\n"
            % (out_width * out_width, out_width, out_width, img_width, stride, filt_win)
        )
        write_matrix(input_conv, "In_Img", img_width * img_width, f, input_conv.dtype)
        write_matrix(
            filter_conv, "Filter_Kern", filt_win * filt_win, f, filter_conv.dtype
        )
        write_matrix(res, "ref", out_width * out_width, f, res.dtype)
        f.write("#endif \n")

    with open("config.h", "w", encoding="utf-8") as f:
        f.write(f"#define FILT_WIN {filt_win} \n\n")


# ----------------------------
# Core MAC over a submatrix (uses cfg.*)
# ----------------------------


def mac(img: torch.Tensor, kernel: torch.Tensor, cfg: ConvConfig) -> torch.Tensor:
    """
    Compute sum(img * kernel) with casting/quantization policies preserved.
    Returns a scalar tensor.
    """
    dev = img.device

    # -------- Non-vectorized --------
    if not cfg.vec:
        # MAIN LOOP accumulator dtype per policy
        acc_main_dtype = _acc_dtype_main(cfg)
        temp = torch.zeros(
            1,
            dtype=(
                torch.float32
                if cfg.dt_out is DKind.FP8_CUSTOM
                else acc_main_dtype
            ),
            device=dev,
        )
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                a = img[i][j]
                b = kernel[i][j]

                # Apply casting if needed
                if not ((cfg.mac and not cfg.cast) or cfg.hw_mixed):
                    if cfg.cast and cfg.cast_to is not None:
                        a = _cast_apply(a, cfg.cast_to, cfg.mantissa_bits)
                        b = _cast_apply(b, cfg.cast_to, cfg.mantissa_bits)

                temp = acc_add(temp, a , b, cfg, leftover= False, in_main= True)

                # progressive quantization/casting
                if not cfg.hw_mixed:
                    temp = finalize_out_dtype(temp, cfg)

        if cfg.hw_mixed:
            temp = finalize_out_dtype(temp, cfg)

        return temp.squeeze()

    # -------- Vectorized --------
    if cfg.dt_img is DKind.FP8_CUSTOM:
        vec_step = 3 if img.shape[1] == 3 else 4
    else:
        vec_step = 2

    a = torch.zeros(
        vec_step,
        dtype=(
            torch.float32 if cfg.dt_img is DKind.FP8_CUSTOM else cfg.dt_img.to_torch()
        ),
        device=dev,
    )
    b = torch.zeros(
        vec_step,
        dtype=(
            torch.float32 if cfg.dt_filt is DKind.FP8_CUSTOM else cfg.dt_filt.to_torch()
        ),
        device=dev,
    )
    # main-loop accumulator dtype policy
    acc_main_dtype = _acc_dtype_main(cfg)
    sum_val = torch.zeros(1, dtype=torch.float32, device=dev) if cfg.mixed_vec else None
    temp = torch.zeros(
            vec_step,
            dtype=torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else acc_main_dtype,
            device=dev,
        )

    # before the loops
    acc_dtype = _acc_dtype_main(cfg)              # your accumulator dtype chooser
    zero_ten = torch.zeros((), dtype=acc_dtype, device=dev)   # scalar 0 on the right device/dtype

    # rows major pass
    for i in range(img.shape[0]):
        for j in range(0, (img.shape[1] & 0xFFFFFFFE), vec_step):
            for x_input in range(vec_step):
                a[x_input] = img[i][j + x_input]
                b[x_input] = kernel[i][j + x_input]
            
            for x_mac in range(vec_step):
                if cfg.mixed_vec:
                    temp[x_mac] = a[x_mac] * b[x_mac]
                    temp[x_mac] = acc_add(zero_ten, a[x_mac], b[x_mac], cfg, leftover= False, in_main= True)
                else:
                    tx = temp[x_mac].unsqueeze(0)
                    tx = acc_add(tx, a[x_mac], b[x_mac], cfg, leftover= False, in_main= True)
                    tx = finalize_out_dtype(tx, cfg)
                    temp[x_mac] = tx.squeeze(0)

            if cfg.mixed_vec:
                if vec_step == 2:
                    temp[0] += temp[1]
                elif vec_step == 3:
                    temp[0] += temp[1]
                    temp[0] += temp[2]
                else:
                    temp[0] += temp[1]
                    temp[2] += temp[3]
                    temp[0] += temp[2]
                sum_val += temp[0]
            else:
                temp = finalize_out_dtype(temp, cfg)

    # odd column handling (original behavior)
    if not (
        (img.shape[1] & 0x1) and (cfg.dt_img is DKind.FP8_CUSTOM and img.shape[1] == 3)
    ):
        for i in range(0, img.shape[0] - 1, vec_step):
            for x_input in range(vec_step):
                a[x_input] = img[i + x_input][img.shape[1] - 1]
                b[x_input] = kernel[i + x_input][img.shape[1] - 1]

            for x_mac in range(vec_step):
                if cfg.mixed_vec:
                    
                    temp[x_mac] = a[x_mac] * b[x_mac]
                    temp[x_mac] = acc_add(zero_ten, a[x_mac], b[x_mac], cfg, leftover= False, in_main= True)
                else:
                    tx = temp[x_mac].unsqueeze(0)
                    tx = acc_add(tx, a[x_mac], b[x_mac], cfg, leftover= False, in_main= True)
                    tx = finalize_out_dtype(tx, cfg)
                    temp[x_mac] = tx.squeeze(0)

            if cfg.mixed_vec:
                if vec_step == 2:
                    temp[0] += temp[1]
                elif vec_step == 3:
                    temp[0] += temp[1]
                    temp[0] += temp[2]
                else:
                    temp[0] += temp[1]
                    temp[2] += temp[3]
                    temp[0] += temp[2]
                sum_val += temp[0]
            else:
                temp = finalize_out_dtype(temp, cfg)

        # final element
        if cfg.mixed_vec:
            sum_val = acc_add(sum_val, img[img.shape[0] - 1][img.shape[1] - 1], kernel[img.shape[0] - 1][img.shape[1] - 1], cfg, leftover= False, in_main= True)
        else:
            tx = temp[0].unsqueeze(0)
            tx = acc_add(tx, img[img.shape[0] - 1][img.shape[1] - 1], kernel[img.shape[0] - 1][img.shape[1] - 1], cfg, leftover= False, in_main= True)
            tx = finalize_out_dtype(tx, cfg)
            temp[0] = tx.squeeze(0)

    # reduction
    if not cfg.mixed_vec:
        for y in range(1, vec_step):
            temp[0] += temp[y]
            temp = finalize_out_dtype(temp, cfg)
        return temp[0]
    else:
        # if cfg.mac:
        sum_val = finalize_out_dtype(sum_val, cfg)
        return sum_val.squeeze()


# ----------------------------
# 2D convolution (extract + MAC)
# ----------------------------


def convolve(
    img: torch.Tensor,
    kernel: torch.Tensor,
    out_width: int,
    cfg: ConvConfig,
    stride: int = 1,
) -> torch.Tensor:
    out_dtype = (
        torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()
    )
    out_img = torch.zeros((out_width, out_width), dtype=out_dtype, device=img.device)

    target_size = out_img.shape[0]
    ksz = kernel.shape[0]

    if not cfg.vec:
        for i in range(target_size):
            for j in range(target_size):
                sub = img[i * stride : i * stride + ksz, j * stride : j * stride + ksz]
                out_img[i, j] = mac(sub, kernel, cfg)
    else:
        # original behavior: disable casting in vectorized path
        cfg_no_cast = ConvConfig(
            mac=cfg.mac,
            vec=cfg.vec,
            cast=False,
            hw_mixed=cfg.hw_mixed,
            mixed_vec=cfg.mixed_vec,
            mantissa_bits=cfg.mantissa_bits,
            dt_img=cfg.dt_img,
            dt_filt=cfg.dt_filt,
            dt_out=cfg.dt_out,
            cast_to=None,
        )
        for j in range(target_size):
            for i in range(target_size):
                sub = img[i * stride : i * stride + ksz, j * stride : j * stride + ksz]
                out_img[i, j] = mac(sub, kernel, cfg_no_cast)

    return out_img


# ----------------------------
# CLI config
# ----------------------------


def get_initial_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width", required=True)
    parser.add_argument("--filt_win", required=True)
    parser.add_argument("--stride", default=1)
    parser.add_argument("--padding", default="valid")
    parser.add_argument("--std", default=0.01)
    parser.add_argument("--vec_flag", default=False)
    parser.add_argument("--mac_flag", default=True)
    parser.add_argument("--float_type", default="FP32")
    parser.add_argument("--mantissa_bits", default=2)
    parser.add_argument("--exploration_flag", default=False)
    parser.add_argument("--hwmixed_flag", default=False)
    args = parser.parse_args()

    img_width = int(args.img_width)
    filt_win = int(args.filt_win)
    stride = int(args.stride)
    padding = str(args.padding)
    std = float(args.std)
    mac_flag = str2bool(args.mac_flag)
    vec_flag = str2bool(args.vec_flag)
    mantissa_bits = int(args.mantissa_bits)
    exploration_flag = str2bool(args.exploration_flag)
    hwmixed_flag = str2bool(args.hwmixed_flag)
    bits = [s.strip() for s in args.float_type.split(",")]

    if padding == "same" and stride != 1:
        sys.exit("ValueError: padding='same' is not supported for strided convolutions")

    return (
        img_width,
        filt_win,
        stride,
        padding,
        std,
        bits,
        mac_flag,
        vec_flag,
        exploration_flag,
        mantissa_bits,
        hwmixed_flag,
    )


# ----------------------------
# Main
# ----------------------------


def main():
    (
        img_width,
        filt_win,
        stride,
        padding,
        std,
        bits,
        mac_flag,
        vec_flag,
        exploration_flag,
        mantissa_bits,
        hwmixed_flag,
    ) = get_initial_config()

    # PULP warnings
    if not exploration_flag:
        check_pulp_warnings(bits, mac_flag, hwmixed_flag, vec_flag)

    # Determine mixed vector flag from types + external vec_flag
    datatypes = select_dtypes(bits, 3)  # [dt_img, dt_filt, dt_out]-> List[DKind]
    mixed_vec_flag = check_vec_flag(datatypes, vec_flag)
    if not exploration_flag:
        print(f"Mixed Vectorization Flag: {mixed_vec_flag}")

    # Inputs
    if exploration_flag:
        input_ref = torch.normal(0.0, std, (img_width, img_width), dtype=torch.float32)
        filter_ref = torch.normal(0.0, std, (filt_win, filt_win), dtype=torch.float32)
    else:
        input_ref = torch.randn((img_width, img_width), dtype=torch.float32)
        filter_ref = torch.randn((filt_win, filt_win), dtype=torch.float32)

    # Output size & padding
    out_width = img_width
    if padding == "same":
        pad = get_padding_width_per_side(filt_win)
        input_ref = add_padding_to_image(input_ref, pad)
        out_width = img_width
        img_width = input_ref.shape[0]
    else:  # valid
        out_width = calculate_target_size(
            img_width=img_width, kernel_width=filt_win, stride=stride, padding=0
        )

    # FP32 reference (no MAC / no vec / no cast)
    cfg_ref = ConvConfig(
        mac=False,
        vec=False,
        cast=False,
        hw_mixed=False,
        mixed_vec=False,
        mantissa_bits=0,
        dt_img=DKind.FP32,
        dt_filt=DKind.FP32,
        dt_out=DKind.FP32,
        cast_to=None,
    )
    ref = convolve(
        img=input_ref,
        kernel=filter_ref,
        out_width=out_width,
        cfg=cfg_ref,
        stride=stride,
    )

    # Casting decision and cast_to selection (keep original rule)
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

    # Original script's cast_to priority derived from first two inputs
    # Choose cast_to
    cast_to = bits[-1] if len(bits) else "FP32"
    cast_to_kind = _to_dkind(cast_to) if cast_flag else None

    if not exploration_flag and cast_flag:
        print(f"Running with casting to {cast_to}")

    # Output folder
    output_folder = os.path.join(os.getcwd(), "exploration", str(std))
    os.makedirs(output_folder, exist_ok=True)

    # Init tensors in declared input kinds
    input_conv = matrix_init(input_ref, datatypes[0], mantissa_bits=mantissa_bits)
    filter_conv = matrix_init(
        filter_ref, datatypes[1], mantissa_bits=mantissa_bits
    )

    # Build runtime cfg
    cfg_run = ConvConfig(
        mac=mac_flag,
        vec=vec_flag,
        cast=cast_flag,
        hw_mixed=hwmixed_flag,
        mixed_vec=mixed_vec_flag,
        mantissa_bits=mantissa_bits,
        dt_img=datatypes[0],
        dt_filt=datatypes[1],
        dt_out=datatypes[2],
        cast_to=cast_to_kind,
    )

    # Actual run
    res = convolve(
        img=input_conv,
        kernel=filter_conv,
        out_width=out_width,
        cfg=cfg_run,
        stride=stride,
    )

    # Metrics
    dtype_tag = f"{dkind_name(datatypes[0])}_{dkind_name(datatypes[1])}_{dkind_name(datatypes[2])}"
    output_file = (
        os.path.join(
            output_folder,
            f"error_metric__{img_width}__{filt_win}_{dtype_tag}_{mantissa_bits}_{std}.txt",
        )
        if exploration_flag
        else None
    )
    error_metric(ref, res, output_file)

    # Emit headers
    if not exploration_flag:
        save_data_into_hfile(
            out_width, img_width, filt_win, stride, res, filter_conv, input_conv
        )


if __name__ == "__main__":
    main()
# EOF
