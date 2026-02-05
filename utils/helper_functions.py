import argparse
import json
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, List, Tuple

import numpy as np
import torch
from mpmath import mp

from .fp_quantization import fp8_quantizer
from .fma_rounding import FP16, BF16, FP32 as F32FMT, fma_round_value, round_value


# ----------------------------
# Type system & configuration
# ----------------------------

class DKind(Enum):
    FP32 = auto()
    FP16 = auto()
    BF16 = auto()         
    FP8_CUSTOM = auto()

    def to_torch(self) -> Optional[torch.dtype]:
        if self is DKind.FP32: return torch.float32
        if self is DKind.FP16: return torch.float16
        if self is DKind.BF16: return torch.bfloat16
        if self is DKind.FP8_CUSTOM: return None
        return None

# ----------------------------
# Small utilities (adapters)
# ----------------------------

def _to_dkind(s: str) -> DKind:
    """Convert a string representation of a dtype to a DKind enum.

    Args:
        s (str): The string representation of the dtype.

    Raises:
        ValueError: If the string representation is unsupported.

    Returns:
        DKind: The corresponding DKind enum.
    """
    s = str(s).strip().upper()
    if s == "FP32": return DKind.FP32
    if s == "FP16": return DKind.FP16
    if s in ("FP16ALT", "BF16", "BFLOAT16"): return DKind.BF16
    if s == "FP8_CUSTOM": return DKind.FP8_CUSTOM
    raise ValueError(f"Unsupported dtype: {s}")

def dkind_name(k: DKind) -> str:
    """
    Return a stable string name for a DKind value.
    Ensures consistency between CLI spelling (e.g. FP16ALT) and enum members.
    """
    if k is DKind.BF16:
        return "FP16ALT"
    return k.name

def _quant_mbits_tensor_like(t: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Creates a tensor containing the specified number of mantissa bits, 
    ensuring that the new tensor is on the same device as the input tensor.

    Args:
        t (torch.Tensor): The reference tensor whose device will be used for the new tensor.
        bits (int): The number of mantissa bits to be represented in the new tensor.

    Returns:
        torch.Tensor: A tensor containing the value of `bits`, located on the same device as `t`.
    """

    # ensure the mantissa-bits tensor lives on the same device to avoid device mismatch
    return torch.tensor(bits, device=t.device)


def _cast_apply(t: torch.Tensor, kind: DKind, mantissa_bits: int) -> torch.Tensor:
    """
    Casts or quantizes a tensor to a specified data kind and mantissa precision.
    Args:
        t (torch.Tensor): The input tensor to be cast or quantized.
        kind (DKind): The target data kind (e.g., float16, float32, or custom FP8).
        mantissa_bits (int): Number of mantissa bits for quantization (used for custom FP8).
    Returns:
        torch.Tensor: The tensor cast to the specified data kind or quantized to the given mantissa precision.
    Notes:
        - If `kind` corresponds to a standard torch dtype, the tensor is cast using `t.to(td)`.
        - If `kind` is a custom FP8 type, quantization is performed using `fp8_quantizer` with the specified mantissa bits.
    """
    
    td = kind.to_torch()
    if td is None:  # FP8_CUSTOM
        return fp8_quantizer(t, _quant_mbits_tensor_like(t, mantissa_bits))
    return t.to(td)

# ---- Config accessors (duck-typing friendly) ----
from typing import Any

__all__ = (
    "_cfg_get",
    "_cfg_dt_in0",
    "_cfg_dt_in1",
    "_cfg_dt_out",
    "_cfg_cast_to",
)

def _cfg_get(cfg: Any, *names: str) -> Any:
    """
    Return the first present attribute from `names` on `cfg`.
    Raises AttributeError if none exist.
    """
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    raise AttributeError(f"Config is missing any of attributes {names!r}")

def _cfg_dt_in0(cfg: Any):
    """
    First input dtype.
    - Matmul: dt_a
    - Conv:   dt_x | dt_img
    """
    return _cfg_get(cfg, "dt_a", "dt_x", "dt_img")

def _cfg_dt_in1(cfg: Any):
    """
    Second input dtype.
    - Matmul: dt_b
    - Conv:   dt_f | dt_filt
    """
    return _cfg_get(cfg, "dt_b", "dt_f", "dt_filt")

def _cfg_dt_out(cfg: Any):
    """Output dtype: dt_out."""
    return _cfg_get(cfg, "dt_out")

def _cfg_cast_to(cfg: Any):
    """Optional cast destination dtype (may be None)."""
    return getattr(cfg, "cast_to", None)


def promote_for_mac(t: torch.Tensor, cfg: Any, leftover: bool = False) -> torch.Tensor:
    """
    Dtype policy only (no correctness dependence when FMA is used).
    - Main loop: fp32 if hw_mixed or mixed_vec; else fp64 if dt_out==FP32, else fp32.
    - Leftovers: special fp64 case when dt_out==FP32 & cast_to==FP32 & mixed_vec & mac.
    """
    # if not cfg.mac:
    #     return t
    target = _acc_dtype_leftover(cfg, leftover) if leftover else _acc_dtype_main(cfg)
    return t.to(target)



def finalize_out_dtype(t: torch.Tensor, cfg: Any) -> torch.Tensor:
    """
    Convert/quantize intermediate to declared output dtype with correct rounding.
    """
    dt_out = _cfg_dt_out(cfg)
    if dt_out is DKind.FP8_CUSTOM:
        return fp8_quantizer(t, _quant_mbits_tensor_like(t, cfg.mantissa_bits))
    if dt_out in (DKind.FP16, DKind.BF16, DKind.FP32):
        return _final_round_tensor(t, dt_out)
    return t


def _fmt_from_dkind(k: DKind):
    if k is DKind.FP16:  return FP16
    if k is DKind.BF16:  return BF16
    if k is DKind.FP32:  return F32FMT
    return None  # FP8_CUSTOM or others

def _acc_dtype_main(cfg: Any) -> torch.dtype:
    """
    Accumulator dtype for MAIN loop iters.
    - If hw_mixed or mixed_vec → fp32
    - Else if dt_out==FP32      → fp64
    - Else                      → fp32
    """
    if cfg.hw_mixed or cfg.mixed_vec:
        return torch.float32
    # if _cfg_dt_out(cfg) is DKind.FP32:
    #     return torch.float64
    return torch.float64

def _acc_dtype_leftover(cfg: Any, left_over: bool) -> torch.dtype:
    """
    Accumulator dtype for LEFTOVER iters.
    - Special case: dt_out==FP32 & cast_to==FP32 & mixed_vec & mac → fp64
    - Else mirror main-loop choice
    """

    if (
        # (cfg.mixed_vec
        # or cfg.mac) and left_over
        cfg.mac and left_over
    ):
        return torch.float64
    return _acc_dtype_main(cfg)
def cfg_requires_fmadd(cfg: Any, fmt: Any, left_over: bool = False) -> bool:
    if (
        (cfg.mixed_vec
        and cfg.mac and left_over) or (cfg.mac and not cfg.hw_mixed and not cfg.mixed_vec and fmt is not None)
    ):
        return True
    return False

def _final_round_tensor(t: torch.Tensor, kind: DKind) -> torch.Tensor:
    """
    Exact IEEE rounding (ties-to-even) to fp16/bf16/fp32, taken DIRECTLY from the
    tensor's float64 value (single rounding). Avoids any float32 staging before
    NumPy so we don't double-round.

    For bf16, NumPy lacks a dtype, so we compute scalar-rounded values into a
    float32 container (exactly representable), then cast to torch.bfloat16.
    """
    fmt = _fmt_from_dkind(kind)
    if fmt is None:
        # FP8_CUSTOM handled elsewhere
        return t

    # --- Key change: keep full precision all the way into NumPy ---
    t_cpu64 = t.detach().to(torch.float64).cpu()
    x = t_cpu64.numpy()  # x is float64; we will round ONCE to the target

    if kind is DKind.FP16:
        # single rounding 64 -> 16 via NumPy's half
        out_np = x.astype(np.float16, copy=False)
        out_torch = torch.from_numpy(out_np).to(t.device)
        return out_torch.to(torch.float16)

    if kind is DKind.BF16:
        # single rounding 64 -> bf16:
        # use your scalar round_value(..., fmt_bf16) into a float32 container
        # (bf16 values are exactly representable in float32), then cast to bfloat16.
        out_np = np.empty_like(x, dtype=np.float32)
        it = np.nditer(x, flags=["multi_index"])
        while not it.finished:
            out_np[it.multi_index] = round_value(float(it[0]), fmt)  # uses bf16 format
            it.iternext()
        out_torch = torch.from_numpy(out_np).to(t.device)
        return out_torch.to(torch.bfloat16)

    # kind is DKind.FP32  → single rounding 64 -> 32
    out_np = x.astype(np.float32, copy=False)
    out_torch = torch.from_numpy(out_np).to(t.device)
    return out_torch.to(torch.float32)

def _fp8_quantize_scalar(v: float, mant_bits: int, device) -> torch.Tensor:
    t = torch.tensor(v, dtype=torch.float32, device=device)
    return fp8_quantizer(t, torch.tensor(mant_bits, device=device))

def acc_add(temp: torch.Tensor, a_val: torch.Tensor, b_val: torch.Tensor,
            cfg: Any, *, leftover: bool = False, in_main: bool = False) -> torch.Tensor:
    """
    One step of accumulation: temp <- temp (+) a*b with your exact policies.

    FMA (single rounding) only when:
    cfg.mac and not cfg.hw_mixed and not cfg.mixed_vec and dt_out in {fp16,bf16,fp32}.

    Otherwise: compute product, and quantize that PRODUCT before adding with rules:
    - MAIN loop:     quantize iff (not cfg.mac and not cfg.hw_mixed)
    - LEFTOVERS:     quantize iff ((not cfg.mac) or cfg.hw_mixed)

    FP8: product computed at high precision; quantize per the same rules.


    Fix: in mixed-vec leftovers (leftover=True and in_main=False), the incoming
    accumulator is already wide (fp32). Keep the math in that wide dtype and do
    NOT narrow (no FMA-to-dt_out and no quantize-before-add). Let the caller
    finalize with finalize_out_dtype(...) later.
    """

    # --- Special-case: mixed-vec leftovers should stay wide (temp is fp32) ---
    keep_wide = bool(cfg.mixed_vec) and leftover and (not in_main)# and (temp.dtype == torch.float32)
    if keep_wide:
        # Do the product/add in the current accumulator dtype, return wide.
        acc_dtype = temp.dtype  # fp32
        prod = (a_val.to(acc_dtype) * b_val.to(acc_dtype))
        return temp + prod

    
    # Accumulator dtype for this step
    acc_dtype = _acc_dtype_main(cfg) if in_main else _acc_dtype_leftover(cfg, left_over=leftover)

    # Convenience: decide if we must quantize product before add at this step
    if in_main:
        quantize_before_add = ((not cfg.mac) and (not cfg.hw_mixed)) and not cfg.mixed_vec
    else:
        quantize_before_add = ((not cfg.mac) or cfg.hw_mixed) and not cfg.mixed_vec
    d_out = _cfg_dt_out(cfg)
    # ---------- FP8 path ----------
    if d_out is DKind.FP8_CUSTOM:
        # High-precision product
        mp.prec = 120
        prod_hp = float(mp.mpf(float(a_val.item())) * mp.mpf(float(b_val.item())))
        if quantize_before_add:
            reg = _fp8_quantize_scalar(prod_hp, cfg.mantissa_bits, temp.device)   # quantized product
            return temp.to(acc_dtype) + reg.to(acc_dtype)
        else:
            # keep wide (fp32/fp64) product; quantize temp outside if needed
            reg = torch.tensor(prod_hp, dtype=acc_dtype, device=temp.device)
            return temp.to(acc_dtype) + reg

    # ---------- IEEE targets (fp16/bf16/fp32) ----------
    # d_out= DKind.FP32
    fmt = _fmt_from_dkind(d_out)
    use_fma = cfg_requires_fmadd(cfg, fmt, left_over=leftover)
    if use_fma and fmt is None:
        # Should not happen: FMA only for fp16/bf16/fp32
        raise RuntimeError("Internal error: FMA requested but no fmt")
    if use_fma:
        # True fused multiply-add with correct rounding to target format
        fused = fma_round_value(float(a_val.item()), float(b_val.item()), float(temp.item()), fmt)
        return torch.tensor(fused, dtype=acc_dtype, device=temp.device)

    # Non-fused: product then optional quantize-before-add
    prod = (a_val.to(acc_dtype) * b_val.to(acc_dtype))
    if quantize_before_add:
        # Quantize the product to the target format BEFORE adding
        prod = finalize_out_dtype(prod, cfg)

    return temp.to(acc_dtype) + prod.to(acc_dtype)

# ----------------------------
# Metrics & helpers
# ----------------------------

def relative_absolute_error(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the Relative Absolute Error (RAE) between the true and predicted tensors.
    """
    true_mean = torch.mean(true)
    squared_error_num = torch.sum(torch.abs(true - pred))
    squared_error_den = torch.sum(torch.abs(true - true_mean))
    return squared_error_num / squared_error_den


def mean_squared_error(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Squared Error (MSE) between the true and predicted tensors.
    """
    diff = true - pred
    return torch.sum(diff * diff) / true.numel()


# =========================
# Data init (1D/2D)
# =========================

def matrix_init_like(input_tensor: torch.Tensor, kind: DKind, mantissa_bits: int) -> torch.Tensor:
    """
    C-like copy into a new tensor with same shape as input, applying quantization
    if kind is FP8_CUSTOM, otherwise preserving values and changing dtype.
    Supports 1D and 2D (and will work for N-D by elementwise loop).
    """
    device = input_tensor.device
    out_dtype = torch.float32 if kind is DKind.FP8_CUSTOM else kind.to_torch()
    out = torch.zeros(size=input_tensor.shape, dtype=out_dtype, device=device)

    if kind is DKind.FP8_CUSTOM:
        # elementwise quant to avoid device mismatch
        it = torch.nditer(input_tensor, flags=['multi_index']) if hasattr(torch, "nditer") else None
        if it is None:
            # Generic fallback for all dims
            out_view = out.view(-1)
            inp_view = input_tensor.view(-1)
            for i in range(inp_view.numel()):
                out_view[i] = fp8_quantizer(inp_view[i], _quant_mbits_tensor_like(inp_view[i], mantissa_bits))
        else:
            for x in it:
                idx = it.multi_index  # type: ignore[attr-defined]
                out[idx] = fp8_quantizer(input_tensor[idx], _quant_mbits_tensor_like(input_tensor[idx], mantissa_bits))
    else:
        out.copy_(input_tensor.to(out_dtype))

    return out


def error_metric(reference: torch.Tensor, result: torch.Tensor, output_file: Optional[str]) -> dict:
    """
    Computes error metrics between the reference and result tensors.
    """
    diff = reference - result

    mse = torch.mean(diff ** 2)
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(mse)
    denom = torch.sum((reference - torch.mean(reference)) ** 2)
    # Guard against a constant reference vector
    r2 = 1 - (torch.sum(diff ** 2) / torch.clamp(denom, min=torch.finfo(torch.float32).eps))
    rae = relative_absolute_error(reference, result)

    metrics_dict = {
        "MAE": float(mae.item()),
        "MSE": float(mse.item()),
        "RMSE": float(rmse.item()),
        "R2": float(r2.item()),
        "RAE": float(rae.item()),
    }

    if output_file:
        lines = [
            "Results of metrics:",
            f"MAE: {metrics_dict['MAE']}",
            f"MSE: {metrics_dict['MSE']}",
            f"RMSE: {metrics_dict['RMSE']}",
            f"R-Squared: {metrics_dict['R2']}",
            f"RAE is {metrics_dict['RAE']}",
        ]
        with open(output_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

        json_output_file = output_file.replace(".txt", ".json")
        with open(json_output_file, "w") as jf:
            json.dump(metrics_dict, jf, indent=4)
    else:
        print("Results of metrics:")
        for k, v in metrics_dict.items():
            print(f"{k}: {v}")

    return {
        "MAE": metrics_dict["MAE"],
        "MSE": metrics_dict["MSE"],
        "RMSE": metrics_dict["RMSE"],
        "R-Squared": metrics_dict["R2"],
        "RAE": metrics_dict["RAE"],
    }

def str2bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, int): return bool(v)
    if v is None: return False
    v = str(v).lower()
    if v in ("yes", "true", "t", "1"): return True
    if v in ("no", "false", "f", "0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def select_dtypes(user_dtypes: List[str], num_param: int) -> List[DKind]:
    """
    Map CLI strings to DKind triplet (A,B,OUT) using your original fallback rules.
    """
    if len(user_dtypes) == 1:
        return [_to_dkind(user_dtypes[0])] * num_param
    if len(user_dtypes) >= num_param:
        return [_to_dkind(user_dtypes[i]) for i in range(num_param)]

    # len is 2 → fill remaining spot with preference FP32 > BF16 > FP16 > FP8_CUSTOM
    mapped = [_to_dkind(x) for x in user_dtypes]
    present = {x.upper() for x in user_dtypes}
    if "FP32" in present:
        fill = DKind.FP32
    elif "FP16ALT" in present or "BF16" in present or "BFLOAT16" in present:
        fill = DKind.BF16
    elif "FP16" in present:
        fill = DKind.FP16
    elif "FP8_CUSTOM" in present:
        fill = DKind.FP8_CUSTOM
    else:
        fill = DKind.BF16
    return mapped + [fill] * (num_param - len(mapped))


def check_cast(datatypes: List[DKind]) -> bool:
    """
    Determines if casting is required based on input data types.

    Args:
        datatypes (list): List of data types.

    Returns:
        bool: True if casting is required, otherwise False.
    """
    
    return len(set(datatypes)) != 1


def check_vec_flag(datatypes: List[DKind], vec_flag: bool) -> bool:
    """
    Determine whether vectorization should be enabled based on input data types.

    Rules:
    - If `vec_flag` is explicitly False, vectorization is disabled.
    - If all data types in the list are identical, vectorization is disabled (fixed precision).
    - If not all types are equal, but the first two are equal, vectorization is enabled.
    - Otherwise, vectorization is disabled.

    Args:
        datatypes (list[str]): A list of data type strings (e.g., ["FP16", "FP16", "FP32"]).
        vec_flag (bool): External vectorization flag (True or False).

    Returns:
        bool: True if vectorization should be enabled, otherwise False.
    """
    
    if not vec_flag:
        return False
    if len(set(datatypes)) == 1:
        return False
    if len(set(datatypes[:2])) == 1:
        return True
    return False


def check_pulp_warnings(bits: List[str], mac_flag: bool, hwmixed_flag: bool, vec_flag: bool) -> None:
    """
    Checks and warns the user about PULP-related requirements:
    1. Certain dtype triplets require mac_flag to be True.
    2. Uniform FP8_CUSTOM / FP16 / FP16ALT triplets also require mac_flag.
    3. HW mixed precision requires mac_flag to be True.

    Args:
        bits (list): List of dtype strings (e.g., ["FP16", "FP16ALT", "FP16ALT"]).
        mac_flag (str): The MAC flag string (True or False).
        hwmixed_flag (str): The HW mixed precision flag (True or False).
    """
    if not mac_flag:
        warnings.warn(
            f"On PULP, with float_type={bits} you must set --mac_flag true (MAC in float32).",
            stacklevel=1,
        )

    if hwmixed_flag and not mac_flag:
        warnings.warn("Running with HW mixed precision on PULP requires --mac_flag true.", stacklevel=1)

    if vec_flag and not mac_flag:
        warnings.warn("Running with vectorization on PULP requires --mac_flag true.", stacklevel=1)