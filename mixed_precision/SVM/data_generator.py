#!/bin/python3
import json
import sys
import os
import argparse
from pathlib import Path
import struct
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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
    matrix_init_like as matrix_init,  # same C-like copy + fp8 quant
    str2bool,
    error_metric,  # optional if you later want error stats
)
from utils.fp_quantization import fp8_quantizer
import time


# ----------------------------
# Config / constants
# ----------------------------

FOLDER_ADDR = "."

# GIST-style fast exp constants (float32 path)
GIST_A = 12102203.17133801
GIST_B = 1064986823.010288
GIST_C = 8388608
GIST_D = 2139095040

def fastexp_gist(x: float) -> float:
    """Fast exp approx via float32 bit-hack (matches your previous code)."""
    x = float(x)
    x = GIST_A * x + GIST_B
    if x < GIST_C or x > GIST_D:
        x = 0.0 if x < GIST_C else float.fromhex("0x1.fffffep+127")
    n = int(x)
    return struct.unpack("f", struct.pack("I", n))[0]
# ----------------------------

# ----------------------------
# SVM config
# ----------------------------

@dataclass(frozen=True)
class SVMConfig:
    # flags
    mac: bool
    vec: bool
    cast: bool
    hw_mixed: bool
    mixed_vec: bool
    mantissa_bits: int

    # types
    dt_x: DKind
    dt_w: DKind
    dt_out: DKind
    cast_to: Optional[DKind]  # None if no casting destination

# ----------------------------
# Data helpers (unchanged spirit)
#

def create_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_state: int = None,
) -> Tuple:
    """
    Creates a synthetic dataset for classification tasks.

    Args:
        n_samples (int): Number of samples to generate. Default is 1000.
        n_features (int): Number of total features. Default is 20.
        n_classes (int): Number of target classes. Default is 2.
        random_state (int): Random seed for reproducibility. Default is None.

    Returns:
        Tuple:
            x_train (ndarray): Training feature matrix.
            x_test (ndarray): Testing feature matrix.
            y_train (ndarray): Training target vector.
            y_test (ndarray): Testing target vector.
    """
    # Generate a synthetic dataset with double the required samples
    total_samples = n_samples * 4
    features, targets = make_classification(
        n_samples=total_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_classes,  # Number of informative features set to match the classes
        random_state=random_state,
    )

    # Split the dataset so that test set has exactly n_samples
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=n_samples, random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def scale_data(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales the dataset using Min-Max scaling and scales the results further by a factor of 0.2.

    Args:
        x_train (np.ndarray): Training data to scale.
        x_test (np.ndarray): Test data to scale.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Scaled training and test datasets.
    """
    scaler = MinMaxScaler()

    # Fit the scaler on the training set only
    scaler.fit(x_train)

    # Transform both the training set and the test set
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Scale the transformed data further by 0.2
    return x_train_scaled * 0.2, x_test_scaled * 0.2


def create_mdataset(
    n_samples: int = 1000, n_classes: int = 2, n_features: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a scaled dataset for machine learning experiments.

    Args:
        n_samples (int, optional): The number of samples to generate. Defaults to 1000.
        n_classes (int, optional): The number of classes in the dataset. Defaults to 2.
        n_features (int, optional): The number of features per sample. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Scaled training features, scaled test features, training labels, and test labels.
    """
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Generate the dataset
    x_train, x_test, y_train, y_test = create_dataset(
        n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=42
    )

    # Scale the data
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)


    return x_train_scaled, x_test_scaled, y_train, y_test

def min_max_scaling(data, column, a, b):
    """
    Perform min-max scaling on a specified column of the data.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing the column to scale.
    column : str
        The column name to be scaled.
    a : float
        The desired minimum value after scaling.
    b : float
        The desired maximum value after scaling.

    Returns
    -------
    None
        The data is modified in place.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column] = ((data[column] - min_val) / (max_val - min_val)) * (b - a) + a


def read_dataset(name: str, input_size: float, c: int = 1, f: int = 10) -> tuple:
    """
    Reads and loads a dataset based on the specified name and input size.

    Parameters
    ----------
    name : str
        The name of the dataset to read. Options are "custom", "bill", and "cancer".
    input_size : float
        The proportion of the dataset to be used for the test set.
    c : int, optional
        The number of classes (default is 1).
    f : int, optional
        The number of features (default is 10).

    Returns
    -------
    tuple
        A tuple containing the training data (x_train, y_train) and
            the testing data (x_test, y_test).
        x_train, x_test: torch.Tensor
        y_train, y_test: torch.Tensor
    """

    if name == "custom":
        c = 2 # number of classes
        f = 35 # number of features
        x_train, x_test, y_train, y_test = create_mdataset(
            n_samples=input_size, n_classes=c, n_features=f
        )

    elif name == "bill":
        filepath = FOLDER_ADDR + "/dataset/bill_authentication.csv"
        bankdata = pd.read_csv(filepath)
        x_df = bankdata.drop("Class", axis=1)
        y_df = bankdata["Class"]

        x = x_df.values
        y = y_df.values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=input_size)

    elif name == "cancer":
        filepath = FOLDER_ADDR + "/dataset/breast-cancer.csv"
        dataset = pd.read_csv(filepath)
        # Changing the datatype of the column - dignosis in the dataset
        dataset.diagnosis = dataset.diagnosis.astype("category")
        # Drop the column - id
        dataset.drop(["id"], axis=1, inplace=True)
        dataset.head(10)
        # Scaling all the numerical columns
        columns = list(dataset.columns)
        numerical_columns = columns[1:]
        for each_column in numerical_columns:
            min_max_scaling(dataset, each_column, 0, 1)

        # Encoding the target column - diagnosis
        target_column = dataset["diagnosis"]
        encoded_target = [0 if value == "B" else 1 for value in target_column]

        dataset["diagnosis"] = encoded_target
        # Splitting the data into independent and dependent matrices - X and Y

        x = dataset.iloc[:, 1:].values
        y = dataset.iloc[:, 0].values
        # Dividing the data into training and test sets in the ratio of 80:20

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=input_size, shuffle=True, random_state=27
        )
    else:
        sys.exit("ValueError: this dataset is not supported!!!!!")

    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
    x_test, y_test = x_test.type(torch.float32), y_test.type(torch.float32)

    return x_train, x_test, y_train, y_test

# ----------------------------
# Training
# ----------------------------

def svm_train(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    kernel_function: str,
    tol: float = 1e-3,
    max_passes: int = 25,
    args: tuple = (),
) -> dict:
    """
    Trains an SVM classifier using a simplified version of the SMO algorithm.

    Parameters
    ----------
    x : numpy.ndarray
        (m x n) Matrix of training examples. Each row is a training example, and the
        jth column holds the jth feature.
    y : numpy.ndarray
        (m, ) A vector containing 1 for positive examples and 0 for negative examples.
    c : float
        The standard SVM regularization parameter.
    kernel_function : str
        The name of the kernel function. Supported values are 'linearKernel' and 'gaussianKernel'.
    tol : float, optional
        Tolerance value used for determining equality of floating point numbers. Default is 1e-3.
    max_passes : int, optional
        The number of iterations over the dataset (without changes to alpha)
        before the algorithm quits. Default is 25.
    args : tuple, optional
        Extra arguments for the kernel function, such as the sigma parameter for the Gaussian
        kernel.

    Returns
    -------
    dict
        The trained SVM model containing:
        - 'x': Support vectors.
        - 'y': Support vector labels.
        - 'kernel_function': The kernel function used.
        - 'b': Bias term.
        - 'args': Additional kernel arguments.
        - 'alphas': Lagrange multipliers for the support vectors.
        - 'w': Weights (only for linear kernels).
    Notes
    -----
    This is a simplified version of the SMO algorithm for training SVMs. In practice, if
    you want to train an SVM classifier, we recommend using an optimized package such as:
    - LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
    - SVMLight (http://svmlight.joachims.org/)
    - scikit-learn (http://scikit-learn.org/stable/modules/svm.html) which contains python wrappers
    for the LIBSVM library.
    """
    # make sure data is signed int
    y = y.astype(int)
    # Dataset size parameters
    m, _ = x.shape

    passes = 0
    errors = np.zeros(m)
    alphas = np.zeros(m)
    bias = 0

    # Map 0 to -1 for SVM compatibility
    y[y == 0] = -1

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    # gracefully will **not** do this)

    # We have implemented the optimized vectorized version of the Kernels here so
    # that the SVM training will run faster
    if kernel_function == "linearKernel":
        # Vectorized computation for the linear kernel
        # This is equivalent to computing the kernel on every pair of examples
        kernel_matrix = np.dot(x, x.T)
    elif kernel_function == "gaussianKernel":
        # vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        kernel_matrix = np.zeros((x.shape[0], x.shape[0]))
        for i, x1 in enumerate(x):
            for j, x2 in enumerate(x):
                x1 = x1.ravel()
                x2 = x2.ravel()
                kernel_matrix[i, j] = np.exp(
                    -np.sum(np.square(x1 - x2)) / (2 * (args[0] ** 2))
                )
    else:
        kernel_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                kernel_matrix[i, j] = kernel_function(x[i, :], x[j, :])
                kernel_matrix[j, i] = kernel_matrix[i, j]

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            errors[i] = bias + np.sum(alphas * y * kernel_matrix[:, i]) - y[i]

            if (y[i] * errors[i] < -tol and alphas[i] < c) or (
                y[i] * errors[i] > tol and alphas[i] > 0
            ):
                # select the alpha_j randomly
                j = np.random.choice(list(range(i)) + list(range(i + 1, m)), size=1)[0]

                errors[j] = bias + np.sum(alphas * y * kernel_matrix[:, j]) - y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y[i] == y[j]:
                    l = max(0, alphas[j] + alphas[i] - c)
                    h = min(c, alphas[j] + alphas[i])
                else:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(c, c + alphas[j] - alphas[i])

                if l == h:
                    continue

                eta = (
                    2 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
                )

                # objective function positive definite, there will be a minimum along the direction
                # of linear equality constrain, and eta will be greater than zero
                # we are actually computing -eta here (so we skip of eta >= 0)
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (errors[i] - errors[j]) / eta
                alphas[j] = max(l, min(h, alphas[j]))

                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue
                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = (
                    bias
                    - errors[i]
                    - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[i, j]
                    - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[i, j]
                )

                b2 = (
                    bias
                    - errors[j]
                    - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[i, j]
                    - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[j, j]
                )

                if 0 < alphas[i] < c:
                    bias = b1
                elif 0 < alphas[j] < c:
                    bias = b2
                else:
                    bias = (b1 + b2) / 2

                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    idx = alphas > 0
    model = {
        "X": x[idx, :],
        "y": y[idx],
        "kernelFunction": kernel_function,
        "b": bias,
        "args": args,
        "alphas": alphas[idx],
        "w": np.dot(alphas * y, x),
    }
    return model
# ----------------------------

# ----------------------------
# Core math: matmul + predict
# ----------------------------
def matrix_mult(xs: torch.Tensor, ys: torch.Tensor, cfg: SVMConfig) -> torch.Tensor:
    """
    Custom matrix multiplication with support for vectorization, type casting, and quantization.

    Parameters
    ----------
    xs : torch.Tensor
        Input matrix X.
    ys : torch.Tensor
        Input matrix Y.
    cfg : SVMConfig
        Configuration object containing flags and data types.

    Returns
    -------
    torch.Tensor
        Resultant matrix from the multiplication.
    """

    device = xs.device
    out_dtype = torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()
    rs = torch.zeros((xs.shape[0], ys.shape[1]), dtype=out_dtype, device=device)

    # ------------- scalar path -------------
    if not cfg.vec:
        acc_main_dtype = _acc_dtype_main(cfg)
        # iterate through rows of X
        for i in range(xs.shape[0]):
            # iterate through columns of Y
            for j in range(ys.shape[1]):
                temp = torch.zeros(
                                1,
                                dtype=(torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else acc_main_dtype),
                                device=device,
                                )
                # iterate through rows of Y
                for k in range(ys.shape[0]):
                    a, b = xs[i][k], ys[k][j]
                    
                    # cast if needed (same rule as FIR/matmul)
                    # if not ((cfg.mac and not cfg.cast) or cfg.hw_mixed):
                    if cfg.cast and cfg.cast_to is not None:
                        a = _cast_apply(a, cfg.cast_to, cfg.mantissa_bits)
                        b = _cast_apply(b, cfg.cast_to, cfg.mantissa_bits)
                    
                    # one FMA step with the shared policy
                    temp = acc_add(temp, a, b, cfg, leftover=False, in_main=True)
                                        
                    # in main loop, finalize unless HW-mixed keeps it wide
                    if not cfg.hw_mixed:
                        temp = finalize_out_dtype(temp, cfg)
                # finalize once at the end if HW-mixed kept it wide
                if cfg.hw_mixed:
                    temp = finalize_out_dtype(temp, cfg)
                
                rs[i][j] = temp
        return rs
    # ---------- vectorized path ----------
    vec_step = 4 if cfg.dt_x is DKind.FP8_CUSTOM else 2
    processed = (ys.shape[0] // vec_step) * vec_step # number of full chunks
    
    a = torch.zeros(
        vec_step,
        dtype=(torch.float32 if cfg.dt_x is DKind.FP8_CUSTOM else cfg.dt_x.to_torch()),
        device=device,
    )
    b = torch.zeros(
        vec_step,
        dtype=(torch.float32 if cfg.dt_w is DKind.FP8_CUSTOM else cfg.dt_w.to_torch()),
        device=device,
    )

    acc_main_dtype = _acc_dtype_main(cfg)
    
    for i in range(xs.shape[0]):
        # iterate through columns of Y
        for j in range(ys.shape[1]):
            temp = torch.zeros(
                vec_step,
                dtype=(torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else acc_main_dtype),
                device=device,
            )
            sum_val = torch.zeros(1, dtype=acc_main_dtype, device=device)

            # iterate through rows of Y
            for k in range(0, processed, vec_step):
                for l in range(vec_step):
                    a[l] = xs[i][k + l]
                    b[l] = ys[k + l][j]
                
                # lane-wise multiply-accumulate
                if cfg.mixed_vec:
                    for l in range(vec_step):
                        t = torch.zeros(1, dtype=acc_main_dtype, device=device)
                        t = acc_add(t, a[l], b[l], cfg, leftover=False, in_main=True)
                        temp[l] = t.squeeze(0)
                    # horizontal reduce lanes into sum_val
                    if vec_step == 2:
                        temp[0] += temp[1]
                    else:
                        temp[0] += temp[1]
                        temp[2] += temp[3]
                        temp[0] += temp[2]
                    sum_val += temp[0]
                    
                else:
                    for l in range(vec_step):
                        tl = temp[l].unsqueeze(0)
                        tl = acc_add(tl, a[l], b[l], cfg, leftover=False, in_main=True)
                        tl = finalize_out_dtype(tl, cfg)
                        temp[l] = tl.squeeze(0)

            # leftovers
            for k in range(processed, ys.shape[0]):
                a[0] = xs[i][k]
                b[0] = ys[k][j]

                if cfg.mixed_vec:
                    sum_val = acc_add(sum_val, a[0], b[0], cfg, leftover=True, in_main=False)
                else:
                    t0 = temp[0].unsqueeze(0)
                    t0 = acc_add(t0, a[0], b[0], cfg, leftover=True, in_main=False)
                    temp[0] = finalize_out_dtype(t0, cfg).squeeze(0)
            # finalize to result
            if cfg.mixed_vec:
                rs[i, j] = finalize_out_dtype(sum_val, cfg)
            else:
                acc = torch.zeros(1, dtype=acc_main_dtype, device=device)
                for l in range(vec_step):
                    acc += temp[l]
                    acc = finalize_out_dtype(acc, cfg)
                rs[i, j] = acc

    return rs


def svm_predict(
    model: Dict[str, Union[torch.Tensor, str, float]],
    x: torch.Tensor,
    dt:  List[DKind],
    mac_flag: bool = False,
    vec_flag: bool = False,
    cast_flag: bool = False,
    cast_to: Optional[str] = "FP32",
    mantissa_bits: int = 2,
    hwmixed_flag: bool = False,
    mixed_vec_flag: bool = False
) -> torch.Tensor:
    """
    Predicts labels using a trained SVM model.

    Parameters
    ----------
    model : Dict[str, Union[torch.Tensor, str, float]]
        The trained SVM model, as returned by the `svm_train` function.
    x : torch.Tensor
        A (m x n) matrix where each row represents a data point.
    dt : tuple
        Data type configuration (e.g., ("fp8_custom", ..., torch.float32)).
    mac_flag : str
        If "true", enables MAC precision adjustment to float32.
    vec_flag : str
        If "true", enables vectorized operations.
    cast_flag : str
        If "true", enables data type casting.
    cast_to : str
        Target data type for casting (e.g., "fp16", "fp16alt").
    mantissa_bits : int
        Number of mantissa bits for FP8 quantization.
    hwmixed_flag : str
        If "true", enables hardware mixed precision.
    mixed_vec_flag : str
        If "true", enables  vectorized operations for mixed precision.

    Returns
    -------
    torch.Tensor
        A (m,) tensor of predictions with values {0, 1}.
    """
    # check if we are getting a vector. If so, then assume we only need to do predictions
    # for a single example

    if x.ndim == 1:
        x = x[np.newaxis, :]
        
    cast_to_kind = None if (cast_to is None or str(cast_to).lower() in ("false","none","")) else _to_dkind(cast_to)
    cfg = SVMConfig(
        mac=mac_flag, vec=vec_flag, cast=cast_flag, hw_mixed=hwmixed_flag, mixed_vec=mixed_vec_flag,
        mantissa_bits=mantissa_bits,
        dt_x=dt[0], dt_w=dt[1], dt_out=dt[2], cast_to=cast_to_kind
    )
        
    pred = torch.zeros((x.shape[0], 1), dtype=torch.float32)
    # ---- linear kernel: y = X @ w + b ----
    if model["kernelFunction"] == "linearKernel":
        w = model["w"] 
        b = model["b"]
        y = torch.zeros((x.shape[0], 1), dtype=torch.float32)

        # if b.dtype == torch.bfloat16: # unsupported ScalarType BFloat16 in numpy
        #     b = b.to(torch.float32)
        bias = finalize_out_dtype(b, cfg)
        y = matrix_mult(x, w,cfg=cfg)
        y = y + bias
        if cfg.dt_out is DKind.BF16:
            y = y.to(torch.bfloat16)
        else:
            y = finalize_out_dtype(y, cfg)
        
        pred[y >= 0] = 1
        pred[y < 0] = 0
        return pred

    # ---- RBF kernel: K(X, X_ref), then pred = K @ w + b ----
    elif model["kernelFunction"] == "gaussianKernel":
        # Model parts
        x_q  = model["X"]
        bias  = model["b"]
        sigma = model["sigma"]          # this is γ = 1/(2σ^2) in the training flow

        # Quantize model parts to match input dtypes
        bias_q  = finalize_out_dtype(bias,  cfg)
        sigma_q = finalize_out_dtype(sigma, cfg)

        # Output and accumulator dtypes
        out_torch  = torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()
        acc_dtype  = _acc_dtype_main(cfg)
        k_matrix   = torch.zeros((x.shape[0], x_q.shape[0]), dtype=out_torch, device=x.device)


        # ------------- scalar path -------------
        if not cfg.vec:
            
            for i in range(x.shape[0]):
                for j in range(x_q.shape[0]):
                    sum = torch.zeros(1, dtype=acc_dtype, device=x.device)   # accumulator for ||x_i - xref_j||^2
                    for k in range(x.shape[1]):
                        a = x[i, k]
                        b = x_q[j, k]
                        # Optional cast stage
                        if cfg.cast and cfg.cast_to is not None:
                            a = _cast_apply(a, cfg.cast_to, cfg.mantissa_bits)
                            b = _cast_apply(b, cfg.cast_to, cfg.mantissa_bits)
                        # diff for RBF distance
                        d = a - b
                        d = finalize_out_dtype(d, cfg)  # diff always finalized
                        # accumulate d*d using centralized MAC policy
                        sum = acc_add(sum, d, d, cfg, leftover=False, in_main=True)
                        # finalize each step when not in HW-mixed (mirrors FIR)
                        if not cfg.hw_mixed:
                            sum = finalize_out_dtype(sum, cfg)

                    if cfg.hw_mixed:
                        sum = finalize_out_dtype(sum, cfg)
                    # Multiply by -gamma, then finalize before exp
                    t = sum * (-sigma_q)
                    # finalize before exp
                    t = finalize_out_dtype(t, cfg)
                    # Fast exp approximation (same as your gist path). Compute in float, then finalize to OUT.
                    val = fastexp_gist(t)
                    
                    k_matrix[i, j] = finalize_out_dtype(
                        torch.tensor(val, device=x.device), cfg
                    )
                    
        else:  # vectorized version (FIR-style policy)
            device = x.device
            vec_step = 4 if cfg.dt_x is DKind.FP8_CUSTOM else 2
            feat_dim = x_q.shape[1]
            processed = (feat_dim // vec_step) * vec_step  # floor to multiple of vec_step

            acc_dtype = _acc_dtype_main(cfg)  # main-loop accumulator dtype

            for i in range(x.shape[0]):
                for z in range(x_q.shape[0]):

                    if cfg.mixed_vec:
                        # accumulate reduced lane sum directly
                        sum_val = torch.zeros(1, dtype=acc_dtype, device=device)
                    else:
                        # keep per-lane partials then fold
                        temp = torch.zeros(vec_step, dtype=acc_dtype, device=device)

                    # -------- main chunks
                    for j in range(0, processed, vec_step):
                        # slice lanes
                        a = x[i, j:j+vec_step]
                        b = x_q[z, j:j+vec_step]

                        # diffs for RBF distance
                        d = a - b

                        # BECAUSE OF MIXED VEC, WE NEED TO QUANTIZE to the input or fil dtype
                        d = _cast_apply(d, cfg.dt_w, cfg.mantissa_bits)
                        
                        # accumulate d*d
                        if cfg.mixed_vec:
                            # lane-wise (d*d) then intra-lane reduction to sum_val
                            # do products via acc_add (so product-side rules apply)
                            lane_acc = torch.zeros(vec_step, dtype=acc_dtype, device=device)
                            for k in range(vec_step):
                                lane_acc[k] = acc_add(
                                    torch.zeros(1, dtype=acc_dtype, device=device),
                                    d[k], d[k], cfg, leftover=False, in_main=True
                                ).squeeze(0)

                            # lane_acc holds 4 lane partials at fp32 (or whatever promote_for_mac chose)
                            if vec_step == 2:
                                t = lane_acc[0] + lane_acc[1]
                            else:
                                t01 = lane_acc[0] + lane_acc[1]
                                t23 = lane_acc[2] + lane_acc[3]
                                t   = t01 + t23

                            # If your model requires rounding per fold, do it here; otherwise postpone:
                            # t = finalize_out_dtype(t, cfg)     # only if HW does per-step rounding

                            sum_val = sum_val + t
                            # keep sum_val finalized as we go
                            # sum_val = finalize_out_dtype(sum_val, cfg)
                        else:
                            # keep per-lane partials in temp[k] using acc_add
                            for k in range(vec_step):
                                tk = temp[k].unsqueeze(0)
                                tk = acc_add(tk, d[k], d[k], cfg, leftover=False, in_main=True)
                                tk = finalize_out_dtype(tk, cfg)
                                temp[k] = tk.squeeze(0)

                    # -------- leftovers
                    for j in range(processed, feat_dim):
                        a0 = x[i, j]
                        b0 = x_q[z, j]

                        d0 = a0 - b0

                        d0 = _cast_apply(d0, cfg.dt_w, cfg.mantissa_bits)

                        if cfg.mixed_vec:
                            sum_val = acc_add(sum_val, d0, d0, cfg, leftover=True, in_main=False)
                            # sum_val = finalize_out_dtype(sum_val, cfg)
                        else:
                            t0 = temp[0].unsqueeze(0)
                            t0 = acc_add(t0, d0, d0, cfg, leftover=True, in_main=False)
                            t0 = finalize_out_dtype(t0, cfg)
                            temp[0] = t0.squeeze(0)

                    # -------- fold lanes if needed
                    if not cfg.mixed_vec:
                        s = torch.zeros(1, dtype=acc_dtype, device=device)
                        for k in range(vec_step):
                            s = s + temp[k]
                            s = finalize_out_dtype(s, cfg)
                    else:
                        s = sum_val

                    # finalize sum-of-squares
                    s = finalize_out_dtype(s, cfg)

                    # multiply by -sigma (gamma)
                    tmp = s * (-sigma_q)
                    tmp = finalize_out_dtype(tmp, cfg)

                    # exp: fastexp_gist works on float, then finalize result
                    kval = fastexp_gist(float(tmp.item()))
                    k_t = torch.tensor(kval, dtype=acc_dtype, device=device)
                    k_t = finalize_out_dtype(k_t, cfg)

                    k_matrix[i, z] = k_t

        # Final prediction stage: pred = K @ w + b
        cfg = SVMConfig(
        mac=mac_flag, vec=False, cast=cast_flag, hw_mixed=False, mixed_vec=mixed_vec_flag,
        mantissa_bits=mantissa_bits,
        dt_x=dt[0], dt_w=dt[1], dt_out=dt[2], cast_to=cast_to_kind
        )
        pred = (
            matrix_mult(
                k_matrix,
                model["w"],
                cfg=cfg,
            )
            + bias_q
        )
        pred = finalize_out_dtype(pred, cfg)

    # Convert predictions to binary labels: 1 if >= 0, else 0
    pred[pred >= 0] = 1
    pred[pred < 0] = 0
    return pred
# ----------------------------
# I/O for C headers (same format as your original)
# ----------------------------

def write_matrix(
    matrix_to_write: torch.Tensor,
    name: str,
    length: int,
    file_pointer: object,
    float_type: torch.dtype,
):
    """
    Writes a matrix to a file in a specific format based on its name and type.

    Parameters
    ----------
    matrix_to_write : torch.Tensor
        The matrix to be written.
    name : str
        The name of the matrix.
    length : int
        The length of the matrix (usually size along one dimension).
    file_pointer : object
        The file pointer where the matrix should be written.
    float_type : torch.dtype
        The data type for the matrix (e.g., torch.float32, torch.float16).

    Returns
    -------
    None
    """
    matrix_string = ""

    if "check" in name:
        file_pointer.write("PI_L2 static int %s[%s] = {" % (name, length))
    elif "data_model" in name:
        file_pointer.write("DATA_LOCATION INP_TYPE %s[%s] = {" % (name, length))
    elif "sv_coef" in name:
        file_pointer.write("DATA_LOCATION FIL_TYPE %s[%s] = {" % (name, length))
    elif "bias" in name:
        file_pointer.write("DATA_LOCATION FIL_TYPE %s[%s] = {" % (name, length))
    else:
        file_pointer.write("DATA_LOCATION static INP_TYPE %s[%s] = {" % (name, length))

    if float_type == torch.float32:
        name = ")"
    elif float_type == torch.float16:
        name = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        name = ", dtype=torch.bfloat16)"
    sz0, sz1 = matrix_to_write.shape
    for i in range(sz0):
        for j in range(sz1):
            matrix_string += (
                str(matrix_to_write[i][j].item())
                .replace("tensor(", "")
                .replace(name, "")
            )
            matrix_string += ","
    file_pointer.write("%s" % matrix_string)
    file_pointer.write("};\n")

def save_data_into_hfile(
    kernel_type: str,
    x_test: torch.Tensor,
    x_ref: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    y_test: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    param: Optional[Dict[str, torch.Tensor]] = None,
    accuracy: Optional[float] = None
) -> None:
    """
    Saves the SVM model data into a header file.

    Parameters
    ----------
    kernel_type : str
        Type of the kernel function used in the SVM ("linear" or "rbf").
    x_test : torch.Tensor
        Test dataset.
    x_ref : torch.Tensor
        Reference data for the model.
    w : torch.Tensor
        Model coefficients.
    b : torch.Tensor
        Bias term for the model.
    y_test : torch.Tensor
        Test labels.
    sigma : Optional[torch.Tensor], optional
        The sigma value for the RBF kernel (default is None).
    param : Optional[Dict[str, torch.Tensor]], optional
        A dictionary of additional parameters (default is None).
    accuracy : Optional[float], optional
        The accuracy of the model (default is None).
    """
    # Generate header files

    if kernel_type == "linear":
        with open("configs.h", "w", encoding="utf-8") as config_file:
            config_file.write("#define LINEAR_KERNEL\n")
            if x_test.shape[1] > 32:
                config_file.write("#define FDIM_GT_32 1\n")
            else:
                config_file.write("// No specific configuration for FDIM_GT_32\n")
            config_file.close()
    elif kernel_type == "rbf":
        with open("configs.h", "w", encoding="utf-8") as config_file:
            config_file.write("#define RBF_KERNEL\n")
            config_file.close()

    with open("modelSVM.h", "w", encoding="utf-8") as f:

        if kernel_type == "linear":
            f.write(
                '\
#ifndef MODELSVM_H_ \n\
#define MODELSVM_H_\n\
#define KERNEL_TYPE_    0\n\
#define GAMMA1_         0.0f\n\
#define SVS_            %s\n\
#define COEF_DIM_      1\n\
#define F_DIM_             %s\n\
#define N_CLASS_       2\n\
#define N_DEC_VALUES_  %s\n\
#include <stdio.h>\n\
#include "defines.h"\n\n'
                % (x_test.shape[0], x_test.shape[1], x_test.shape[0])
            )
            write_matrix(x_test, "data_model", "SVS_*F_DIM_", f, x_test.dtype)
            write_matrix(w, "sv_coef", "COEF_DIM_*F_DIM_", f, w.dtype)
            write_matrix(b, "bias", "1", f, b.dtype)
            write_matrix(x_ref, "X_ref", "1", f, torch.float16)
            write_matrix(y_test, "check_result", "N_DEC_VALUES_", f, torch.float32)



        elif kernel_type == "rbf":
            f.write(
                '\
#ifndef MODELSVM_H_ \n\
#define MODELSVM_H_\n\
#define KERNEL_TYPE_    2\n\
#define GAMMA1_         %s\n\
#define SVS_            %s\n\
# define COEF_DIM_      %s\n\
# define F_DIM_             %s\n\
# define N_CLASS_       2\n\
# define N_DEC_VALUES_  %s\n\
#include <stdio.h>\n\
#include "defines.h"\n\n'
                % (
                    sigma.item(),
                    x_test.shape[0],
                    param["X"].shape[0],
                    x_test.shape[1],
                    x_test.shape[0],
                )
            )
            write_matrix(x_test, "data_model", "SVS_*F_DIM_", f, x_test.dtype)
            write_matrix(x_ref, "X_ref", "COEF_DIM_*F_DIM_", f, x_ref.dtype)
            write_matrix(w, "sv_coef", "COEF_DIM_*F_DIM_", f, w.dtype)
            write_matrix(b, "bias", "1", f, b.dtype)
            write_matrix(y_test, "check_result", "N_DEC_VALUES_", f, torch.float32)
        f.write(
            "\
#define ACCURACY_     %s\n\
    #endif \n"
            % (accuracy if accuracy is not None else 0.0)
        )
        f.close()

def calculate_accuracy(y_test: torch.Tensor, y_pred: torch.Tensor, output_file: Optional[str]) -> float:

    """
    Calculates and prints the accuracy of predictions.

    Parameters
    ----------
    y_test : torch.Tensor
        The ground truth labels.
    y_pred : torch.Tensor
        The predicted labels.
    description : str
        A description of the accuracy being calculated (e.g., data type or kernel type).
    output_file (str): The file path to save the error metrics.
    Returns
    -------
    float
        The calculated accuracy as a float value.
    """
    accuracy = (torch.sum(y_test == y_pred) / y_test.shape[0]) * 100

    # If an output file is provided, write the results to it
    if output_file:
        with open(output_file, "w") as f:
            f.write(f"{accuracy.item()}\n")
    
        # Build JSON filename based on output_file
        json_output_file = output_file.replace(".txt", ".json")

        # Save metrics as JSON too
        metrics_dict = {
            "ACC": accuracy.item()
        }

        with open(json_output_file, "w") as jf:
            json.dump(metrics_dict, jf, indent=4)
    return accuracy.item()

def get_initial_config() -> Tuple[str, int, List[str], str, str]:
    """
    Parse the command-line arguments and return the initial configuration.

    Returns
    -------
    Tuple[str, int, List[str], str, str]
        kernel_type: The type of the kernel function used in SVM.
        input_size: The size of the dataset.
        bits: List containing the float type used for data and coefficients.
        dataset: The dataset to be used.
        mac_flag: A flag indicating if MAC is enabled.
        vec_flag: A flag indicating if vectorization is enabled.
        exploration_flag: A flag indicating if exploration mode is enabled.
        hwmixed_flag: A flag indicating if hardware mixed precision is enabled.
    """
    # get input size and datatypes
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--input_size", default=100)
    parser.add_argument("--mac_flag", default=False)
    parser.add_argument("--vec_flag", default=False)
    parser.add_argument("--mantissa_bits", default=2)
    parser.add_argument("--exploration_flag", default=False)
    parser.add_argument("--hwmixed_flag", default=False)
    parser.add_argument(
        "--float_type", default="FP32"
    )  # input data, x_ref, and (coef + bias)
    args = parser.parse_args()

    kernel_type = str(args.kernel)
    dataset = str(args.dataset)
    bits = [s.strip() for s in args.float_type.split(",")]
    input_size = int(args.input_size)
    mac_flag = str2bool(args.mac_flag)
    vec_flag = str2bool(args.vec_flag)
    mantissa_bits = int(args.mantissa_bits)
    exploration_flag = str2bool(args.exploration_flag)
    hwmixed_flag = str2bool(args.hwmixed_flag)
    return kernel_type, input_size, bits, dataset, mac_flag, vec_flag, exploration_flag, mantissa_bits, hwmixed_flag


def main():
    kernel_type, input_size, bits, dataset, mac_flag, vec_flag, exploration_flag, mantissa_bits, hwmixed_flag = get_initial_config()

    if not exploration_flag:
        check_pulp_warnings(bits, mac_flag, hwmixed_flag, vec_flag)
    
    # dtypes & derived flags
    datatypes = select_dtypes(bits, 3)  # [X, W, OUT]
    
    # Check if mixed vectorization flag must be enabled
    mixed_vec_flag = check_vec_flag(bits, vec_flag)
    if not exploration_flag:
        print(f"Mixed Vectorization Flag: {mixed_vec_flag}")
        # read dataset
        print(f"Loading dataset {dataset} with input size {input_size}...")
    x_train, x_test, y_train, y_test = read_dataset(dataset, input_size)

    output_folder = os.path.join(os.getcwd(), "exploration",kernel_type, dataset)
    os.makedirs(output_folder, exist_ok=True)

    if kernel_type == "linear":
        c = 1

        if not exploration_flag: print("Train is started (Linear)...")
        model = svm_train(x_train, y_train, c, "linearKernel")
        if not exploration_flag: print("Train is finished...")

        # FP32 ref
        param = {"kernelFunction": "linearKernel",
                "b": torch.tensor(model["b"], dtype=torch.float32).reshape(1,1),
                "w": torch.from_numpy(model["w"]).reshape(-1,1).float()}
        if not exploration_flag:
            print("Predict is started (FP32)...")
            y_pred_fp32 = svm_predict(param, x_test, [DKind.FP32, DKind.FP32, DKind.FP32])
            acc = calculate_accuracy(y_test, y_pred_fp32, None)
            print(f"Accuracy in FP32: {acc}")

        # typed run
        cast_flag = check_cast(datatypes)
        cast_to = bits[-1] if len(bits) else "FP32"

        x_typed = matrix_init(x_test, datatypes[0], mantissa_bits)
        w_typed = matrix_init(param["w"], datatypes[1], mantissa_bits)
        b_typed = matrix_init(param["b"], datatypes[1], mantissa_bits)

        param_typed = {"kernelFunction": "linearKernel", "b": b_typed, "w": w_typed}
        if not exploration_flag:
            if DKind.FP8_CUSTOM in datatypes:
                if cast_flag:
                    print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
                    print(f"and mantissa = {mantissa_bits} bits and casting to {cast_to}")
                else:
                    print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
                    print(f"and mantissa = {mantissa_bits} bits")
            else:
                if cast_flag:
                    print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
                    print(f"and casting to {cast_to}")
                else:
                    print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")

        y_pred = svm_predict(
            param_typed, x_typed, datatypes,
            mac_flag=mac_flag, vec_flag=vec_flag, cast_flag=cast_flag, cast_to=cast_to,
            mantissa_bits=mantissa_bits, hwmixed_flag=hwmixed_flag, mixed_vec_flag=mixed_vec_flag
        )
        output_file = os.path.join(output_folder, f"acc__{input_size}_{dkind_name(datatypes[0])}_{dkind_name(datatypes[1])}_{dkind_name(datatypes[2])}_{mantissa_bits}.txt") if exploration_flag else None
        acc = calculate_accuracy(y_test, y_pred, output_file)
        if not exploration_flag:
            if DKind.FP8_CUSTOM in datatypes:
                print(f"Accuracy in {dkind_name(datatypes[0])},{dkind_name(datatypes[1])},{dkind_name(datatypes[2])} (mantissa {mantissa_bits}): {acc}")
            else:
                print(f"Accuracy in {dkind_name(datatypes[0])},{dkind_name(datatypes[1])},{dkind_name(datatypes[2])}: {acc}")
            # emit headers (X_ref unused for linear; keep shape (1,1) to satisfy writer)
            x_ref = torch.zeros((1,1), dtype=torch.float16)
            save_data_into_hfile("linear", x_typed, x_ref, w_typed, b_typed, y_test, accuracy=acc)


    # ---------------------- RBF Kernel
    elif kernel_type == "rbf":

        if not exploration_flag: print("Train is started (RBF)...")
        
        if dataset == "custom":
            # gamma = 1 / (n_features * var)
            gamma = 1.0 / (x_train.shape[1] * np.var(x_train))
        else:
            # keep your old sigma_init path (4.4721) but convert to gamma
            sigma_init = float(4.4721)
            gamma = 1.0 / (2.0 * sigma_init * sigma_init)

        # convert gamma back to sigma for TRAINING (svm_train expects σ, not γ)
        sigma_train = float(np.sqrt(0.5 / gamma))
        model = svm_train(x_train, y_train, c = 2, kernel_function= "gaussianKernel", args=(sigma_train,))
        if not exploration_flag: print("Train is finished...")

        # assemble model tensors
        w = (model["alphas"] * model["y"]).astype(np.float32).reshape(-1,1)
        b = np.array([[model["b"]]], dtype=np.float32)
        X_ref = model["X"].astype(np.float32)
        
        sigma0 = torch.as_tensor(model["args"][0], dtype=torch.float32,
                                device=model["args"][0].device if torch.is_tensor(model["args"][0]) else None)

        sigma = 0.5 / (sigma0 * sigma0)        # == 1 / (2 * sigma^2)
        param = {
            "kernelFunction": "gaussianKernel",
            "X": torch.from_numpy(X_ref).float(),
            "b": torch.from_numpy(b).float(),
            "w": torch.from_numpy(w).float(),
            "sigma": sigma,
        }
        
        if not exploration_flag:
            print("Predict is started (FP32)...")
            y_pred_fp32 = svm_predict(param, x_test, [DKind.FP32, DKind.FP32, DKind.FP32])
            acc = calculate_accuracy(y_test, y_pred_fp32, None)
            print(f"Accuracy in FP32: {acc}")
                
        # typed run
        datatypes = select_dtypes(bits, 3)
        cast_flag = check_cast(datatypes)
        cast_to = bits[-1] if len(bits) else "FP32"
        
        if not exploration_flag:
            if DKind.FP8_CUSTOM in datatypes:
                if cast_flag:
                    print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
                    print(f"and mantissa = {mantissa_bits} bits and casting to {cast_to}")
                else:
                    print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
                    print(f"and mantissa = {mantissa_bits} bits")
            else:
                if cast_flag:
                    print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")
                    print(f"and casting to {cast_to}")
                else:
                    print(f"Running with {dkind_name(datatypes[0])}, {dkind_name(datatypes[1])}, {dkind_name(datatypes[2])}")

        x_typed  = matrix_init(x_test, datatypes[0], mantissa_bits)
        xr_typed = matrix_init(param["X"], datatypes[0], mantissa_bits)
        w_typed  = matrix_init(param["w"], datatypes[1], mantissa_bits)
        b_typed  = matrix_init(param["b"], datatypes[1], mantissa_bits)

        # sigma = 1 / (2 * (model["args"][0] ** 2))
        # sigma_init was passed into training as model["args"][0]
        # sigma0 may be a Python float or a Tensor; this keeps device/dtype and grad
        sigma_typed = matrix_init(sigma.reshape(1, 1), datatypes[1], mantissa_bits)

        param_typed = {
            "kernelFunction": "gaussianKernel",
            "X": xr_typed,
            "b": b_typed,
            "w": w_typed,
            "sigma": sigma_typed,
        }

        if not exploration_flag:
            print(f"Predict is started in {dkind_name(datatypes[0])},{dkind_name(datatypes[1])},{dkind_name(datatypes[2])} with mantissa bits {mantissa_bits}...")

        y_pred = svm_predict(
            param_typed,
            x_typed,
            dt=datatypes,
            mac_flag=mac_flag,
            vec_flag=vec_flag,
            cast_flag=cast_flag,
            cast_to=cast_to,
            mantissa_bits=mantissa_bits,
            hwmixed_flag=hwmixed_flag,
            mixed_vec_flag=mixed_vec_flag
        )
        
        # calculate the accuracy of the prediction
        output_file = os.path.join(output_folder, f"acc__{input_size}_{dkind_name(datatypes[0])}_{dkind_name(datatypes[1])}_{dkind_name(datatypes[2])}_{mantissa_bits}.txt") if exploration_flag else None
        acc = calculate_accuracy(y_test, y_pred, output_file)
        if not exploration_flag:
            print(f"Accuracy in {dkind_name(datatypes[0])},{dkind_name(datatypes[1])},{dkind_name(datatypes[2])} (mantissa {mantissa_bits}): {acc}")
            save_data_into_hfile("rbf", x_typed, xr_typed, w_typed, b_typed, y_test, sigma=sigma, param=param_typed, accuracy=acc)
            print("############################## Done! ###################################")
    else:
        print("kernel type is invalid")


if __name__ == "__main__":
    main()