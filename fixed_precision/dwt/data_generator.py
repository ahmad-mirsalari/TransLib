import copy

import pandas as pd
import os
import torch
import numpy as np
import argparse

import sys
import matplotlib.pyplot as plt
import pywt
import scipy.io
import math
from data import *


def dwt_step(input_sig, output_sig, n, idx_level, NC, Lo, Hi, mac_flag, vec_flag, dt):
    ii = 0
    core_id = 0
    NUM_CORES = 1
    if vec_flag == "false":
        for i in range((NC // 2) - 1):
            a, b, j = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt), 0
            for j in range(2 * (i + 1)):
                inp = input_sig[2 * (i + 1) - j - 1]
                lo = Lo[j]
                hi = Hi[j]
                if mac_flag == "true":
                    ## cast to fp32
                    inp = inp.type(torch.float32)
                    lo = lo.type(torch.float32)
                    hi = hi.type(torch.float32)
                    b = b.type(torch.float32)
                    a = a.type(torch.float32)
                a = a + inp * lo
                b = b + inp * hi
                if mac_flag == "true":
                    a = a.type(dt)
                    b = b.type(dt)
            output_sig[ii + core_id] = a
            output_sig[ii + core_id + idx_level] = b
            ii += NUM_CORES
        for i in range((NC - 1), n, 2):
            a, b = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt),
            for j in range(NC):
                inp = input_sig[i - j]
                lo = Lo[j]
                hi = Hi[j]
                if mac_flag == "true":
                    inp = inp.type(torch.float32)
                    lo = lo.type(torch.float32)
                    hi = hi.type(torch.float32)
                    b = b.type(torch.float32)
                    a = a.type(torch.float32)
                a = a + inp * lo
                b = b + inp * hi
                if mac_flag == "true":
                    a = a.type(dt)
                    b = b.type(dt)
            output_sig[ii + core_id] = a
            output_sig[ii + core_id + idx_level] = b
            ii += NUM_CORES
        if n % 2 == 0:  # even
            for i in range((NC // 2) - 1):
                a, b, j = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt), 0
                for j in range(NC - 2 * (i + 1)):
                    inp = input_sig[n - j - 1]
                    lo = Lo[2 * (i + 1) + j]
                    hi = Hi[2 * (i + 1) + j]
                    if mac_flag == "true":
                        inp = inp.type(torch.float32)
                        lo = lo.type(torch.float32)
                        hi = hi.type(torch.float32)
                        b = b.type(torch.float32)
                        a = a.type(torch.float32)

                    a = a + inp * lo
                    b = b + inp * hi
                    if mac_flag == "true":
                        a = a.type(dt)
                        b = b.type(dt)
                output_sig[ii + core_id] = a
                output_sig[ii + core_id + idx_level] = b
                ii += NUM_CORES
        if n % 2 == 1:  # odd
            for i in range(NC // 2):
                a, b = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt)
                for j in range(NC - 2 * (i + 1) + 1):
                    inp = input_sig[n - j - 1]
                    lo = Lo[2 * (i + 1) + j - 1]
                    hi = Hi[2 * (i + 1) + j - 1]
                    if mac_flag == "true":
                        inp = inp.type(torch.float32)
                        lo = lo.type(torch.float32)
                        hi = hi.type(torch.float32)
                        b = b.type(torch.float32)
                        a = a.type(torch.float32)
                    a = a + inp * lo
                    b = b + inp * hi
                    if mac_flag == "true":
                        a = a.type(dt)
                        b = b.type(dt)
                output_sig[ii + core_id] = a
                output_sig[ii + core_id + idx_level] = b
                ii += NUM_CORES
        next_inputs = (n + NC - 1) // 2
        for i in range(0, (next_inputs // 2) * 2, 2):
            a = output_sig[i]
            b = output_sig[i + 1]
            input_sig[i] = a
            input_sig[i + 1] = b
        if next_inputs & 0x1:
            input_sig[next_inputs - 1] = output_sig[next_inputs - 1]
    else:
        for i in range((NC // 2) - 1):
            a, b, j = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt), 0
            a1, b1 = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt)
            for j in range(2 * (i + 1) - 1, 0, -2):
                inp = input_sig[2 * (i + 1) - j - 1]
                inp1 = input_sig[(2 * (i + 1) - j - 1) + 1]
                lo = Lo[(NC - 1) - j]
                hi = Hi[(NC - 1) - j]
                lo1 = Lo[(NC - 1) - j + 1]
                hi1 = Hi[(NC - 1) - j + 1]
                if mac_flag == "true":
                    ## cast to fp32
                    inp = inp.type(torch.float32)
                    lo = lo.type(torch.float32)
                    hi = hi.type(torch.float32)
                    b = b.type(torch.float32)
                    a = a.type(torch.float32)

                    inp1 = inp1.type(torch.float32)
                    lo1 = lo1.type(torch.float32)
                    hi1 = hi1.type(torch.float32)
                    b1 = b1.type(torch.float32)
                    a1 = a1.type(torch.float32)
                a = a + inp * lo
                b = b + inp * hi
                a1 = a1 + inp1 * lo1
                b1 = b1 + inp1 * hi1

                if mac_flag == "true":
                    a = a.type(dt)
                    b = b.type(dt)
                    a1 = a1.type(dt)
                    b1 = b1.type(dt)
            output_sig[ii + core_id] = a + a1
            output_sig[ii + core_id + idx_level] = b + b1
            ii += NUM_CORES
        for i in range((NC - 1), n, 2):
            a, b = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt)
            a1, b1 = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt)
            for j in range(NC - 1, 0, -2):
                inp = input_sig[i - j]
                inp1 = input_sig[i - j + 1]
                lo = Lo[(NC - 1) - j]
                hi = Hi[(NC - 1) - j]
                lo1 = Lo[(NC - 1) - j + 1]
                hi1 = Hi[(NC - 1) - j + 1]
                if mac_flag == "true":
                    inp = inp.type(torch.float32)
                    lo = lo.type(torch.float32)
                    hi = hi.type(torch.float32)
                    b = b.type(torch.float32)
                    a = a.type(torch.float32)
                    inp1 = inp1.type(torch.float32)
                    lo1 = lo1.type(torch.float32)
                    hi1 = hi1.type(torch.float32)
                    b1 = b1.type(torch.float32)
                    a1 = a1.type(torch.float32)
                a = a + inp * lo
                b = b + inp * hi
                a1 = a1 + inp1 * lo1
                b1 = b1 + inp1 * hi1
                if mac_flag == "true":
                    a = a.type(dt)
                    b = b.type(dt)
                    a1 = a1.type(dt)
                    b1 = b1.type(dt)
            if NC & 0x00000001:
                inp = input_sig[i - 0]
                lo = Lo[NC - 1]
                hi = Hi[NC - 1]
                if mac_flag == "true":
                    inp = inp.type(torch.float32)
                    lo = lo.type(torch.float32)
                    hi = hi.type(torch.float32)
                    b = b.type(torch.float32)
                    a = a.type(torch.float32)

                a = a + inp * lo
                b = b + inp * hi
                if mac_flag == "true":
                    a = a.type(dt)
                    b = b.type(dt)
            output_sig[ii + core_id] = a + a1
            output_sig[ii + core_id + idx_level] = b + b1
            ii += NUM_CORES
        if n % 2 == 0:  # even
            for i in range((NC // 2) - 1):
                a, b, j = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt), 0
                a1, b1 = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt)
                for j in range((NC - 2 * (i + 1)) - 1, 0, -2):
                    inp = input_sig[n - j - 1]
                    lo = Lo[(NC - 1) - (2 * (i + 1) + j)]
                    hi = Hi[(NC - 1) - (2 * (i + 1) + j)]
                    inp1 = input_sig[n - j - 1 + 1]
                    lo1 = Lo[(NC - 1) - (2 * (i + 1) + j) + 1]
                    hi1 = Hi[(NC - 1) - (2 * (i + 1) + j) + 1]
                    if mac_flag == "true":
                        inp = inp.type(torch.float32)
                        lo = lo.type(torch.float32)
                        hi = hi.type(torch.float32)
                        b = b.type(torch.float32)
                        a = a.type(torch.float32)

                        inp1 = inp1.type(torch.float32)
                        lo1 = lo1.type(torch.float32)
                        hi1 = hi1.type(torch.float32)
                        b1 = b1.type(torch.float32)
                        a1 = a1.type(torch.float32)

                    a = a + inp * lo
                    b = b + inp * hi

                    a1 = a1 + inp1 * lo1
                    b1 = b1 + inp1 * hi1

                    if mac_flag == "true":
                        a = a.type(dt)
                        b = b.type(dt)
                        a1 = a1.type(dt)
                        b1 = b1.type(dt)

                if NC & 0x00000001:
                    inp = input_sig[i - 0]
                    lo = Lo[(NC - 1) - (2 * (i + 1))]
                    hi = Hi[(NC - 1) - (2 * (i + 1))]
                    if mac_flag == "true":
                        inp = inp.type(torch.float32)
                        lo = lo.type(torch.float32)
                        hi = hi.type(torch.float32)
                        b = b.type(torch.float32)
                        a = a.type(torch.float32)

                    a = a + inp * lo
                    b = b + inp * hi
                    if mac_flag == "true":
                        a = a.type(dt)
                        b = b.type(dt)

                output_sig[ii + core_id] = a + a1
                output_sig[ii + core_id + idx_level] = b + b1
                ii += NUM_CORES
        if n % 2 == 1:  # odd
            for i in range(NC // 2):
                a, b = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt)
                a1, b1 = torch.zeros(1, dtype=dt), torch.zeros(1, dtype=dt)
                for j in range(NC - 2 * (i + 1), 0, -2):
                    inp = input_sig[n - j - 1]
                    lo = Lo[(NC - 1) - (2 * (i + 1) + j - 1)]
                    hi = Hi[(NC - 1) - (2 * (i + 1) + j - 1)]
                    inp1 = input_sig[n - j - 1 + 1]
                    lo1 = Lo[(NC - 1) - (2 * (i + 1) + j - 1) + 1]
                    hi1 = Hi[(NC - 1) - (2 * (i + 1) + j - 1) + 1]
                    if mac_flag == "true":
                        inp = inp.type(torch.float32)
                        lo = lo.type(torch.float32)
                        hi = hi.type(torch.float32)
                        b = b.type(torch.float32)
                        a = a.type(torch.float32)
                        inp1 = inp1.type(torch.float32)
                        lo1 = lo1.type(torch.float32)
                        hi1 = hi1.type(torch.float32)
                        b1 = b1.type(torch.float32)
                        a1 = a1.type(torch.float32)
                    a = a + inp * lo
                    b = b + inp * hi
                    a1 = a1 + inp1 * lo1
                    b1 = b1 + inp1 * hi1
                    if mac_flag == "true":
                        a = a.type(dt)
                        b = b.type(dt)

                        a1 = a1.type(dt)
                        b1 = b1.type(dt)
                if not NC & 0x00000001:
                    inp = input_sig[n - 0 - 1]
                    lo = Lo[(NC - 1) - (2 * (i + 1) + 0 - 1)]
                    hi = Hi[(NC - 1) - (2 * (i + 1) + 0 - 1)]
                    if mac_flag == "true":
                        inp = inp.type(torch.float32)
                        lo = lo.type(torch.float32)
                        hi = hi.type(torch.float32)
                        b = b.type(torch.float32)
                        a = a.type(torch.float32)

                    a = a + inp * lo
                    b = b + inp * hi
                    if mac_flag == "true":
                        a = a.type(dt)
                        b = b.type(dt)

                output_sig[ii + core_id] = a + a1
                output_sig[ii + core_id + idx_level] = b + b1
                ii += NUM_CORES
        next_inputs = (n + NC - 1) // 2
        for i in range(0, (next_inputs // 2) * 2, 2):
            a = output_sig[i]
            b = output_sig[i + 1]
            input_sig[i] = a
            input_sig[i + 1] = b
        if next_inputs & 0x1:
            input_sig[next_inputs - 1] = output_sig[next_inputs - 1]
    return input_sig, output_sig


def write_matrix(matrix_to_write, name, len, file_pointer, float_type):
    matrix_string = ''
    if 'ref' in name:
        file_pointer.write("PI_L2 DATA_TYPE %s[%s] = {" % (name, len))
    else:
        file_pointer.write("DATA_LOCATION DATA_TYPE  %s[%s] = {" % (name, len))

    if float_type == torch.float32:
        rem_part = ")"
    elif float_type == torch.float16:
        rem_part = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        rem_part = ", dtype=torch.bfloat16)"
    sz0 = matrix_to_write.shape[0]
    for i in range(sz0):
        matrix_string += str(matrix_to_write[i].item()).replace('tensor(', '').replace(rem_part, '')
        matrix_string += ','
    file_pointer.write("%s" % matrix_string)
    file_pointer.write("};\n")


def matrix_init(IN, dt):
    temp = torch.zeros((IN.shape[0]), dtype=dt)
    # iterate through rows of IN
    for i in range(int(IN.shape[0])):
        temp[i] = IN[i]
    return temp


def mean_squared_error(true, pred):
    squared_error = torch.square(true - pred)
    sum_squared_error = torch.sum(squared_error)
    size = true.size(dim=0)
    mse_loss = sum_squared_error / size
    return mse_loss


def relative_absolute_error(true, pred):
    true_mean = torch.mean(true)
    squared_error_num = torch.sum(torch.abs(true - pred))
    squared_error_den = torch.sum(torch.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss


def error_metric(ref, res):

    # calculate manually because metrics doesn't supprt bfloat16
    d = ref - res
    mse_f = torch.mean(d**2)
    mae_f = torch.mean(abs(d))
    rmse_f = torch.sqrt(mse_f)
    r2_f = 1-(torch.sum(d**2)/torch.sum((ref-torch.mean(ref))**2))
    print("Results of metrics:")
    print("MAE:",mae_f.item())
    print("MSE:", mse_f.item())
    print("RMSE:", rmse_f.item())
    print("R-Squared:", r2_f.item())
    rae = relative_absolute_error(ref, res)
    print("RAE is", rae.item())

def get_inital_config():
    # get input size and datatypes
    # python data_generator.py --input_size=256 --levels=4 --mode=sym4 --float_type=FP16 --flag=true
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--levels', default=4)
    parser.add_argument('--mode', default='sym4')
    parser.add_argument('--MAC_flag', default="true")
    parser.add_argument('--vec_flag', default="false")
    parser.add_argument('--float_type', default='FP32')  # input data,filters,output
    args = parser.parse_args()

    input_size = int(args.input_size)
    levels = int(args.levels)
    family = str(args.mode)
    mac_flag = str(args.MAC_flag)
    vec_flag = str(args.vec_flag)
    bits = args.float_type.split(",")
    return input_size, levels, family, bits, mac_flag, vec_flag


def load_data(input_size):
    # load data
    dataset = torch.from_numpy(np.asarray(Input_Signal[0:input_size]))
    x_test = dataset.type(torch.float32)
    return x_test


def get_filters(x_test, levels, family):
    # execute dwt in FP32 as reference
    wavelet = pywt.Wavelet(family)
    coeff = pywt.wavedec(x_test, wavelet, mode='zero', level=levels)

    LO = wavelet.dec_lo  # lowpass filter
    HI = wavelet.dec_hi  # highpass filter
    NC = len(LO)  # length of the filter

    # change filter type to torch
    LO_ = torch.from_numpy(np.asarray(LO))
    Lo = LO_.type(torch.float32)

    HI_ = torch.from_numpy(np.asarray(HI))
    Hi = HI_.type(torch.float32)

    return Hi, Lo, NC, coeff


def get_outdim(coeff):
    # define the output length
    ref = []
    for i in range(len(coeff)):
        ref = np.concatenate((ref, coeff[i]))
    return len(ref)


def select_dtypes(user_dtypes, num_param):
    types_dict = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "FP16ALT": torch.bfloat16
    }
    dtypes = []
    if len(user_dtypes) == 1:
        for i in range(num_param):
            dtypes.append(types_dict[user_dtypes[0]])
    elif len(user_dtypes) == num_param:
        for i in range(num_param):
            dtypes.append(types_dict[user_dtypes[i]])
    else:
        for i in range(len(user_dtypes)):
            dtypes.append(types_dict[user_dtypes[i]])
        if 'FP32' in user_dtypes:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP32"])
        elif 'FP16' in user_dtypes:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP16"])
        else:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP16ALT"])
    return dtypes


def dwt(input_size, coeff, levels, x, NC, Lo, Hi, mac_flag, vec_flag, dt):
    level_dim = input_size
    output_dim = get_outdim(coeff)
    x_test = copy.deepcopy(x)
    # generate outputs in FP32
    if vec_flag == "true":
        Lo = torch.flip(Lo, [0])
        Hi = torch.flip(Hi, [0])

    output_temp = torch.zeros(output_dim, dtype=dt)
    for i in range(levels):
        input_dim = level_dim
        level_dim = math.floor((level_dim + NC - 1) / 2)
        output_dim = output_dim - level_dim
        x_test, output_temp = dwt_step(x_test, output_temp, input_dim, output_dim, NC, Lo, Hi, mac_flag, vec_flag, dt)
    return output_temp


def save_data_into_hfile(x_test, output_dim, levels, NC, output_sig, Lo, Hi):
    g = open('config.h', 'w')
    g.write('\
#ifndef _CONF_\n\
#define _CONF_\n\
#include "config.h"\n\
#define WINDOW_LEN %s\n\
#define DWT_LEN      %s  //input\n\
#define DWT_LEN_OUT  %s   //output\n\
#define LEVELS       %s     //((int)ceil(log2(DWT_LEN)) //2^LEVELS=DWT_LEN if it executes all levels until max number of levels=log2(N)\n\
#define NC      %s\n\
        \n\n' % (x_test.shape[0], x_test.shape[0], output_dim, levels, NC))
    g.write('\
#endif \n')
    g.close()

    f = open('input_ch2_off.h', 'w')
    f.write('\
#ifndef INPUT_CH2_OFF_H\n\
#define INPUT_CH2_OFF_H\n\
#include "config.h"\n')

    write_matrix(x_test, 'Input_Signal', '', f, x_test.dtype)
    write_matrix(output_sig, 'ref', '', f, output_sig.dtype)
    f.write('\
#endif \n')
    f.close()

    # change filter type to torch
    LO_rec = torch.flip(Lo, [0])
    HI_rec = torch.flip(Hi, [0])

    # print("rec lo", Lo[::-1])
    # print("rec hi", Hi[::-1])
    f = open('kernels.def', 'w')
    if NC != 2:
        f.write('\
#ifdef VECTORIAL\n')
        write_matrix(LO_rec, 'Lo', 'NC', f, LO_rec.dtype)
        write_matrix(HI_rec, 'Hi', 'NC', f, HI_rec.dtype)

        f.write('\
#else\n')
        write_matrix(Lo, 'Lo', 'NC', f, Lo.dtype)
        write_matrix(Hi, 'Hi', 'NC', f, Hi.dtype)
        f.write('\
#endif \n')
        f.close()
    if NC == 2:
        f.write('\
DATA_LOCATION  DATA_TYPE R2_2 = 0.70710678118654752440f; //wfilters in Matlab\n')


def main():
    # read configs
    input_size, levels, family, bits, mac_flag, vec_flag = get_inital_config()

    # load input data
    X_test = load_data(input_size)

    # read filters
    Hi, Lo, NC, coeff = get_filters(X_test, levels, family)

    output_sig_fp32 = dwt(input_size, coeff, levels, X_test, NC, Lo, Hi, mac_flag="false", vec_flag="false",
                          dt=torch.float32)
    # set the data types based on the parser input
    datatypes = select_dtypes(bits, 3)
    dataset = load_data(input_size)
    x_test = matrix_init(dataset, dt=datatypes[0])
    lo = matrix_init(Lo, dt=datatypes[1])
    hi = matrix_init(Hi, dt=datatypes[1])

    output_sig = dwt(input_size, coeff, levels, x_test, NC, lo, hi, mac_flag, vec_flag, dt=datatypes[2])


    error_metric(output_sig_fp32, output_sig)

    save_data_into_hfile(x_test, get_outdim(coeff), levels, NC, output_sig, lo, hi)
    print("############################## Done! ###################################")


if __name__ == "__main__":
    main()
    pass
