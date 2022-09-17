import copy
import os

import torch
import numpy as np
import argparse


# Function to check if x is power of 2
def isPowerOfTwo(x):
    # First x in the below expression
    # is for the case when x is 0
    return x and (not (x & (x - 1)))


def bracewell_buneman(xarray, length, log2length):
    '''
    bracewell-buneman bit reversal function
    inputs: xarray is array; length is array length; log2length=log2(length).
    output: bit reversed array xarray.
    '''
    muplus = int((log2length + 1) // 2)
    mvar = 1
    reverse = torch.zeros(length, dtype=torch.int)
    upper_range = muplus + 1
    for _ in torch.arange(1, upper_range):
        for kvar in torch.arange(0, mvar):
            tvar = 2 * reverse[kvar]
            reverse[kvar] = tvar
            reverse[kvar + mvar] = tvar + 1
        mvar = mvar + mvar
    if log2length & 0x01:
        mvar = mvar // 2
    for qvar in torch.arange(1, mvar):
        nprime = qvar - mvar
        rprimeprime = reverse[qvar] * mvar
        for pvar in torch.arange(0, reverse[qvar]):
            nprime = nprime + mvar
            rprime = rprimeprime + reverse[pvar]
            temp = copy.deepcopy(xarray[nprime])  # .clone().detach()
            temp2 = xarray[rprime]
            xarray[nprime] = temp2
            xarray[rprime] = temp

    return xarray


def dif_fft0(x_real, x_imag, t_real, t_imag, log2length, dt, mac_flag="false", vec_flag="false"):
    '''
    radix-2 dif fft
    '''
    xarray_real = copy.deepcopy(x_real)
    xarray_imag = copy.deepcopy(x_imag)
    twiddle_real = copy.deepcopy(t_real)
    twiddle_imag = copy.deepcopy(t_imag)
    b_p = 1
    nvar_p = len(xarray_imag)
    twiddle_step_size = 1
    for _ in range(0, log2length):  # pass loop
        nvar_pp = nvar_p // 2
        base_e = 0

        for _ in range(0, b_p):  # block loop
            base_o = base_e + nvar_pp
            for nvar in range(0, nvar_pp):  # butterfly loop
                ovar_real = torch.zeros(1, dtype=dt)
                ovar_imag = torch.zeros(1, dtype=dt)
                evar_real = xarray_real[base_e + nvar] + xarray_real[base_o + nvar]
                evar_imag = xarray_imag[base_e + nvar] + xarray_imag[base_o + nvar]
                if nvar == 0:
                    ovar_real = xarray_real[base_e + nvar] - xarray_real[base_o + nvar]
                    ovar_imag = xarray_imag[base_e + nvar] - xarray_imag[base_o + nvar]
                else:
                    twiddle_factor = nvar * twiddle_step_size

                    ovar_r_temp = (xarray_real[base_e + nvar] - xarray_real[base_o + nvar])
                    ovar_i_temp = (xarray_imag[base_e + nvar] - xarray_imag[base_o + nvar])
                    twiddle_real_temp = twiddle_real[twiddle_factor]
                    twiddle_imag_temp = twiddle_imag[twiddle_factor]
                    if mac_flag == "true":
                        ## cast to fp32
                        ovar_r_temp = ovar_r_temp.type(torch.float32)
                        ovar_i_temp = ovar_i_temp.type(torch.float32)
                        twiddle_real_temp = twiddle_real_temp.type(torch.float32)
                        twiddle_imag_temp = twiddle_imag_temp.type(torch.float32)
                        ovar_real = ovar_real.type(torch.float32)
                        ovar_imag = ovar_imag.type(torch.float32)
                    ovar_real = -ovar_i_temp * twiddle_imag_temp
                    ovar_real = ovar_r_temp * twiddle_real_temp + ovar_real
                    ovar_imag = ovar_i_temp * twiddle_real_temp
                    ovar_imag = ovar_imag + ovar_r_temp * twiddle_imag_temp
                    if mac_flag == "true":
                        ## cast to fp16
                        ovar_real = ovar_real.type(dtype=dt)
                        ovar_imag = ovar_imag.type(dtype=dt)
                    '''ovar_real = (
                            ovar_r_temp * twiddle_real[twiddle_factor] - ovar_i_temp * twiddle_imag[twiddle_factor])
                    ovar_imag = (
                            ovar_r_temp * twiddle_imag[twiddle_factor] + ovar_i_temp * twiddle_real[twiddle_factor])'''
                xarray_real[base_e + nvar] = evar_real
                xarray_imag[base_e + nvar] = evar_imag
                xarray_real[base_o + nvar] = ovar_real
                xarray_imag[base_o + nvar] = ovar_imag

                # print("b_p %s nvar %s evar real %s evar im %s ovar rea %s ovar img %s"
                #       %(b_p, nvar,ovar_real.item(),ovar_imag.item(),evar_real.item(),evar_imag.item()))
                # print(b_p,"\t", ovar_real.item(),"\t",ovar_imag.item())
            base_e = base_e + nvar_p
        b_p = b_p * 2
        nvar_p = nvar_p // 2
        twiddle_step_size = 2 * twiddle_step_size
    xarray_real = bracewell_buneman(xarray_real, len(xarray_real), log2length)
    xarray_imag = bracewell_buneman(xarray_imag, len(xarray_imag), log2length)
    return xarray_real, xarray_imag


def write_matrix(matrix_to_write, matrix_to_write2, name, len, file_pointer, float_type):
    matrix_string = ''
    if 'ref' in name:
        file_pointer.write("PI_L2 Complex_type %s[%s] = {" % (name, len))
    else:
        file_pointer.write("DATA_LOCATION Complex_type  %s[%s] = {" % (name, len))

    if float_type == torch.float32:
        rem_part = ")"
    elif float_type == torch.float16:
        rem_part = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        rem_part = ", dtype=torch.bfloat16)"
    sz0 = matrix_to_write.shape[0]

    for i in range(sz0):
        file_pointer.write("{")
        matrix_string += str(matrix_to_write[i].item()).replace('tensor(', '').replace(rem_part, '')
        matrix_string += ','
        matrix_string += str(matrix_to_write2[i].item()).replace('tensor(', '').replace(rem_part, '')
        file_pointer.write("%s" % matrix_string)
        file_pointer.write("},\n")
        matrix_string = ''
    file_pointer.write("};\n")

def error_metric(ref, res):

    # calculate manually because metrics doesn't supprt bfloat16
    d = ref - res
    mse_f = torch.mean(d**2)
    mae_f = torch.mean(abs(d))
    rmse_f = torch.sqrt(mse_f)
    r2_f = 1-(torch.sum(d**2)/torch.sum((ref-torch.mean(ref))**2))
    print("MAE:",mae_f.item())
    print("MSE:", mse_f.item())
    print("RMSE:", rmse_f.item())
    print("R-Squared:", r2_f.item())
    rae = relative_absolute_error(ref, res)
    print("RAE is", rae.item())

def save_data_into_hfile(xtest_real, xtest_imag, output_real, output_imag):
    g = open('data_signal.h', 'w')
    g.write('\
    #ifndef FFT_DATA_H\n\
    #define FFT_DATA_H\n\
    #ifdef FABRIC\n\
    #define DATA_LOCATION\n\
    #else\n\
    #define DATA_LOCATION __attribute__((section(".data_l1")))\n\
    #endif\n\
    \n\n')

    write_matrix(xtest_real, xtest_imag, 'Input_Signal', '', g, xtest_real.dtype)

    g.write('\
    #ifndef SORT_OUTPUT\n\
    DATA_LOCATION Complex_type Buffer_Signal_Out[FFT_LEN_RADIX2];\n\
    #endif\n')
    g.write('\
    #endif\n')
    g.close()

    f = open('data_out.h', 'w')
    write_matrix(output_real, output_imag, 'ref', '', f, output_real.dtype)
    f.close()

    h = open('config.h', 'w')
    h.write('\
    #define LOG2_FFT_LEN    %s \n' % (int(np.log2(len(xtest_real)))))

    if not os.path.exists('../fft_radix8'):
        os.makedirs('../fft_radix8')
    h = open('../fft_radix8/config.h', 'w')
    h.write('\
    #define LOG2_FFT_LEN    %s \n' % (int(np.log2(len(xtest_real)))))


def matrix_init(IN, dt):
    # iterate through rows of IN
    temp = torch.zeros(len(IN), dtype=dt)
    for k in range(int(IN.shape[0])):
        temp[k] = IN[k]
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
            for i in range(len(user_dtypes), num_param + 1):
                dtypes.append(types_dict["FP32"])
        elif 'FP16' in user_dtypes:
            for i in range(len(user_dtypes), num_param + 1):
                dtypes.append(types_dict["FP16"])
        else:
            for i in range(len(user_dtypes), num_param + 1):
                dtypes.append(types_dict["FP16ALT"])
    return dtypes


def get_inital_config():
    # get input size and datatypes
    # python data_generator.py --input_size=256 --float_type=FP16 --MAC_flag=true --vec_flag=false
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=2048)
    parser.add_argument('--MAC_flag', default="false")
    parser.add_argument('--vec_flag', default="false")
    parser.add_argument('--float_type', default='FP32')  # input data,filters,output
    args = parser.parse_args()

    input_size = int(args.input_size)
    mac_flag = str(args.MAC_flag)
    vec_flag = str(args.vec_flag)
    bits = args.float_type.split(",")
    if not isPowerOfTwo(input_size):
        print("The input size should be in the form of power of 2")
        raise SystemExit
    return input_size, bits, mac_flag, vec_flag


def load_data(n):
    # Generating time data using arange function from numpy
    time = np.linspace(0.0, (n / 20) * np.pi, n)
    # Finding amplitude at each time
    amplitude = np.sin(time)
    dataset = torch.from_numpy(amplitude)
    # dataset = torch.from_numpy(np.asarray(Input_Signal[0:n]))

    Xtest_real = dataset[:].type(torch.float32)
    Xtest_imag = torch.zeros(n, dtype=torch.float32)
    return Xtest_real, Xtest_imag


def get_twiddle(n):
    Xtest_length = n  # length of the signal
    twiddle = torch.exp(-2j * torch.pi * torch.arange(0, 0.5, 1. / Xtest_length))
    Twiddle_real = twiddle.real.type(torch.float32)  # .clone().detach()
    Twiddle_imag = twiddle.imag.type(torch.float32)  # .clone().detach()

    return Twiddle_real, Twiddle_imag


def main():
    input_size, bits, mac_flag, vec_flag = get_inital_config()

    x_test_real, x_test_imag = load_data(input_size)

    twiddle_real, twiddle_imag = get_twiddle(input_size)
    # execute dwt in FP32
    ref_real, ref_imag = dif_fft0(x_test_real, x_test_imag, twiddle_real, twiddle_imag,
                                  int(np.log2(input_size)), dt=torch.float32,
                                  mac_flag="false", vec_flag="false")  # fft normalized magnitude

    datatypes = select_dtypes(bits, 3)
    xtest_real = matrix_init(x_test_real, dt=datatypes[0])
    xtest_imag = torch.zeros(input_size, dtype=xtest_real.dtype)

    twid_real = matrix_init(twiddle_real, dt=datatypes[1])
    twid_imag = matrix_init(twiddle_imag, dt=datatypes[1])

    output_real, output_imag = dif_fft0(xtest_real, xtest_imag, twid_real, twid_imag,
                                        int(np.log2(input_size)), dt=datatypes[2],
                                        mac_flag=mac_flag, vec_flag=vec_flag)  # fft normalized magnitude
    mse = mean_squared_error(ref_real, output_real)

    print("Error metrics of real part:")

    error_metric(ref_real, output_real)
    
    print("Error metrics of imag part:")

    error_metric(ref_imag, output_imag)

    save_data_into_hfile(xtest_real, xtest_imag, output_real, output_imag)
    print("############################## Done! ###################################")


if __name__ == "__main__":
    main()
    pass
