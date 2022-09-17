#!/bin/python3

import os
import argparse
import torch
import sys


def add_padding_to_image(img, padding_width: int):
    # Array of zeros of shape (img + padding_width)
    img_with_padding = torch.zeros((
        img.shape[0] + padding_width * 2,  # Multiply with two because we need padding on all sides
        img.shape[1] + padding_width * 2
    ))

    # Change the inner elements
    # For example, if img.shape = (224, 224), and img_with_padding.shape = (226, 226)
    # keep the pixel wide padding on all sides, but change the other values to be the same as img
    img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img

    return img_with_padding


def get_padding_width_per_side(kernel_size: int) -> int:
    # Simple integer division
    return kernel_size // 2  # p = [K/2]


def calculate_target_size(Img_Width, Kernel_Width, Stride, P):
    '''you can use this formula [(W−K+2P)/S]+1.
    W is the input volume - in your case 16
    K is the Kernel size - in your case 5
    P is the padding - 0 for valid and 1 is for same
    S is the stride - which you have not provided.'''
    W = Img_Width
    K = Kernel_Width
    S = Stride
    pixels = ((W - K + 2 * P) // S) + 1  # I added padding to input data, so I removed the 2P in this formula
    return pixels


def mac(img, kernel, dt, mac_flag, vec_flag, cast_flag):
    if vec_flag == "false":
        temp = torch.zeros(1, dtype=dt)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                a = img[i][j]
                b = kernel[i][j]
                if mac_flag == "true":
                    a = a.type(torch.float32)
                    b = b.type(torch.float32)
                    temp = temp.type(torch.float32)
                elif cast_flag == "true":
                    a = a.type(torch.float16)
                    b = b.type(torch.float16)
                temp += a * b
                if mac_flag == "true":
                    temp = temp.type(dt)
        return temp
    else:
        flag = True
        temp = torch.zeros(1, dtype=dt)
        temp1 = torch.zeros(1, dtype=dt)
        for i in range(img.shape[0]):
            for j in range(0, (img.shape[1] & 0xfffffffe), 2):
                a = img[i][j]
                a1 = img[i][j + 1]
                b = kernel[i][j]
                b1 = kernel[i][j + 1]
                if mac_flag == "true":
                    a = a.type(torch.float32)
                    b = b.type(torch.float32)
                    a1 = a1.type(torch.float32)
                    b1 = b1.type(torch.float32)
                    temp = temp.type(torch.float32)
                    temp1 = temp1.type(torch.float32)
                temp += a * b
                temp1 += a1 * b1
                if mac_flag == "true":
                    temp = temp.type(dt)
                    temp1 = temp1.type(dt)
        if img.shape[1] & 0x00000001:
            for i in range(img.shape[0]):
                a = img[i][img.shape[1] - 1]
                b = kernel[i][img.shape[1] - 1]
                if flag:  # temp
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp = temp.type(torch.float32)
                    temp += a * b
                    if mac_flag == "true":
                        temp = temp.type(dt)
                    flag = False
                else:  # temp1
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp1 = temp1.type(torch.float32)
                    temp1 += a * b
                    if mac_flag == "true":
                        temp1 = temp1.type(dt)
                    flag = True
        return temp + temp1


def convolve(img, kernel, out_width, dt, Stride, mac_flag, vec_flag, cast_flag):
    out_img = torch.zeros((out_width, out_width), dtype=dt)
    tgt_size = out_img.shape[0]
    # To simplify things
    k = kernel.shape[0]
    if vec_flag == "false":
        # Iterate over the rows
        for i in range(tgt_size):
            # Iterate over the columns
            for j in range(tgt_size):
                # img[i, j] = individual pixel value
                # Get the current matrix
                mat = img[i * Stride:i * Stride + k, j * Stride:j * Stride + k]
                # Apply the convolution - element-wise multiplication and summation of the result
                # Store the result to i-th row and j-th column of our convolved_img array
                out_img[i, j] = mac(mat, kernel, dt, mac_flag, vec_flag, cast_flag)
    else:  # based on the vectorized c code
        # Iterate over the columns
        for j in range(tgt_size):
            # Iterate over the rows
            for i in range(tgt_size):
                # img[i, j] = individual pixel value
                # Get the current matrix
                mat = img[i * Stride:i * Stride + k, j * Stride:j * Stride + k]
                # Apply the convolution - element-wise multiplication and summation of the result
                # Store the result to i-th row and j-th column of our convolved_img array
                out_img[i, j] = mac(mat, kernel, dt, mac_flag, vec_flag, cast_flag)
    return out_img


def relative_absolute_error(true, pred):
    true_mean = torch.mean(true)
    squared_error_num = torch.sum(torch.abs(true - pred))
    squared_error_den = torch.sum(torch.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss


def mean_squared_error(true, pred):
    squared_error = torch.square(true - pred)
    sum_squared_error = torch.sum(squared_error)
    size = true.size(dim=0) * true.size(dim=1)
    mse_loss = sum_squared_error / size
    return mse_loss


def matrix_init(IN, dt):
    # iterate through rows of IN
    temp = torch.zeros((IN.shape[0], IN.shape[1]), dtype=dt)
    # iterate through rows of IN
    for i in range(IN.shape[0]):
        # iterate through columns of IN
        for j in range(IN.shape[1]):
            temp[i][j] = IN[i][j] 
    return temp


def write_matrix(matrix_to_write, name, len, file_pointer, float_type):
    matrix_string = ''
    sz0 = matrix_to_write.size()[0]
    sz1 = matrix_to_write.size()[1]
    if 'Filter_Kern' in name:
        file_pointer.write("DATA_LOCATION FIL_TYPE %s[%s] = {" % (name, len))
    elif 'ref' in name:
        file_pointer.write("PI_L2 OUT_TYPE %s[%s] = {" % (name, len))
    else:
        file_pointer.write("DATA_LOCATION INP_TYPE %s[%s] = {" % (name, len))

    if float_type == torch.float32:
        name = ")"
    elif float_type == torch.float16:
        name = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        name = ", dtype=torch.bfloat16)"
    for i in range(sz0):
        for j in range(sz1):
            matrix_string += str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(name, '')
            matrix_string += ', '
    file_pointer.write("%s" % matrix_string)
    file_pointer.write("};\n")


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

def check_cast(datatypes):
    result = len(set(datatypes)) == 1  
    if result : #All Elements in List are Equal
        return "false"
    else: #All Elements in List are Not Equal
        if torch.float32 in datatypes:
            return "false"
        else:
            return "true"

def get_inital_config():
    # get arguments  and data format
    parser = argparse.ArgumentParser()
    parser.add_argument('--IMG_WIDTH')
    parser.add_argument('--FILT_WIN')
    parser.add_argument('--STRIDE', default=1)
    parser.add_argument('--PADDING', default='valid')
    parser.add_argument('--vec_flag', default="false")
    parser.add_argument('--MAC_flag', default="true")
    parser.add_argument('--float_type', default='FP32')
    args = parser.parse_args()

    IMG_WIDTH = int(args.IMG_WIDTH)
    FILT_WIN = int(args.FILT_WIN)
    STRIDE = int(args.STRIDE)
    PADDING = str(args.PADDING)
    mac_flag = str(args.MAC_flag)
    vec_flag = str(args.vec_flag)
    bits = args.float_type.split(",")
    if PADDING == 'same' and STRIDE != 1:
        sys.exit("ValueError: padding='same' is not supported for strided convolutions")
    return IMG_WIDTH, FILT_WIN, STRIDE, PADDING, bits, mac_flag, vec_flag


def save_data_into_hfile(OUT_WIDTH, IMG_WIDTH, FILT_WIN, STRIDE, res, filter_conv, input_conv):
    # Generate header file
    f = open('data.h', 'w')

    f.write('\
#ifndef _INPUT_IMAGE_ \n\
#define _INPUT_IMAGE_\n\
#pragma GCC diagnostic ignored "-Woverflow"\n\n')
    f.write('\
#define OUT_DIM %s\n\
#define OUT_ROW %s\n\
#define OUT_COL %s\n\
#define INP_COL %s\n\
#define STRIDE %s\n\
#define FILT_WIN %s\n\n' % (OUT_WIDTH * OUT_WIDTH, OUT_WIDTH, OUT_WIDTH, IMG_WIDTH, STRIDE, FILT_WIN))
    write_matrix(input_conv, 'In_Img', IMG_WIDTH * IMG_WIDTH, f, input_conv.dtype)
    write_matrix(filter_conv, 'Filter_Kern', FILT_WIN * FILT_WIN, f, filter_conv.dtype)
    write_matrix(res, 'ref', OUT_WIDTH * OUT_WIDTH, f, res.dtype)
    f.write('\
#endif \n')
    f.close()

    f = open('config.h', 'w')

    f.write('\
#define FILT_WIN %s \n\n' % FILT_WIN)
    f.close()

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
 


def main():
    IMG_WIDTH, FILT_WIN, STRIDE, PADDING, bits, mac_flag, vec_flag = get_inital_config()

    # Create reference matrices
    input_ref = torch.rand((IMG_WIDTH, IMG_WIDTH), dtype=torch.float32)
    filter_ref = torch.randn((FILT_WIN, FILT_WIN), dtype=torch.float32) * 4

    OUT_WIDTH = IMG_WIDTH
    if PADDING == 'same':
        pad = get_padding_width_per_side(kernel_size=FILT_WIN)
        input_ref = add_padding_to_image(img=input_ref, padding_width=pad)
        OUT_WIDTH = IMG_WIDTH
        IMG_WIDTH = input_ref.shape[0]

    elif PADDING == 'valid':
        P = 0
        OUT_WIDTH = calculate_target_size(
            Img_Width=IMG_WIDTH,
            Kernel_Width=FILT_WIN, Stride=STRIDE, P=P
        )
    # calculate reference output
    ref = convolve(img=input_ref, kernel=filter_ref, dt=torch.float32,
                   out_width=OUT_WIDTH, mac_flag="false", Stride=STRIDE, vec_flag="false", cast_flag="false")

    # set the data types based on the parser input
    datatypes = select_dtypes(bits, 3)
    cast_flag = check_cast(datatypes[0:2])
    input_conv = matrix_init(input_ref, dt=datatypes[0])
    filter_conv = matrix_init(filter_ref, dt=datatypes[1])
    res = convolve(img=input_conv, kernel=filter_conv, dt=datatypes[2],
                   out_width=OUT_WIDTH, Stride=STRIDE, mac_flag=mac_flag, vec_flag=vec_flag, cast_flag=cast_flag)


    error_metric(ref, res)


    save_data_into_hfile(OUT_WIDTH, IMG_WIDTH, FILT_WIN, STRIDE, res, filter_conv, input_conv)
    print("############################## Done! ###################################")
    return None


if __name__ == "__main__":
    main()
    pass
