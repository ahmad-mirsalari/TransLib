import pandas as pd
import os
import torch
import numpy as np
import argparse
import sys

'''
If you are working offline in local computer, just set folder_addr to '.' or absolute address of project folder in your computer, 
but if you are working in online, you may set folder_addr to the path of the project in the Internet.
'''
folder_addr = "."


def svmTrain(X, Y, C, kernelFunction, tol=1e-3, max_passes=5, args=()):
    """
    Trains an SVM classifier using a  simplified version of the SMO algorithm.
    Parameters
    ---------
    X : numpy ndarray
        (m x n) Matrix of training examples. Each row is a training example, and the
        jth column holds the jth feature.
    Y : numpy ndarray
        (m, ) A vector (1-D numpy array) containing 1 for positive examples and 0 for negative examples.
    C : float
        The standard SVM regularization parameter.
    kernelFunction : func
        A function handle which computes the kernel. The function should accept two vectors as
        inputs, and returns a scalar as output.
    tol : float, optional
        Tolerance value used for determining equality of floating point numbers.
    max_passes : int, optional
        Controls the number of iterations over the dataset (without changes to alpha)
        before the algorithm quits.
    args : tuple
        Extra arguments required for the kernel function, such as the sigma parameter for a
        Gaussian kernel.
    Returns
    -------
    model :
        The trained SVM model.
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
    Y = Y.astype(int)
    # Dataset size parameters
    m, n = X.shape

    passes = 0
    E = np.zeros(m)
    alphas = np.zeros(m)
    b = 0

    # Map 0 to -1
    Y[Y == 0] = -1

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    # gracefully will **not** do this)

    # We have implemented the optimized vectorized version of the Kernels here so
    # that the SVM training will run faster
    if kernelFunction == 'linearKernel':
        # Vectorized computation for the linear kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = np.dot(X, X.T)
    elif kernelFunction == 'gaussianKernel':
        # vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        '''X2 = np.sum(X**2, axis=1)
        K = X2 + X2[:, None] - 2 * np.dot(X, X.T)

        if len(args) > 0:
            K /= 2*args[0]**2

        K = np.exp(-K)'''
        K = np.zeros((X.shape[0], X.shape[0]))
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X):
                x1 = x1.ravel()
                x2 = x2.ravel()
                K[i, j] = np.exp(-np.sum(np.square(x1 - x2)) / (2 * (args[0] ** 2)))
    else:
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                K[i, j] = kernelFunction(X[i, :], X[j, :])
                K[j, i] = K[i, j]

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = b + np.sum(alphas * Y * K[:, i]) - Y[i]

            if (Y[i] * E[i] < -tol and alphas[i] < C) or (Y[i] * E[i] > tol and alphas[i] > 0):
                # select the alpha_j randomly
                j = np.random.choice(list(range(i)) + list(range(i + 1, m)), size=1)[0]

                E[j] = b + np.sum(alphas * Y * K[:, j]) - Y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if Y[i] == Y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]

                # objective function positive definite, there will be a minimum along the direction
                # of linear equality constrain, and eta will be greater than zero
                # we are actually computing -eta here (so we skip of eta >= 0)
                if eta >= 0:
                    continue

                alphas[j] -= Y[j] * (E[i] - E[j]) / eta
                alphas[j] = max(L, min(H, alphas[j]))

                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue
                alphas[i] += Y[i] * Y[j] * (alpha_j_old - alphas[j])

                b1 = b - E[i] - Y[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                     - Y[j] * (alphas[j] - alpha_j_old) * K[i, j]

                b2 = b - E[j] - Y[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                     - Y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    idx = alphas > 0
    model = {'X': X[idx, :],
             'y': Y[idx],
             'kernelFunction': kernelFunction,
             'b': b,
             'args': args,
             'alphas': alphas[idx],
             'w': np.dot(alphas * Y, X)}
    return model


def matrix_mult(Xs, Ys, dt, mac_flag, vec_flag, cast_flag, cast_to):
    Rs = torch.zeros((Xs.shape[0], Ys.shape[1]), dtype=dt)

    if vec_flag =="false":

        # iterate through rows of X
        for i in range(Xs.shape[0]):
            # iterate through columns of Y
            for j in range(Ys.shape[1]):
                temp = torch.tensor([0], dtype=dt)
                # iterate through rows of Y
                for k in range(Ys.shape[0]):
                    a = Xs[i][k]
                    b = Ys[k][j]
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp = temp.type(torch.float32)
                    if cast_flag == "true":
                        if cast_to == "FP16":
                            a = a.type(torch.float16)
                            b = b.type(torch.float16)
                        elif cast_to == "FP16ALT":
                            a = a.type(torch.bfloat16)
                            b = b.type(torch.bfloat16)
                    temp += a * b
                    if mac_flag == "true":
                        temp = temp.type(dt)

                Rs[i][j] = temp
    else:
        # iterate through rows of X
        for i in range(Xs.shape[0]):
            # iterate through columns of Y
            for j in range(Ys.shape[1]):
                temp = torch.tensor([0], dtype=dt)
                temp1 = torch.tensor([0], dtype=dt)
                # iterate through rows of Y
                for k in range(0, Ys.shape[0] & 0xfffffffe, 2):
                    a = Xs[i][k]
                    b = Ys[k][j]
                    a1 = Xs[i][k+1]
                    b1 = Ys[k+1][j]
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
                if Ys.shape[0] & 0x00000001:
                    a = Xs[i][Ys.shape[0]-1]
                    b = Ys[Ys.shape[0]-1][j]
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp = temp.type(torch.float32)
                    temp += a * b
                    if mac_flag == "true":
                        temp = temp.type(dt)
                Rs[i][j] = temp + temp1
    return Rs


def svmPredict(model, X, dt, mac_flag,vec_flag, cast_flag, cast_to):
    """
    Returns a vector of predictions using a trained SVM model.
    Parameters
    ----------
    model : dict
        The parameters of the trained svm model, as returned by the function svmTrain
    X : array_like
        A (m x n) matrix where each example is a row.
    Returns
    -------
    pred : array_like
        A (m,) sized vector of predictions {0, 1} values.
    """
    # check if we are getting a vector. If so, then assume we only need to do predictions
    # for a single example

    if X.ndim == 1:
        X = X[np.newaxis, :]

    pred = torch.zeros((X.shape[0], 1), dtype=dt)
    if model['kernelFunction'] == 'linearKernel':
        # we can use the weights and bias directly if working with the linear kernel
        # p = np.dot(X, model['w']) + model['b']

        pred = matrix_mult(X, model['w'], dt, mac_flag,vec_flag, cast_flag, cast_to)  + model['b']

    elif model['kernelFunction'] == 'gaussianKernel':
        expont = torch.tensor(2.71828182845904)
        K = torch.zeros((X.shape[0], model['X'].shape[0]), dtype=dt)
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(model['X']):
                x1 = x1.ravel()
                x2 = x2.ravel()
                subs = torch.square(x1 - x2)
                temp = torch.sum(subs, dtype=dt)
                temp = -temp * model['sigma']
                
                K[i, j] = expont ** temp  # torch.exp(-temp)
        pred = matrix_mult(K, model['w'], dt, mac_flag,vec_flag, cast_flag, cast_to) + model['b']

    pred[pred >= 0] = 1
    pred[pred < 0] = 0
    return pred


def write_matrix(matrix_to_write, name, len, file_pointer, float_type):
    matrix_string = ''

    if 'check' in name:
        file_pointer.write("PI_L2 static int %s[%s] = {" % (name, len))
    elif 'data_model' in name:
        file_pointer.write("DATA_LOCATION INP_TYPE %s[%s] = {" % (name, len))
    elif 'sv_coef' in name:
        file_pointer.write("DATA_LOCATION FIL_TYPE %s[%s] = {" % (name, len))
    elif 'bias' in name:
        file_pointer.write("DATA_LOCATION FIL_TYPE %s[%s] = {" % (name, len))
    else:
        file_pointer.write("DATA_LOCATION static INP_TYPE %s[%s] = {" % (name, len))

    if float_type == torch.float32:
        name = ")"
    elif float_type == torch.float16:
        name = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        name = ", dtype=torch.bfloat16)"
    sz0, sz1 = matrix_to_write.shape
    for i in range(sz0):
        for j in range(sz1):
            matrix_string += str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(name, '')
            matrix_string += ','
    file_pointer.write("%s" % matrix_string)
    file_pointer.write("};\n")


def matrix_init(IN, dt):
    temp = torch.zeros((IN.shape[0], IN.shape[1]), dtype=dt)
    # iterate through rows of IN
    for i in range(IN.shape[0]):
        # iterate through columns of IN
        for j in range(IN.shape[1]):
            temp[i][j] = IN[i][j]
    return temp


### A function to perform min-max scaling

def min_max_scaling(dataset, column):
    data = list(dataset[column])
    new_data = [(value - min(data)) / (max(data) - min(data)) for value in data]
    dataset[column] = new_data


from sklearn.model_selection import train_test_split


def read_dataset(name, input_size):
    if name == "bill":
        filepath = folder_addr + "/dataset/bill_authentication.csv"
        bankdata = pd.read_csv(filepath)
        X_df = bankdata.drop('Class', axis=1)
        y_df = bankdata['Class']

        X = X_df.values
        y = y_df.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=input_size)

    elif name == "cancer":
        filepath = folder_addr + '/dataset/breast-cancer.csv'
        dataset = pd.read_csv(filepath)
        # Changing the datatype of the column - dignosis in the dataset
        dataset.diagnosis = dataset.diagnosis.astype('category')
        # Drop the column - id
        dataset.drop(['id'], axis=1, inplace=True)
        dataset.head(10)
        # Scaling all the numerical columns
        columns = list(dataset.columns)
        numerical_columns = columns[1:]
        for each_column in numerical_columns:
            min_max_scaling(dataset, each_column)

        # Encoding the target column - diagnosis
        target_column = dataset['diagnosis']
        encoded_target = [0 if value == 'B' else 1 for value in target_column]

        dataset['diagnosis'] = encoded_target
        # Splitting the data into independent and dependent matrices - X and Y

        X = dataset.iloc[:, 1:].values
        Y = dataset.iloc[:, 0].values
        # Dividing the data into training and test sets in the ratio of 80:20

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=input_size, shuffle=True, random_state=27)
    else:
        sys.exit("ValueError: this dataset is not supported!!!!!")
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    x_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)
    x_test, y_test = x_test.type(torch.float32), y_test.type(torch.float32)
    return X_train, x_test, y_train, y_test


def get_inital_config():
    # get input size and datatypes
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', default='rbf')
    parser.add_argument('--dataset', default='cancer')
    parser.add_argument('--input_size', default=100)
    parser.add_argument('--MAC_flag', default="true")
    parser.add_argument('--vec_flag', default="false")
    parser.add_argument('--float_type', default='FP32')  # input data, x_ref, and (coef + bias)
    args = parser.parse_args()

    kernel_type = str(args.kernel)
    dataset = str(args.dataset)
    bits = args.float_type.split(",")
    input_size = int(args.input_size)
    mac_flag = str(args.MAC_flag)
    vec_flag = str(args.vec_flag)
    return kernel_type, input_size, bits, dataset, mac_flag, vec_flag


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

def save_data_into_hfile(kernel_type, x_test, x_ref, w, b, y_test, sigma=None, param=None):
    # Generate header files
    f = open('modelSVM.h', 'w')

    if kernel_type == "linear":
        f.write('\
#ifndef MODELSVM_H_ \n\
#define MODELSVM_H_\n\
#define KERNEL_TYPE_    0\n\
#define GAMMA1_         0.0f\n\
#define SVS_            %s\n\
# define COEF_DIM_      1\n\
# define F_DIM_             %s\n\
# define N_CLASS_       2\n\
# define N_DEC_VALUES_  %s\n\
#include <stdio.h>\n\
#include "defines.h"\n\n' % (x_test.shape[0], x_test.shape[1], x_test.shape[0]))
        write_matrix(x_test, 'data_model', 'SVS_*F_DIM_', f, x_test.dtype)
        write_matrix(w, 'sv_coef', 'COEF_DIM_*F_DIM_', f, w.dtype)
        write_matrix(b, 'bias', '1', f, b.dtype)
        write_matrix(x_ref, 'X_ref', '1', f, torch.float16)
        write_matrix(y_test, 'check_result', 'N_DEC_VALUES_', f, torch.float32)

    elif kernel_type == "rbf":
        f.write('\
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
#include "defines.h"\n\n' % (
            sigma.item(), x_test.shape[0], param['X'].shape[0], x_test.shape[1], x_test.shape[0]))
        write_matrix(x_test, 'data_model', 'SVS_*F_DIM_', f, x_test.dtype)
        write_matrix(x_ref, 'X_ref', 'COEF_DIM_*F_DIM_', f, x_ref.dtype)
        write_matrix(w, 'sv_coef', 'COEF_DIM_*F_DIM_', f, w.dtype)
        write_matrix(b, 'bias', '1', f, b.dtype)
        write_matrix(y_test, 'check_result', 'N_DEC_VALUES_', f, torch.float32)
    f.write('\
#endif \n')
    f.close()


def main():
    kernel_type, input_size, bits, dataset, mac_flag, vec_flag = get_inital_config()
    X_train, x_test, y_train, y_test = read_dataset(dataset, input_size)

    if kernel_type == "linear":
        C = 1
        print("Train is started (Linear)...")
        model = svmTrain(X_train, y_train, C, 'linearKernel')
        print("Train is finished...")

        # calculate the accuracy of FP32 data_type

        # read the weights and bias from trained model
        weight, bias = np.asarray(model['w']), np.asarray(model['b'])
        weight, bias = np.reshape(weight, (weight.shape[0], 1)), np.reshape(bias, (1, 1))

        # change numpy to tensor
        weight = torch.from_numpy(weight)
        bias = torch.from_numpy(bias)
        # change data type to fp32
        weight, bias = weight.type(torch.float32), bias.type(torch.float32)

        # create dictionary to pass to predict function
        param = {'kernelFunction': 'linearKernel',
                 'b': bias,
                 'w': weight}
        # predict x_test
        y_pred_fp32 = svmPredict(param, x_test, dt=torch.float32, mac_flag="false", vec_flag="false", cast_flag="false", cast_to="false")
        acc_fp32 = torch.sum(y_test == y_pred_fp32) / y_test.shape[0]
        print("Accuracy of Linear in FP32 data-type:", acc_fp32.item())

        # set the data types based on the parser input
        datatypes = select_dtypes(bits, 3)
        cast_flag = check_cast(datatypes[0:2])
        cast_to = "FP16ALT"
        # change the datatypes
        x_test = matrix_init(x_test, dt=datatypes[0])
        w = matrix_init(weight, dt=datatypes[1])
        b = matrix_init(bias, dt=datatypes[1])
        # predict y in the desired data type
        param = {'kernelFunction': 'linearKernel',
                 'b': b,
                 'w': w}
        y_pred = svmPredict(param, x_test, dt=datatypes[2], mac_flag=mac_flag, vec_flag=vec_flag, cast_flag=cast_flag, cast_to=cast_to)
        acc_fp = torch.sum(y_test == y_pred) / y_test.shape[0]
        print("Accuracy of Linear in the desired data-type:", acc_fp.item())
        x_ref = torch.zeros((1, 1), dtype=torch.float16)
        save_data_into_hfile(kernel_type, x_test, x_ref, w, b, y_test)
        print("############################## Done! ###################################")
        return None
    elif kernel_type == "rbf":  ############## RBF kernel

        C = 2
        sigma = torch.tensor(4.4721, dtype=torch.float16)
        # sigma = (2 * (sigma ** 2))
        print("Train is started (RBF)...")

        model = svmTrain(X_train, y_train, C, 'gaussianKernel', args=(sigma,))
        print("Train is finished...")

        # read data from trained model
        weight, bias = np.asarray(model['alphas'] * model['y']), np.asarray(model['b'])
        weight, bias = np.reshape(weight, (weight.shape[0], 1)), np.reshape(bias, (1, 1))
        x_ref = np.asarray(model['X'])

        # change numpy array to tensor
        weight = torch.from_numpy(weight)
        bias = torch.from_numpy(bias)
        x_ref = torch.from_numpy(x_ref)
        x_ref, weight, bias = x_ref.type(torch.float32), weight.type(torch.float32), bias.type(torch.float32)
        sigma = 1 / (2 * (model['args'][0] ** 2))
        # create dictionary to pass to predict function
        param = {'kernelFunction': 'gaussianKernel',
                 'X': x_ref,
                 'b': bias,
                 'w': weight,
                 'sigma': sigma}
        y_pred_fp32 = svmPredict(param, x_test, dt=torch.float32, mac_flag="false",vec_flag="false", cast_flag="false", cast_to="false")
        acc_fp32 = torch.sum(y_test == y_pred_fp32) / y_test.shape[0]
        print("Accuracy of Linear in FP32 data-type:", acc_fp32.item())

        # set the data types based on the parser input
        datatypes = select_dtypes(bits, 3)
        cast_flag = check_cast(datatypes[0:2])
        cast_to = "FP16ALT"
        x_test = matrix_init(x_test, dt=datatypes[0])
        x_ref = matrix_init(x_ref, dt=datatypes[0])
        w = matrix_init(weight, dt=datatypes[1])
        b = matrix_init(bias, dt=datatypes[1])

        sigma = 1 / (2 * (model['args'][0] ** 2))
        param = {'kernelFunction': 'gaussianKernel',
                 'X': x_ref,
                 'b': b,
                 'w': w,
                 'sigma': sigma}

        y_pred = svmPredict(param, x_test, dt=datatypes[2], mac_flag=mac_flag, vec_flag=vec_flag, cast_flag=cast_flag, cast_to=cast_to)
        acc_fp = torch.sum(y_test == y_pred) / y_test.shape[0]

        print("Accuracy of RBF in the desired data-type:", acc_fp.item())
        save_data_into_hfile(kernel_type, x_test, x_ref, w, b, y_test, sigma=sigma, param=param)
        print("############################## Done! ###################################")
        return None
    else:
        print("kernel type is invalid\n")


if __name__ == "__main__":
    main()
    pass
