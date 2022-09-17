import pandas as pd
import os
import torch
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

'''
If you are working offline in local computer, just set folder_addr to '.' or absolute address of project folder in your computer, 
but if you are working in online, you may set folder_addr to the path of the project in the Internet.
'''
folder_addr = "."


class KMeansClustering:
    def __init__(self, X, num_clusters, dt, mac_flag, vec_flag):
        self.K = num_clusters
        self.max_iterations = 5000
        self.plot_figure = False
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]
        self.dt = dt
        self.mac_flag = mac_flag
        self.vec_flag = vec_flag

    def initialize_random_centroids(self, X):
        centroids = torch.zeros((self.K, self.num_features), dtype=self.dt)

        for k in range(self.K):
            centroid = X[k]  # np.random.random(self.num_features,)#X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid
        return centroids

    def mysum(self, point, centroids, mac_flag, vec_flag):

        if vec_flag == "false":
            diff = torch.zeros(self.K, dtype=self.dt)
            for i in range(centroids.shape[0]):
                temp = torch.zeros(1, dtype=self.dt)
                for j in range(centroids.shape[1]):
                    a = point[j]
                    b = centroids[i][j]
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp = temp.type(torch.float32)
                    temp += (a - b) ** 2
                    if mac_flag == "true":
                        temp = temp.type(torch.float16)
                #temp = temp ** (1 / 2)
                diff[i] = temp
        else:
            diff = torch.zeros(self.K, dtype=self.dt)
            for i in range(centroids.shape[0]):
                temp = torch.zeros(1, dtype=self.dt)
                temp1 = torch.zeros(1, dtype=self.dt)
                for j in range(0, centroids.shape[1] & 0xfffffffe, 2):
                    a = point[j]
                    b = centroids[i][j]
                    a1 = point[j + 1]
                    b1 = centroids[i][j + 1]
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        a1 = a1.type(torch.float32)
                        b1 = b1.type(torch.float32)
                        temp = temp.type(torch.float32)
                        temp1 = temp1.type(torch.float32)
                    temp += (a - b) ** 2
                    temp1 += (a1 - b1) ** 2
                    if mac_flag == "true":
                        temp = temp.type(self.dt)
                        temp1 = temp1.type(self.dt)
                if centroids.shape[1] & 0x00000001:
                    a = point[centroids.shape[1] - 1]
                    b = centroids[i][centroids.shape[1] - 1]
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp = temp.type(torch.float32)
                    temp += (a - b) ** 2
                    if mac_flag == "true":
                        temp = temp.type(self.dt)

                #temp = (temp + temp1) ** (1 / 2)
                diff[i] = temp
        return diff

    def mymean(self, inp):
        mean = torch.zeros(inp.shape[1])
        for i in range(inp.shape[1]):
            temp = torch.zeros(1, dtype=self.dt)
            for j in range(inp.shape[0]):
                temp += inp[j][i]
            mean[i] = temp / inp.shape[0]
        return mean

    def create_clusters(self, X, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = [[] for _ in range(self.K)]
        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            '''closest_centroid = torch.argmin(
                ((torch.sum((point - centroids) ** 2, dim=1)) ** (1 / 2))
            )'''
            closest_centroid = torch.argmin(self.mysum(point, centroids, self.mac_flag, self.vec_flag))
            clusters[closest_centroid].append(point_idx)
        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = torch.zeros((self.K, self.num_features), dtype=self.dt)
        for idx, cluster in enumerate(clusters):
            # new_centroid = torch.mean(X[cluster], dim=0)
            new_centroid = self.mymean(X[cluster])
            centroids[idx] = new_centroid

        return centroids

    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def fit(self, X):
        centroids = self.initialize_random_centroids(X)
        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            diff = centroids - previous_centroids
            if not diff.any():
                print("Termination criterion satisfied")
                break

        # Get label predictions
        y_pred = self.predict_cluster(clusters, X)

        return centroids, y_pred


def write_matrix(matrix_to_write, name, len, file_pointer, float_type):
    matrix_string = ''
    if 'check' in name:
        file_pointer.write("PI_L2 float %s[] = {" % (name))
    else:
        file_pointer.write("DATA_LOCATION FLOAT  %s[N_OBJECTS][N_COORDS] = {" % (name))

    if float_type == torch.float32:
        rem_part = ")"
    elif float_type == torch.float16:
        rem_part = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        rem_part = ", dtype=torch.bfloat16)"
    sz0, sz1 = matrix_to_write.shape

    if 'check' in name:
        for i in range(sz0):
            for j in range(sz1):
                if float_type == torch.float32:
                    matrix_string += str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(rem_part, '')
                else:
                    matrix_string += str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(rem_part, '')
                matrix_string += ','
        file_pointer.write("%s" % matrix_string)
        file_pointer.write("};\n")
    else:
        for i in range(sz0):
            file_pointer.write("{")
            for j in range(sz1):
                if float_type == torch.float32:
                    matrix_string = str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(rem_part, '')
                else:
                    matrix_string = str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(rem_part, '')
                matrix_string += ','
                file_pointer.write("%s" % matrix_string)
            file_pointer.write("},\n")

        file_pointer.write("};\n")


def matrix_init(IN, dt):
    temp = torch.zeros((IN.shape[0], IN.shape[1]), dtype=dt)
    # iterate through rows of IN
    for i in range(IN.shape[0]):
        # iterate through columns of IN
        for j in range(IN.shape[1]):
            temp[i][j] = IN[i][j]
    return temp


def mean_squared_error(true, pred):
    squared_error = torch.square(true - pred)
    sum_squared_error = torch.sum(squared_error)
    size = true.size(dim=0) * true.size(dim=1)
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
    print("MAE:",mae_f.item())
    print("MSE:", mse_f.item())
    print("RMSE:", rmse_f.item())
    print("R-Squared:", r2_f.item())
    rae = relative_absolute_error(ref, res)
    print("RAE is", rae.item())

def get_inital_config():
    # get input size and datatypes
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=128)
    parser.add_argument('--features', default=8)
    parser.add_argument('--num_clusters', default=9)
    parser.add_argument('--MAC_flag', default="true")
    parser.add_argument('--vec_flag', default="false")
    parser.add_argument('--float_type', default='FP32')
    args = parser.parse_args()

    bits = args.float_type.split(",")
    input_size = int(args.input_size)
    num_clusters = int(args.num_clusters)
    num_features = int(args.features)
    mac_flag = str(args.MAC_flag)
    vec_flag = str(args.vec_flag)
    if num_features < 1 or num_features > 8:
        sys.exit("ValueError: num_features is not supported for this dataset")

    return input_size, num_features, num_clusters, bits, mac_flag, vec_flag


def load_data(input_size, num_features):
    # load data
    filepath = folder_addr + '/dataset/data.csv'
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[0:input_size, 0:num_features].values * 3
    input_fp32 = torch.from_numpy(X)
    X_test = input_fp32.type(torch.float32)
    return X_test


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


def save_data_into_hfile(x_test, num_clusters, centers):
    f = open('data_def.h', 'w')
    f.write('\
# define N_CLUSTERS %s\n\
# define N_OBJECTS %s\n\
# define N_COORDS %s\n\n' % (num_clusters, x_test.shape[0], x_test.shape[1]))
    write_matrix(x_test, 'objects', '', f, x_test.dtype)
    f.close()

    g = open('out_ref.h', 'w')
    g.write('\
#ifndef __CHECKSUM_H__ \n\
#define __CHECKSUM_H__\n\
#include "config.h"\n\
#include "pmsis.h"\n\n')
    write_matrix(centers, 'check', '', g, centers.dtype)
    g.write('\
#endif \n')
    g.close()


def main():
    input_size, num_features, num_clusters, bits, mac_flag, vec_flag = get_inital_config()

    X_test = load_data(input_size, num_features)

    print("Kmeans is started in FP32 data type")
    Kmeans = KMeansClustering(X_test, num_clusters, dt=torch.float32, mac_flag=mac_flag, vec_flag=vec_flag)
    cent_fp32, y_pred_fp32 = Kmeans.fit(X_test)

    # set the data types based on the parser input
    datatypes = select_dtypes(bits, 2)
    # change the datatypes
    x_test = matrix_init(X_test, dt=datatypes[0])

    print("Kmeans is started in the desired data type", datatypes[1])
    Kmeans = KMeansClustering(x_test, num_clusters, dt=datatypes[1], mac_flag=mac_flag, vec_flag=vec_flag)
    cent_des, y_pred = Kmeans.fit(x_test)
    # print("Centroids in the desired data-type:", cent_type.numpy())

    error_metric(cent_fp32,cent_des)

    save_data_into_hfile(x_test, num_clusters, cent_des)
    print("############################## Done! ###################################")


if __name__ == "__main__":
    main()
    pass
