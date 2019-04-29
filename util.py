from torch.utils.data import Dataset
from torch.autograd import Variable
from scipy.stats.stats import pearsonr
import scanpy.api as sc
import pandas as pd
import numpy as np
import os
import torch

def to_img(x):
    x = x.view(x.size(0), 1, 100, 50)
    return x


def normalization(express_data):
    adata = sc.AnnData(express_data.T.values)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    return pd.DataFrame(adata.X.T, columns=express_data.columns.tolist(), index=express_data.index.tolist())


def norm(x, reverse=False):
    if reverse:
        y = np.power(10, x) - 1.01
        y = np.around(y).astype(np.int32)
        return y
    else:
        return np.log10(x + 1.01)


def minmax_0_to_1(x, reverse=False, minmax=1):
    if reverse:
        # x -> [0, 1]
        return x * minmax
        # minmax_x -> [0, 6]
    else:
        # norm_x -> [0, 6]
        return x / minmax
        # minmax_x -> [0, 1]


class LinearPackDataset(Dataset):
    '''
    每一个 Item 是 (1, 5000) 的向量，元素范围[0, ~6]，已 normalization
    '''

    def __init__(self, simulated_csv_data_path, true_csv_data_path, transform=None):
        self.simulated_csv_data = normalization(pd.read_csv(simulated_csv_data_path).iloc[:, 1:])
        self.true_csv_data_path = normalization(pd.read_csv(true_csv_data_path).iloc[:, 1:])
        self.transform = transform

    def __len__(self):
        return len(self.simulated_csv_data.columns)

    def __getitem__(self, index):
        a_column_of_simulated_data = self.simulated_csv_data.iloc[:, index]
        a_column_of_true_data = self.true_csv_data_path.iloc[:, index]

        a_column_of_simulated_data = np.asarray(a_column_of_simulated_data).reshape(1, -1)  # (1, 5000)
        a_column_of_true_data = np.asarray(a_column_of_true_data).reshape(1, -1)

        if self.transform is not None:
            a_column_of_simulated_data = self.transform(a_column_of_simulated_data)
            a_column_of_true_data = self.transform(a_column_of_true_data)

        simulated_true_pack = (a_column_of_simulated_data, a_column_of_true_data)

        return simulated_true_pack


class Conv2d_100x50_Dataset(Dataset):
    '''
    每一个 Item 是 (100, 50) 的矩阵
    '''

    def __init__(self, simulated_csv_data_path, true_csv_data_path, transform=None):
        self.simulated_csv_data = normalization(pd.read_csv(simulated_csv_data_path))
        self.true_csv_data_path = normalization(pd.read_csv(true_csv_data_path))
        self.transform = transform

    def __len__(self):
        return len(self.simulated_csv_data.columns)

    def __getitem__(self, index):
        a_column_of_simulated_data = self.simulated_csv_data.iloc[:, index]
        a_column_of_true_data = self.true_csv_data_path.iloc[:, index]

        a_column_of_simulated_data = np.asarray(a_column_of_simulated_data).reshape(1, 100, 50)
        a_column_of_true_data = np.asarray(a_column_of_true_data).reshape(1, 100, 50)

        a_column_of_simulated_data = a_column_of_simulated_data / np.max(a_column_of_simulated_data) # 根据最大值来归一化
        a_column_of_true_data = a_column_of_true_data / np.max(a_column_of_true_data)

        if self.transform is not None:
            a_column_of_simulated_data = self.transform(a_column_of_simulated_data)
            a_column_of_true_data = self.transform(a_column_of_true_data)

        simulated_true_pack = (a_column_of_simulated_data, a_column_of_true_data)
        return simulated_true_pack


def calculate_pcc(arr1, arr2):
    PCC, _ = pearsonr(
        np.asarray(arr1).reshape(-1),
        np.asarray(arr2).reshape(-1))
    return PCC


def get_predict_norm_dataframe(output_data, simulated_csv_data_path):
    a = normalization(pd.read_csv(simulated_csv_data_path).iloc[:, 1:])  # norm
    for i in range(2000):
        minmax = np.max(a.iloc[:, i])
        data = minmax_0_to_1(output_data[i][0], reverse=True, minmax=minmax)  # 把结果反归一化成norm状态（需要用到norm的最大值）
        a.iloc[:, i] = data  # 用结果覆盖原来的

    # a 是已norm状态
    return a


def calculate_pcc_mse(output, noisy_data, MSE_loss):
    mse = MSE_loss(output, noisy_data).data
    np1 = output.cpu().detach().numpy().reshape(-1)
    np2 = noisy_data.cpu().detach().numpy().reshape(-1)
    PCC, _ = pearsonr(np1, np2)

    return PCC, mse


def minmax_noisy_data(noisy_data, device):
    noisy_data = Variable(noisy_data).float().to(device)
    noisy_data = minmax_0_to_1(noisy_data, False, torch.max(noisy_data))
    return noisy_data


class OutputManager:
    def __init__(self,
                 simulated_csv_data_path="./data/counts_simulated_dataset1_dropout0.05.csv",
                 true_csv_data_path="./data/true_counts_simulated_dataset1_dropout0.05.csv",
                 model_filename="model_dropout0.05.pth",
                 output_path="./output",
                 model_name="LinearAutoEncoder",
                 dropout="0.05"):
        """
        simulated_csv_data_path
        true_csv_data_path
        output_path 输出目录
        model_filename 模型文件名
        model_name 模型名字
        dropout 当前处理的dropout
        """
        self.simulated_csv_data_path = simulated_csv_data_path
        self.true_csv_data_path = true_csv_data_path
        self.output_path = output_path
        self.model_filename = model_filename
        self.model_name = model_name
        self.dropout = dropout

        self.model_all_save_to = output_path + "/" + model_name + "/"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(self.model_all_save_to):
            os.mkdir(self.model_all_save_to)

    def predict_file_path(self, PCC, MSE, prefix="predict"):
        filename = prefix + "_PCC_{:.4f}_MSE_{:.8f}_".format(PCC, MSE) + self.simulated_csv_data_path[7:]
        return self.model_all_save_to + filename

    def model_file_path(self):
        return self.model_all_save_to + self.model_filename


def save_output_data(output, noisy_data, MSE_loss, output_manager):
    # 1. get MSE
    mse = MSE_loss(output, noisy_data).data

    # 2. get PCC
    predict_df = get_predict_norm_dataframe(output.data.numpy(), output_manager.simulated_csv_data_path)
    true_df = normalization(pd.read_csv(output_manager.true_csv_data_path).iloc[:, 1:])
    pcc = calculate_pcc(predict_df.iloc[:, 1:], true_df.iloc[:, 1:])

    # 3. save as '.csv'
    predict_file_path = output_manager.predict_file_path(pcc, mse)
    predict_df.to_csv(predict_file_path, index=0)
    print("save prediction to " + predict_file_path)


def predict_one_by_one(predict_function):
    predict_function(
        "./data/counts_simulated_dataset1_dropout0.05.csv",
        "./data/true_counts_simulated_dataset1_dropout0.05.csv",
        "model_dropout0.05.pth",
        "0.05"
    )
    predict_function(
        "./data/counts_simulated_dataset1_dropout0.10.csv",
        "./data/true_counts_simulated_dataset1_dropout0.10.csv",
        "model_dropout0.10.pth",
        "0.10"
    )
    predict_function(
        "./data/counts_simulated_dataset1_dropout0.15.csv",
        "./data/true_counts_simulated_dataset1_dropout0.15.csv",
        "model_dropout0.15.pth",
        "0.15"
    )
    predict_function(
        "./data/counts_simulated_dataset1_dropout0.20.csv",
        "./data/true_counts_simulated_dataset1_dropout0.20.csv",
        "model_dropout0.20.pth",
        "0.20"
    )
    predict_function(
        "./data/counts_simulated_dataset1_dropout0.25.csv",
        "./data/true_counts_simulated_dataset1_dropout0.25.csv",
        "model_dropout0.25.pth",
        "0.25"
    )
