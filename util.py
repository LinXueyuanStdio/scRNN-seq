from torch.utils.data import Dataset
from scipy.stats.stats import pearsonr
import scanpy.api as sc
import pandas as pd
import numpy as np


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


def calculate_pcc(arr1, arr2):
    PCC, _ = pearsonr(
        np.asarray(arr1).reshape(-1),
        np.asarray(arr2).reshape(-1))
    return PCC


def get_predict_and_true(output_data, simulated_csv_data_path, true_csv_data_path):
    a = normalization(pd.read_csv(simulated_csv_data_path).iloc[:, 1:])  # norm
    for i in range(2000):
        minmax = np.max(a.iloc[:, i])
        data = minmax_0_to_1(output_data[i][0], reverse=True, minmax=minmax)  # 把结果反归一化成norm状态（需要用到norm的最大值）
        a.iloc[:, i] = data  # 用结果覆盖原来的
    b = normalization(pd.read_csv(true_csv_data_path).iloc[:, 1:])

    # a,b 都是已norm状态
    return a, b
