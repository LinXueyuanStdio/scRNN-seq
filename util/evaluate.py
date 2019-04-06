from util.data_process import normalization, minmax_0_to_1
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np


def calculate_pcc(arr1, arr2):
    PCC, _ = pearsonr(
        np.asarray(arr1).reshape(2000*5000),
        np.asarray(arr2).reshape(2000*5000))
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
