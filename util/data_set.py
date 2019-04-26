
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from util.data_process import normalization


class SimulatedDataset(Dataset):
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
