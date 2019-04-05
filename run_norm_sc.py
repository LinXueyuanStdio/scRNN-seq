import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from scipy.stats.stats import pearsonr
from Progbar import Progbar

import scanpy.api as sc
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def to_img(x):
    x = x.view(x.size(0), 1, 100, 50)
    return x


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


def reset_raw_from_norm(norm_x):
    return norm(
        minmax_0_to_1(
            minmax_0_to_1(norm_x, minmax=np.max(norm_x)), True, np.max(norm_x)), True)


def get_predict_and_true(output_data, simulated_csv_data_path, true_csv_data_path):
    a = normalization(pd.read_csv(simulated_csv_data_path).iloc[:, 1:]) # norm
    for i in range(2000):
        minmax = np.max(a.iloc[:, i])
        data = minmax_0_to_1(output_data[i][0], reverse=True, minmax=minmax) # 把结果反归一化成norm状态（需要用到norm的最大值）
        a.iloc[:, i] = data # 用结果覆盖原来的
    b = pd.read_csv(true_csv_data_path)

    # a,b 都是已norm状态
    return a, b


def calculate_pcc(arr1, arr2):
    PCC, _ = pearsonr(
        np.asarray(arr1).reshape(2000*5000),
        np.asarray(arr2).reshape(2000*5000))
    return PCC

def normalization(express_data):
    adata = sc.AnnData(express_data.T.values)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    return pd.DataFrame(adata.X.T, columns=express_data.columns.tolist(), index=express_data.index.tolist())

num_epochs = 10
batch_size = 50
learning_rate = 1e-3
prefix = "BCE_norm"


class SimulatedDataset(Dataset):
    '''
    每一个 Item 是 (1, 5000) 的向量
    transform 默认为归一化
    '''

    def __init__(self, simulated_csv_data_path, true_csv_data_path, transform=norm):
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


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 5000),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def predict(simulated_csv_data_path="./data/counts_simulated_dataset1_dropout0.05.csv",
            true_csv_data_path="./data/true_counts_simulated_dataset1_dropout0.05.csv",
            save_model_filename="./model_dropout0.05.pth", num_epochs=10):
    dataset = SimulatedDataset(simulated_csv_data_path, true_csv_data_path)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=3)
    model = AutoEncoder().to(device)
    MSE_loss = nn.MSELoss()
    BCE_Loss = nn.BCELoss()
    criterion = MSE_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    if os.path.exists(save_model_filename):
        model.load_state_dict(torch.load(save_model_filename, "cpu"))
    else:
        model.train()
        for epoch in range(num_epochs):
            print('epoch [{}/{}]'.format(epoch + 1, num_epochs))
            prog = Progbar(len(dataloader))
            for i, data in enumerate(dataloader):
                (noisy_data, true_data) = data
                noisy_data = minmax_0_to_1(noisy_data, False, torch.max(noisy_data))
                true_data = minmax_0_to_1(true_data, False, torch.max(true_data))
                noisy_data = Variable(noisy_data).float().to(device)
                true_data = Variable(true_data).float().to(device)
                # ===================forward=====================
                output = model(noisy_data)
                loss = criterion(output, true_data)
                mse = MSE_loss(output, true_data).data
                np1 = output.cpu().detach().numpy().reshape(-1)
                np2 = true_data.cpu().detach().numpy().reshape(-1)
                PCC, p_value = pearsonr(np1, np2)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # =====================log=======================
                prog.update(i + 1, [("loss", loss.item()), ("MSE_loss", mse), ("PCC", PCC), ("p-value", p_value)])
        torch.save(model.state_dict(), save_model_filename)

    model.eval()
    dataloader2 = DataLoader(dataset, batch_size=2000, shuffle=True, num_workers=3)
    for data in dataloader2:
        (noisy_data, true_data) = data
        noisy_data = Variable(noisy_data).float().to(device)
        true_data = Variable(true_data).float().to(device)
        noisy_data = minmax_0_to_1(noisy_data, False, torch.max(noisy_data))
        true_data = minmax_0_to_1(true_data, False, torch.max(true_data))
        # ===================forward=====================
        output = model(noisy_data)
        loss = criterion(output, true_data)
        mse = MSE_loss(output, true_data).data
        output_data = output.data.numpy()

        predict_df, true_df = get_predict_and_true(output_data, simulated_csv_data_path, true_csv_data_path)
        pcc = calculate_pcc(predict_df.iloc[:, 1:], true_df.iloc[:, 1:])

        print("predict PCC:{:.4f} MSE:{:.8f}".format(pcc, mse))

        # filepath = "./data/"+prefix+"_predict_PCC_{:.4f}_MSE_{:.8f}_".format(pcc, mse)+simulated_csv_data_path[7:]
        # predict_df.to_csv(filepath, index=0)
        break  # 只有一个 batch, 一次全拿出来了，不会有第二个


predict(
    "./data/counts_simulated_dataset1_dropout0.05.csv",
    "./data/true_counts_simulated_dataset1_dropout0.05.csv",
    "./"+prefix+"_model_dropout0.05.pth"
)
predict(
    "./data/counts_simulated_dataset1_dropout0.10.csv",
    "./data/true_counts_simulated_dataset1_dropout0.10.csv",
    "./"+prefix+"_model_dropout0.10.pth"
)
predict(
    "./data/counts_simulated_dataset1_dropout0.15.csv",
    "./data/true_counts_simulated_dataset1_dropout0.15.csv",
    "./"+prefix+"_model_dropout0.15.pth"
)
predict(
    "./data/counts_simulated_dataset1_dropout0.20.csv",
    "./data/true_counts_simulated_dataset1_dropout0.20.csv",
    "./"+prefix+"_model_dropout0.20.pth"
)
predict(
    "./data/counts_simulated_dataset1_dropout0.25.csv",
    "./data/true_counts_simulated_dataset1_dropout0.25.csv",
    "./"+prefix+"_model_dropout0.25.pth"
)
